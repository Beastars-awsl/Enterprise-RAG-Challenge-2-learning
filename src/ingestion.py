"""
入库阶段（离线建库）：chunked_reports -> 向量索引(.faiss) / BM25 索引(.pkl)。

职责定位:
    - VectorDBIngestor: 把每个报告的 chunks 逐条 embedding 后建成 FAISS 索引。
        契约: chunk 数组下标 == FAISS 索引行号；查询端按行号取回 chunk（retrieval.py），
        因此本阶段与切分阶段的写入顺序必须一致，且同一公司只有一份库文件。
    - BM25Ingestor: 同源构建 BM25 词袋索引（当前问答链路未启用，属保留检索器）。

数据流位置:
    输入: databases(_ser_tab)/chunked_reports/{sha1}.json（text_splitter 产物）；
    输出: databases(_ser_tab)/vector_dbs/{sha1}.faiss 与 bm25_dbs/{sha1}.pkl
          —— 文件主名即报告的 sha1_name，检索端按「同名配对」把文档 JSON 与索引对上。

核心依赖与副作用:
    - embedding 走 OpenAI text-embedding-3-large（网络 I/O；每批最多 1024 条文本，
      失败自动重试 2 次、间隔 20s，最终失败抛错中断建库 —— 残缺索引宁可重跑）；
    - 排序/距离语义: 使用 FAISS IndexFlatIP（内积）。OpenAI embedding 输出单位化向量，
      内积 == 余弦相似度；换 embedding 模型时必须先做 L2 归一化再入库（见 _create_vector_db）；
    - 纯顺序处理（tqdm 进度），无并发、无全局状态。
"""
import os
import json
import pickle
from typing import List, Union
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from tenacity import retry, wait_fixed, stop_after_attempt


class BM25Ingestor:
    """BM25 词袋索引构建器（每份报告一个独立 .pkl 索引）。

    分词口径与检索端必须一致: rank_bm25 只做空白切分（不处理词形/标点），
    BM25Retriever.retrieve_by_company_name 查询侧同样 query.split()，
    两边改动任何一侧都会造成评分失真。
    """

    def __init__(self):
        pass

    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """根据 chunk 文本列表构建 BM25Okapi 索引。

        Args:
            chunks: 报告全部 chunk 的 text（顺序无关紧要，BM25 与序号无耦合）

        Returns:
            可直接 get_scores 查询的 BM25Okapi 实例（序列化前必须先与文档配对保存）。
        """
        tokenized_chunks = [chunk.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """逐报告建 BM25 索引并落盘（索引按 chunk 文本数组顺序与文档 chunks 对齐）。

        Args:
            all_reports_dir: chunked_reports 目录
            output_dir: bm25_dbs 目录（自动创建；每份报告输出 {sha1}.pkl）

        Notes:
            索引文件用 pickle 落盘 —— 依赖 rank_bm25 库版本稳定，升级依赖后旧 .pkl
            可能读不了，届时需重建（vector 的 .faiss 无此问题）。
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_report_dir.glob("*.json"))

        for report_path in tqdm(all_report_paths, desc="Processing reports for BM25"):
            # Load the report
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                
            # Extract text chunks and create BM25 index
            text_chunks = [chunk['text'] for chunk in report_data['content']['chunks']]
            bm25_index = self.create_bm25_index(text_chunks)
            
            # Save BM25 index
            sha1_name = report_data["metainfo"]["sha1_name"]
            output_file = output_dir / f"{sha1_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(bm25_index, f)
                
        print(f"Processed {len(all_report_paths)} reports")

class VectorDBIngestor:
    """向量库构建器：chunk 文本 -> embedding -> FAISS 平面内积索引。

    契约（供检索端依赖）:
        - 索引行 i 对应 content.chunks[i]（构建与检索必须共享同一份 chunked JSON）；
        - 索引类型 IndexFlatIP：暴力全扫描 + 精确分数，年报规模（~百页 x 几十块）
          完全可承受，换来零近似误差；embedding 已单位化故内积即余弦。
    """

    def __init__(self):
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
        )
        return llm

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
    def _get_embeddings(self, text: Union[str, List[str]], model: str = "text-embedding-3-large") -> List[float]:
        """批量 embedding 入口：单条自动兜底为列表，超过 1024 条拆批调用。

        两个防御点:
            - 空字符串直接抛 ValueError：OpenAI 对空输入返回占位向量，混入库会
              制造「语义垃圾」chunk，污染所有查询；
            - 1024/批 —— OpenAI embedding 接口的单请求数组上限，超限会 400。
        重试语义（tenacity）: 网络级瞬断重试 2 次、间隔 20s；持续失败向上抛，
        宁可中断建库也不要产出残缺索引。

        Args:
            text: 单条或一批文本；返回与输入一一对应的 embedding 列表（顺序保持）

        Raises:
            ValueError: 入参为顶层 str 且为空串/纯空白时抛出
        """
        if isinstance(text, str) and not text.strip():
            raise ValueError("Input text cannot be an empty string.")

        if isinstance(text, list):
            text_chunks = [text[i:i + 1024] for i in range(0, len(text), 1024)]
        else:
            text_chunks = [text]

        embeddings = []
        for chunk in text_chunks:
            response = self.llm.embeddings.create(input=chunk, model=model)
            embeddings.extend([embedding.embedding for embedding in response.data])

        return embeddings

    def _create_vector_db(self, embeddings: List[float]):
        """把 embedding 列表构造成 FAISS 内积索引。

        选 IndexFlatIP 而非显式余弦的前提: OpenAI text-embedding-3 系列默认输出
        单位 L2 范数向量（官方保证），故内积 == 余弦。若未来替换为未归一化模型，
        必须在此先做 L2 归一化，否则相似度排序语义整体失真。
        """
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)  # Cosine distance
        index.add(embeddings_array)
        return index

    def _process_report(self, report: dict):
        """单报告建库：按 chunks 顺序逐条 embedding（空 chunk 文本由 _get_embeddings 拦截）。"""
        text_chunks = [chunk['text'] for chunk in report['content']['chunks']]
        embeddings = self._get_embeddings(text_chunks)
        index = self._create_vector_db(embeddings)
        return index

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """目录级建库入口：逐报告 embedding + 建索引 + 按 sha1_name 写 .faiss 文件。

        Args:
            all_reports_dir: chunked_reports 目录（每份 JSON 产出同主名 .faiss）
            output_dir: vector_dbs 目录（自动创建）

        Raises:
            任一报告 embedding 失败（重试耗尽）时向上抛，终止整批 —— 保证目录里
            要么全量、要么一次都没跑完的中间态可辨（半成品 .faiss 会误导检索端）。
        """
        all_report_paths = list(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)

        for report_path in tqdm(all_report_paths, desc="Processing reports"):
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
            index = self._process_report(report_data)
            sha1_name = report_data["metainfo"]["sha1_name"]
            faiss_file_path = output_dir / f"{sha1_name}.faiss"
            faiss.write_index(index, str(faiss_file_path))

        print(f"Processed {len(all_report_paths)} reports")