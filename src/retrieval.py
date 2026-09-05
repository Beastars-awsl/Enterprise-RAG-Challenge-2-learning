"""
召回层（在线检索）：按「公司 + 问题」从离线索引取回候选文本块。

职责定位:
    - VectorRetriever : 向量召回。docling 文档 JSON + FAISS 索引配对加载（同名文件），
                        查询 embedding 后取 top-k；支持 chunk 级与「父文档」页级两种粒度；
    - BM25Retriever   : 词法召回（保留能力，问答主链路未挂接）；
    - HybridRetriever : 向量召回 + LLM 语义重排的融合检索器（reranking.py），
                        llm_reranking 配置开启时的实际入口。

数据流位置:
    输入: databases(_ser_tab)/chunked_reports/{sha1}.json（含 pages 与 chunks）+
          vector_dbs/{sha1}.faiss（或 bm25_dbs/{sha1}.pkl）—— 由 QuestionsProcessor
          注入目录后按 company_name 定位到报告；
    输出: [{distance, page, text}, ...] —— questions_processing 直接拼 RAG 上下文、
          并用 page 做引用校验，text 的物理页号必须与解析阶段口径一致（1 起始）。

关键契约（改动需跨模块同步）:
    - 检索行号 == chunked JSON 的 chunks 下标（ingestion 阶段保证）;
    - 查询向量与库向量必须出自同一 embedding 模型（text-embedding-3-large），
      否则距离分数整体无意义；
    - distance 语义: FAISS 内积相似度，越大越相似（注意与 L2 距离的直觉相反）。

核心依赖与副作用:
    - __init__ 会一次性把目录内全部文档与索引载入内存（all_dbs 缓存，代价是
      报告规模变大后启动慢）；每报告一次网络 embedding 调用（查询侧）；
    - 无写操作、无全局状态；并发问答线程共享同一实例需自行评估线程安全。
"""
import json
import logging
from typing import List, Tuple, Dict, Union
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
from src.reranking import LLMReranker

_log = logging.getLogger(__name__)

class BM25Retriever:
    """BM25 词法召回器（当前主链路未使用，作为无向量时的降级/对照方案保留）。

    使用约束:
        - 需先跑 create_bm25_db 生成与 documents 同主名的 .pkl 索引；
        - 查询切词必须与 BM25Ingestor 建索引口径一致（空格切分），否则评分失真；
        - 性能 O(块数)，全库扫描无索引结构优化。
    """

    def __init__(self, bm25_db_dir: Path, documents_dir: Path):
        self.bm25_db_dir = bm25_db_dir
        self.documents_dir = documents_dir

    def retrieve_by_company_name(self, company_name: str, query: str, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        """按公司名定位报告后执行 BM25 检索，返回 chunk 级或父页级结果。

        加载策略: 每次调用都重新扫描 documents 目录找报告 + 读 .pkl —— 无缓存，
        适合低频/对照使用；高频路径应改用 VectorRetriever 的常驻 all_dbs。

        Args:
            company_name: subset 中的公司全名（与 metainfo.company_name 精确匹配）
            query: 原始问题文本
            top_n: 返回候选数（按 BM25 得分降序）
            return_parent_pages: True 时把命中的 chunk 上卷为其所属整页
                （父文档检索语义；重复页只保留得分最高的首个命中）

        Raises:
            ValueError: 目录中不存在该公司报告（metainfo 缺失 company_name 时查不到）
        """
        document_path = None
        for path in self.documents_dir.glob("*.json"):
            with open(path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                if doc["metainfo"]["company_name"] == company_name:
                    document_path = path
                    document = doc
                    break

        if document_path is None:
            raise ValueError(f"No report found with '{company_name}' company name.")

        # Load corresponding BM25 index
        bm25_path = self.bm25_db_dir / f"{document['metainfo']['sha1_name']}.pkl"
        with open(bm25_path, 'rb') as f:
            bm25_index = pickle.load(f)
            
        # Get the document content and BM25 index
        document = document
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]
        
        # Get BM25 scores for the query
        tokenized_query = query.split()
        scores = bm25_index.get_scores(tokenized_query)
        
        actual_top_n = min(top_n, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_top_n]
        
        retrieval_results = []
        seen_pages = set()

        for index in top_indices:
            score = round(float(scores[index]), 4)
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])

            if return_parent_pages:
                # top_indices 已按得分降序，seen_pages 保证「每页只取首个（最优）命中」，
                # 既给足整页上下文又不让同一页的多条命中挤占 top-n 名额
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": score,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": score,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)

        return retrieval_results



class VectorRetriever:
    """向量召回器：常驻加载全部 报告JSON+FAISS 配对，按公司名查询 top-k 块/页。

    数据配对规则: chunked JSON 与 .faiss 按「文件主名 == sha1_name」配对；
    缺索引或缺 schema 的报告在加载期跳过并记 warning（启动宽容，查询时报错明确）。

    线程安全: __init__ 后所有检索方法只读共享状态；但 faiss 索引与 OpenAI client
    在并发调用下是否安全取决于实现，并行问答（QuestionsProcessor 多线程）复用
    同一实例时需按经验验证。
    """

    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.all_dbs = self._load_dbs()
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
            )
        return llm

    @staticmethod
    def set_up_llm():
        """与 _set_up_llm 重复的静态版本（历史遗留）：仅供不需要实例的静态工具调用。"""
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
            )
        return llm

    def _load_dbs(self):
        """加载期扫描: 读全部 chunked JSON 与其配对 FAISS 索引，构造 all_dbs 缓存。

        逐文件四道校验（任一不过即跳过该报告，不中断整批）:
            1) 存在配对 .faiss；2) JSON 可解析；3) 顶层 schema 含 metainfo/content；
            4) FAISS 可读。
        注意: 跳过只记日志，可能导致后续按公司名查询抛「报告不存在」——
        这是「启动宽容、查询明确」的取舍：单份坏文件不应拖垮整批加载。
        """
        all_dbs = []
        # Get list of JSON document paths
        all_documents_paths = list(self.documents_dir.glob('*.json'))
        vector_db_files = {db_path.stem: db_path for db_path in self.vector_db_dir.glob('*.faiss')}
        
        for document_path in all_documents_paths:
            stem = document_path.stem
            if stem not in vector_db_files:
                _log.warning(f"No matching vector DB found for document {document_path.name}")
                continue
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"Error loading JSON from {document_path.name}: {e}")
                continue
            
            # Validate that the document meets the expected schema
            if not (isinstance(document, dict) and "metainfo" in document and "content" in document):
                _log.warning(f"Skipping {document_path.name}: does not match the expected schema.")
                continue
            
            try:
                vector_db = faiss.read_index(str(vector_db_files[stem]))
            except Exception as e:
                _log.error(f"Error reading vector DB for {document_path.name}: {e}")
                continue
                
            report = {
                "name": stem,
                "vector_db": vector_db,
                "document": document
            }
            all_dbs.append(report)
        return all_dbs

    @staticmethod
    def get_strings_cosine_similarity(str1, str2):
        llm = VectorRetriever.set_up_llm()
        embeddings = llm.embeddings.create(input=[str1, str2], model="text-embedding-3-large")
        embedding1 = embeddings.data[0].embedding
        embedding2 = embeddings.data[1].embedding
        similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity_score = round(similarity_score, 4)
        return similarity_score

    def retrieve_by_company_name(self, company_name: str, query: str, llm_reranking_sample_size: int = None, top_n: int = 3, return_parent_pages: bool = False) -> List[Tuple[str, float]]:
        """向量召回主入口: 查公司报告 -> 查询向量 -> FAISS top-k -> chunk/父页结果。

        Args:
            company_name: subset 公司全名（精确匹配 metainfo.company_name）
            query: 问题原文（与文档同语种直接送 embedding）
            llm_reranking_sample_size: 兼容签名 —— 本方法不做重排，直接作为
                top_n 的候选量；由 HybridRetriever 传入更大的采样窗口
            top_n: 返回候选数；chunk 级时等于 k，父页级时因去重可能 < k
            return_parent_pages: True 时上卷为整页文本（父文档检索）

        Returns:
            [{distance, page, text}]，按相似度降序；distance 为 FAISS 内积相似度
            （越大越相似），调用方（问答 prompt 拼接）无需感知索引内部语义

        Raises:
            ValueError: 该公司无报告（加载期被跳过或根本没建库）
        """
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                _log.error(f"Report '{report.get('name')}' is missing 'metainfo'!")
                raise ValueError(f"Report '{report.get('name')}' is missing 'metainfo'!")
            if metainfo.get("company_name") == company_name:
                target_report = report
                break

        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")

        document = target_report["document"]
        vector_db = target_report["vector_db"]
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]

        # k 不允许超过索引样本数，超出时 FAISS 报错 —— 显式收敛到块数上限
        actual_top_n = min(top_n, len(chunks))

        embedding = self.llm.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        )
        embedding = embedding.data[0].embedding
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        # 查询向量与建库向量同模型（单位化），IP 分数即余弦相似度
        distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)

        retrieval_results = []
        seen_pages = set()

        for distance, index in zip(distances[0], indices[0]):
            distance = round(float(distance), 4)
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])
            if return_parent_pages:
                # faiss 返回按相似度降序 -> 每个页号的首次出现即该页最优 chunk；
                # 去重后整页文本进上下文，同时保证页间不互相挤占名额
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": distance,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": distance,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)

        return retrieval_results

    def retrieve_all(self, company_name: str) -> List[Dict]:
        """无检索路径: 返回整本报告的全部页面（full_context 配置用）。

        设计意图: Gemini 大上下文模式不依赖召回，直接把全部页文本喂给模型；
        distance 固定 0.5 只是占位 —— 页面按页码升序排列而非相似度排序，
        消费方（RAG 上下文格式化）不应对 distance 做任何语义假设。

        Raises:
            ValueError: 该公司报告不存在
        """
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                continue
            if metainfo.get("company_name") == company_name:
                target_report = report
                break

        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")

        document = target_report["document"]
        pages = document["content"]["pages"]

        all_pages = []
        for page in sorted(pages, key=lambda p: p["page"]):
            result = {
                "distance": 0.5,
                "page": page["page"],
                "text": page["text"]
            }
            all_pages.append(result)

        return all_pages


class HybridRetriever:
    """向量召回 + LLM 重排的两段式检索器（QuestionsProcessor 在 llm_reranking 开启时使用）。

    流程: 先向量召回 llm_reranking_sample_size 个粗候选（放宽窗口以免漏召），
    LLM 逐块/逐批打分后按融合分取 top_n。两段式让精排只看「可能有戏」的块，
    控制了昂贵的 LLM 打分次数。
    """

    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir)
        self.reranker = LLMReranker()

    def retrieve_by_company_name(
        self, 
        company_name: str, 
        query: str, 
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 2,
        top_n: int = 6,
        llm_weight: float = 0.7,
        return_parent_pages: bool = False
    ) -> List[Dict]:
        """
        Retrieve and rerank documents using hybrid approach.
        
        Args:
            company_name: 目标公司（subset 全名，精确匹配）
            query: 问题原文
            llm_reranking_sample_size: 向量粗召窗口（> top_n 才有重排意义，
                问答配置如 30 -> top 10）
            documents_batch_size: 单个 LLM 打分 prompt 里塞的块数
                （>1 走「多块一次打分」路径，省调用；=1 走单块打分 + 块级并行路径）
            top_n: 重排后返回的最终结果数（按融合分降序截断）
            llm_weight: LLM 相关性分数的权重（0-1）；融合分 = llm_weight * LLM分 +
                (1-llm_weight) * 向量相似度
            return_parent_pages: 传给向量召回，True 时以整页文本参与重排
                （父文档检索下重排对象是页而非块）

        Returns:
            按融合分降序的 top_n 个结果 dict（在召回结果上增加 relevance_score /
            combined_score 两个字段），供 RAG 上下文直接格式化。

        Notes:
            注意: 传入该方法的调用方负责保证 llm_reranking_sample_size >= top_n。
        """
        # Get initial results from vector retriever
        vector_results = self.vector_retriever.retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages
        )
        
        # Rerank results using LLM
        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=vector_results,
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight
        )
        
        return reranked_results[:top_n]
