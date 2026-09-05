"""
切块阶段（离线）：02 整页 markdown 文本 -> 检索单元 chunks，供建库与召回消费。

职责定位:
    检索粒度的决策点。分块参数（300 token、50 token 重叠）直接决定向量命中质量：
    块太大 -> 语义混杂、top-k 内有效页占比低；块太小 -> 单块信息不全、上下文被截断。
    年报正文多为连续段落 + 表格文本，故用 langchain RecursiveCharacterTextSplitter
    按 gpt-4o 词表（o200k_base）的 token 计数递归切分。

数据流位置:
    输入: debug_data/02_merged_reports(_ser_tab)/*.json（content.pages: [{page, text}]）；
    输出: databases(_ser_tab)/chunked_reports/*.json —— 在 02 结构上追加
          content.chunks: [{id, type, page, text, length_tokens}]；
          其中 type=content 来自页文本，type=serialized_table 来自独立序列化块（休眠特性）。
    chunk 在 JSON 中的顺序 = 之后 FAISS 索引的行序（见 ingestion/retrieval 的契约注释），
    id 即全局单调序号，仅用于可读性，检索端实际以「数组下标 = 索引行号」回取。

核心依赖与副作用:
    - 纯 CPU 计算（tiktoken 编码），无网络 I/O；输出目录按需创建；
    - 输入文件不被改写（每个报告读入后写出新文件到 output_dir）。
"""
import json
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextSplitter():
    """02 整页文本 -> 检索 chunk 列表的转换器（无状态，可复用/可并发）。

    输出结构约定见模块头 docstring；两个切分通道（页文本、独立序列化表块）共用
    同一 id 序列，保证任意报告内 chunk id 全局唯一。
    """

    def _get_serialized_tables_by_page(self, tables: List[Dict]) -> Dict[int, List[Dict]]:
        """从 01 报告中抽出「已序列化」的表格并按键页号分组。

        只收含 serialized 字段的表（序列化前的表由页文本通道负责）；
        每张表拼接其全部 information_block 的 information_block 字段为一段文本
        （即该表对应的「上下文无关检索块」本体）。
        """
        tables_by_page = {}
        for table in tables:
            if 'serialized' not in table:
                continue
                
            page = table['page']
            if page not in tables_by_page:
                tables_by_page[page] = []
            
            table_text = "\n".join(
                block["information_block"] 
                for block in table["serialized"]["information_blocks"]
            )
            
            tables_by_page[page].append({
                "page": page,
                "text": table_text,
                "table_id": table["table_id"],
                "length_tokens": self.count_tokens(table_text)
            })
            
        return tables_by_page

    def _split_report(self, file_content: Dict[str, any], serialized_tables_report_path: Optional[Path] = None) -> Dict[str, any]:
        """对单份 02 报告执行切分，返回带 content.chunks 的同一结构。

        通道顺序（保持页内阅读序）: 第 N 页的文本块 -> 第 N 页的序列化表块 -> 第 N+1 页…

        Args:
            file_content: 02 合并 JSON（content.pages 为输入；content.chunks 会被写入）
            serialized_tables_report_path: 对应 01 报告路径；为 None 时不做独立表块通道。
                仅当上游以 include_serialized_tables=True 调起才有值（当前默认流程为 None，
                见 Pipeline.chunk_reports 注释 —— 独立表块是休眠特性，序列化文本通常已
                经在 02 的页面文本里）。

        Returns:
            输入 dict 本身（就地修改）。
        """
        chunks = []
        chunk_id = 0
        
        tables_by_page = {}
        if serialized_tables_report_path is not None:
            with open(serialized_tables_report_path, 'r', encoding='utf-8') as f:
                parsed_report = json.load(f)
            tables_by_page = self._get_serialized_tables_by_page(parsed_report.get('tables', []))
        
        for page in file_content['content']['pages']:
            page_chunks = self._split_page(page)
            for chunk in page_chunks:
                chunk['id'] = chunk_id
                chunk['type'] = 'content'
                chunk_id += 1
                chunks.append(chunk)
            
            if tables_by_page and page['page'] in tables_by_page:
                for table in tables_by_page[page['page']]:
                    table['id'] = chunk_id
                    table['type'] = 'serialized_table'
                    chunk_id += 1
                    chunks.append(table)
        
        file_content['content']['chunks'] = chunks
        return file_content

    def count_tokens(self, string: str, encoding_name="o200k_base"):
        encoding = tiktoken.get_encoding(encoding_name)

        tokens = encoding.encode(string)
        token_count = len(tokens)

        return token_count

    def _split_page(self, page: Dict[str, any], chunk_size: int = 300, chunk_overlap: int = 50) -> List[Dict[str, any]]:
        """单页文本切块：300 token 目标块、50 token 重叠，逐段补页号与 token 元数据。

        Args:
            page: {page, text} —— 页文本可能同时含正文与已渲染的 markdown 表格
            chunk_size: 目标块长（token）；年报问答需要「单块可自答」，此值偏保守
            chunk_overlap: 相邻块重叠量，缓解句子恰在边界被截断导致的语义断裂

        Returns:
            [{page, length_tokens, text}]（不含 id/type —— 由 _split_report 统一分配）
        """
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(page['text'])
        chunks_with_meta = []
        for chunk in chunks:
            chunks_with_meta.append({
                "page": page['page'],
                "length_tokens": self.count_tokens(chunk),
                "text": chunk
            })
        return chunks_with_meta

    def split_all_reports(self, all_report_dir: Path, output_dir: Path, serialized_tables_dir: Optional[Path] = None):
        """目录级切分入口：02 目录 -> chunked_reports 目录，逐报告同名写出。

        Args:
            all_report_dir: 02_merged_reports(_ser_tab) 目录
            output_dir: chunked_reports 目录（不存在则创建）
            serialized_tables_dir: 可选的 01 目录（按同名文件查找序列化表，
                缺失时仅打印警告继续 —— 独立表块通道容错降级，不影响正文块产出）
        """
        all_report_paths = list(all_report_dir.glob("*.json"))

        for report_path in all_report_paths:
            serialized_tables_path = None
            if serialized_tables_dir is not None:
                serialized_tables_path = serialized_tables_dir / report_path.name
                if not serialized_tables_path.exists():
                    print(f"Warning: Could not find serialized tables report for {report_path.name}")

            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
                
            updated_report = self._split_report(report_data, serialized_tables_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / report_path.name, 'w', encoding='utf-8') as file:
                json.dump(updated_report, file, indent=2, ensure_ascii=False)
                
        print(f"Split {len(all_report_paths)} files")
