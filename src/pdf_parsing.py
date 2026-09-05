"""
PDF 解析阶段（离线 01）：docling 封装 + docling 产物到「下游中间格式」的适配。

职责定位:
    - PDFParser          : docling DocumentConverter 的薄封装（OCR / 表格结构识别配置、
                          串行/多进程驱动、按 subset.csv 注入 sha1->公司名映射）；
    - JsonReportProcessor: 把 docling 的「对象图」导出格式（body/children 全是指针 $ref，
                          正文与文本/表格/图片实体分开存放）折叠成下游可直读的扁平结构：
                          按页归类的 content 列表 + 独立 tables/pictures 列表（按 id 引用）。

数据流位置:
    输入: pdf_reports/*.pdf + subset.csv（sha1 列必须与 PDF 文件名一致）；
    输出: debug_data/01_parsed_reports/{sha1}.json（供 parsed_reports_merging /
          tables_serialization / pipeline 各阶段消费）；docling 原始导出另存
          01_parsed_reports_debug（体积大、仅供排障，无下游消费）。

核心依赖与副作用:
    - 需要 docling 模型（首次自动从 HuggingFace 下载，可先跑 main.py download-models 暖机）；
      OCR 用 EasyOCR，表格结构用 TableFormer ACCURATE 模式 —— 二者都较重，GPU 收益明显；
    - _process_chunk 在多进程子进程内重建 PDFParser（converter 不可 pickle）；
    - 解析失败聚合计数：任一文档失败，parse_and_export 抛 RuntimeError（fail-fast）；
    - 本模块内无全局可变状态。
"""
import os
import time
import logging
import re
import json
from tabulate import tabulate
from pathlib import Path
from typing import Iterable, List

# from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
# from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult

_log = logging.getLogger(__name__)

def _process_chunk(pdf_paths, pdf_backend, output_dir, num_threads, metadata_lookup, debug_data_path):
    """子进程入口：为一个 PDF 分片重建完整的 PDFParser 并执行解析。

    设计约束: docling 的 DocumentConverter（含已加载的模型与 OCR 引擎）无法跨进程
    pickle，因此多进程编排只能把「构造参数」传给子进程、在每个子进程里自建解析器
    （代价：每进程重复加载模型，内存 = 进程数 x 单模型占用）。
    """
    # 1. 每个子进程都必须自建 PDFParser：docling 的 DocumentConverter（含已加载的
    #    模型与 OCR 引擎）无法跨进程 pickle，只能把「构造参数」传进来、在子进程内实例化
    parser = PDFParser(
        pdf_backend=pdf_backend,
        output_dir=output_dir,
        num_threads=num_threads,
        csv_metadata_path=None  # 元数据查找表已直接传参，无需再从 csv 加载
    )
    # 2. 补挂运行时依赖：注入父进程的 sha1 -> 公司名 查找表与 debug 落盘路径
    parser.metadata_lookup = metadata_lookup
    parser.debug_data_path = debug_data_path
    # 3. 串行解析本分片内的全部 PDF（内部任一失败会抛 RuntimeError，由父进程捕获）
    parser.parse_and_export(pdf_paths)
    # 4. 返回一行汇总文本，父进程据此拆分出已处理数量、刷新总进度
    return f"Processed {len(pdf_paths)} PDFs."

class PDFParser:
    """docling 解析门面：负责配置转换管线、串行/并行驱动与产物落盘。

    设计意图:
        把 docling 的复杂配置（OCR、表格结构、后端解析器）与 csv 元数据注入集中在一处，
        让上层 Pipeline 只需声明「输出到哪、元数据在哪」。

    Notes:
        - csv_metadata_path 提供 sha1 -> company_name；PDF 文件名（去 .pdf）必须与
          subset.csv 的 sha1 值对应，否则 metainfo 中 company_name 缺失；
        - 后端固定 DoclingParseV2DocumentBackend（代码中另注释保留了其他后端便于切换）。
    """

    def __init__(
        self,
        pdf_backend=DoclingParseV2DocumentBackend,
        output_dir: Path = Path("./parsed_pdfs"),
        num_threads: int = None,
        csv_metadata_path: Path = None,
    ):
        # 1. 登记外部配置：后端解析器、输出目录、线程上限与 csv 元数据路径
        self.pdf_backend = pdf_backend
        self.output_dir = output_dir
        self.doc_converter = self._create_document_converter()
        self.num_threads = num_threads
        self.metadata_lookup = {}
        self.debug_data_path = None

        # 2. 可选：传入 csv 时预加载 {sha1: company_name} 查找表，
        #    供后续 metainfo 的公司名增强使用（PDF 文件名需与 sha1 列一致）
        if csv_metadata_path is not None:
            self.metadata_lookup = self._parse_csv_metadata(csv_metadata_path)

        # 3. 收紧底层线程数：EasyOCR/ONNX/OpenMP 在进程内自行开线程池，
        #    显式限制可避免多进程叠加时 CPU 线程过载（典型：10 进程 x 16 线程）
        if self.num_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(self.num_threads)

    @staticmethod
    def _parse_csv_metadata(csv_path: Path) -> dict:
        """解析 subset.csv 为 {sha1: {company_name: ...}} 查找表。

        兼容两代赛题 CSV 的列名（company_name / name）；取行内值后剥掉两侧引号，
        防止 CSV 手工导出时公司名被引号包裹导致下游匹配失败。

        Args:
            csv_path: subset.csv 路径

        Returns:
            以 sha1 为主键的元信息字典；主键不存在的报告将拿不到 company_name。
        """
        import csv
        metadata_lookup = {}

        # 1. 打开 csv 并交给 DictReader：首行作为表头，后续按列名取值
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            # 2. 逐行登记元信息：兼容两代赛题列名（company_name / name），
            #    剥掉手工导出可能带入的引号，防止下游公司名匹配失败
            for row in reader:
                company_name = row.get('company_name', row.get('name', '')).strip('"')
                # 3. 以 sha1 为主键建表：PDF 文件名（去 .pdf）将据此反查 company_name
                metadata_lookup[row['sha1']] = {
                    'company_name': company_name
                }
        # 4. 返回查找表；表中缺失的报告拿不到 company_name（不报错，保持兼容）
        return metadata_lookup

    def _create_document_converter(self) -> "DocumentConverter": # type: ignore
        """按解析精度要求构造 docling DocumentConverter。

        关键取舍（对应高精度 RAG 基线）:
            - do_ocr=True + EasyOCR(英文)：扫描页/低质量 PDF 也能出文本，但速度成本大；
            - TableFormer ACCURATE + cell matching：年报表格结构还原精度优先，
              是下游表格 markdown/序列化质量的根基，慢于 FAST 模式。
        """
        from docling.document_converter import DocumentConverter, FormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, EasyOcrOptions
        from docling.datamodel.base_models import InputFormat
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
        
        # 1. 开启 OCR（EasyOCR，英文）：让扫描页/低清 PDF 也能提取出文本，代价是解析变慢
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        ocr_options = EasyOcrOptions(lang=['en'], force_full_page_ocr=False)
        pipeline_options.ocr_options = ocr_options
        # 2. 开启表格结构识别与单元格匹配，TableFormer 用 ACCURATE 模式：
        #    年报表格还原精度优先（下游表格质量的根基），慢于 FAST 模式
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        # 3. 把「PDF 输入格式 -> 标准解析管线」绑定为格式选项（连同管线配置与后端）
        format_options = {
            InputFormat.PDF: FormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=pipeline_options,
                backend=self.pdf_backend
            )
        }

        # 4. 组装 converter：实例化一次后复用于每次转换（模型与 OCR 引擎常驻内存）
        return DocumentConverter(format_options=format_options)

    def convert_documents(self, input_doc_paths: List[Path]) -> Iterable[ConversionResult]:
        """把一批 PDF 交给 docling 转换，返回惰性迭代的 ConversionResult 流。

        注意 docling 转换本身会占用 GPU/CPU 并可能较慢，调用方应保证
        输入列表已按 chunk 划分（并行模式下每进程各自调用本方法）。
        """
        # 1. 整批交给 docling 转换：convert_all 返回惰性迭代器，
        #    真正的解析与耗时发生在调用方逐份消费结果时
        conv_results = self.doc_converter.convert_all(source=input_doc_paths)
        # 2. 返回原始结果流（一次只驻留少量结果），由 process_documents 负责消费
        return conv_results

    def process_documents(self, conv_results: Iterable[ConversionResult]):
        """消费转换结果：成功者写 01 JSON（含 docling 原始 debug 副本），失败者计数。
        它承接 convert_documents 产生的原始解析结果流，负责质量校验、数据标准化、元数据增强以及最终的文件落盘
        
        Args:
            conv_results: convert_documents 的迭代产物（流式处理，一次只驻留少量结果）

        Returns:
            (success_count, failure_count) —— 供上层决定是否 fail-fast。
        """
        # 1. 确保输出目录存在（不存在则递归创建）
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        # 2. 初始化成败计数器：供末尾汇总，并让上层据此决定是否 fail-fast
        success_count = 0
        failure_count = 0

        # 3. 逐个消费转换结果（惰性流式，一次只驻留一份文档的内存）
        for conv_res in conv_results:
            # 4. 分支一：转换成功 -> 归一化、折叠成下游格式并落盘
            if conv_res.status == ConversionStatus.SUCCESS:
                success_count += 1
                processor = JsonReportProcessor(metadata_lookup=self.metadata_lookup, debug_data_path=self.debug_data_path)

                # 5. 导出 docling 对象图，并防御性归一化：若含按页 content 就把页号补成
                #    连续序列（docling 当前版本不含该键，此步是 no-op，仅防将来结构变化）
                data = conv_res.document.export_to_dict()
                normalized_data = self._normalize_page_sequence(data)

                # 6. 折叠成下游扁平 JSON 并写盘；文件名取 PDF 主名（= sha1），
                #    02/03 及后续所有阶段都靠它关联
                processed_report = processor.assemble_report(conv_res, normalized_data)
                doc_filename = conv_res.input.file.stem
                if self.output_dir is not None:
                    with (self.output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
                        json.dump(processed_report, fp, indent=2, ensure_ascii=False)
            else:
                # 7. 分支二：转换失败 -> 仅累计失败数并打日志，不打断其余文档的处理
                failure_count += 1
                _log.info(f"Document {conv_res.input.file} failed to convert.")

        # 8. 汇总本批处理情况并返回 (成功, 失败)，由上层决定是否抛错终止
        _log.info(f"Processed {success_count + failure_count} docs, of which {failure_count} failed")
        return success_count, failure_count

    def _normalize_page_sequence(self, data: dict) -> dict:
        """把缺失的中间页码用空页补齐，保证 content 页序列连续。

        背景: 下游所有阶段都假定「页码 == content 里的物理页序号」且无空洞；
        缺页（例如全空白页被 docling 跳过）会让后续按 page 取数或对齐错位。
        本函数即防御这类脏输入。

        Returns:
            归一化后的 dict；若导出结构不含 'content'（docling 版本差异），
            原样返回 —— 该分支对当前版本是 no-op，保留仅为兼容旧格式。
        """
        # 1. 防御分支：导出结构不含按页 'content'（docling 版本差异）时原样返回，
        #    该分支对当前版本是 no-op，保留仅为兼容旧格式
        if 'content' not in data:
            return data

        # 2. 浅拷贝一份再修改，避免污染上游导出对象
        normalized_data = data.copy()

        # 3. 收集现有页号并取最大页：它决定需要补齐的序列范围上界
        existing_pages = {page['page'] for page in data['content']}
        max_page = max(existing_pages)

        # 4. 空页模板：缺页用「content 为空 + 维度留空」的模板页填充
        empty_page_template = {
            "content": [],
            "page_dimensions": {}  # 缺页没有真实版面，维度留空即可
        }

        # 5. 从第 1 页扫到最大页重排新序列：已有页原样取用，空洞页补空页
        new_content = []
        for page_num in range(1, max_page + 1):
            # 5.1 命中已有页则取回；未命中（页号空洞）则套模板补一个空页
            page_content = next(
                (page for page in data['content'] if page['page'] == page_num),
                {"page": page_num, **empty_page_template}
            )
            new_content.append(page_content)

        # 6. 用连续无空洞的页序列替换原 content 后返回
        normalized_data['content'] = new_content
        return normalized_data

    def parse_and_export(self, input_doc_paths: List[Path] = None, doc_dir: Path = None):
        """串行解析并导出（单进程入口，也是每个并行子进程内部执行的函数）。

        两者必传其一: input_doc_paths 显式给文件列表；doc_dir 则扫描该目录全部 *.pdf。

        Raises:
            RuntimeError: 存在转换失败文档时抛出（列出失败路径），保持 fail-fast 语义，
                          避免带着残缺语料继续跑下游而浪费 API 额度。
        """
        # 1. 记录起始时间：总耗时涵盖模型加载与全部转换/落盘
        start_time = time.time()
        # 2. 输入归一化：未显式给文件列表时，扫描 doc_dir 下全部 *.pdf
        if input_doc_paths is None and doc_dir is not None:
            input_doc_paths = list(doc_dir.glob("*.pdf"))

        total_docs = len(input_doc_paths)
        _log.info(f"Starting to process {total_docs} documents")

        # 3. 分两步执行：先拿惰性转换结果流，再逐份消费并落盘（详见 process_documents）
        conv_results = self.convert_documents(input_doc_paths)
        success_count, failure_count = self.process_documents(conv_results=conv_results)
        elapsed_time = time.time() - start_time

        # 4. fail-fast：存在失败文档就列出明细并抛 RuntimeError，
        #    避免带着残缺语料继续跑下游、白白消耗 API 额度
        if failure_count > 0:
            error_message = f"Failed converting {failure_count} out of {total_docs} documents."
            failed_docs = "Paths of failed docs:\n" + '\n'.join(str(path) for path in input_doc_paths)
            _log.error(error_message)
            _log.error(failed_docs)
            raise RuntimeError(error_message)

        # 5. 全部成功：打印 # 分栏完成摘要（与并行版同款日志格式），便于肉眼扫结果
        _log.info(f"{'#'*50}\nCompleted in {elapsed_time:.2f} seconds. Successfully converted {success_count}/{total_docs} documents.\n{'#'*50}")

    def parse_and_export_parallel(
        self,
        input_doc_paths: List[Path] = None,
        doc_dir: Path = None,
        optimal_workers: int = 10,
        chunk_size: int = None
    ):
        """多进程并行解析：先按 chunk_size 切片，再把每个片作为一个池任务提交。

        Args:
            input_doc_paths: 待处理 PDF 列表（与 doc_dir 二选一）
            doc_dir: 目录（输入列表为空时扫描 *.pdf）
            optimal_workers: 进程数上限；各 worker 独立加载 docling 模型，
                取值需综合内存/显存容量（默认 10 为经验值）
            chunk_size: 每进程一次领取的 PDF 数；缺省按 total//workers 分摊

        Notes:
            chunk 粒度是对两类开销的权衡：_process_chunk 每次执行都会重建解析器并重新
            加载模型，chunk 过小导致模型反复加载；chunk 过大则尾部进程闲置、算力空转
            （本函数不做动态负载均衡）。
        """
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # ========== 准备阶段：输入归一化、worker/chunk 计算与分片 ==========
        # 1. 输入归一化：未显式给文件列表时，扫描 doc_dir 下全部 *.pdf
        if input_doc_paths is None and doc_dir is not None:
            input_doc_paths = list(doc_dir.glob("*.pdf"))

        total_pdfs = len(input_doc_paths)
        _log.info(f"Starting parallel processing of {total_pdfs} documents")

        # 2. 进程数上限：未显式指定时取 CPU 核数与文件数的较小者
        cpu_count = multiprocessing.cpu_count()
        if optimal_workers is None:
            optimal_workers = min(cpu_count, total_pdfs)

        # 3. chunk 大小：缺省按「总量均摊给各 worker」估算（至少 1，防除零/空片）
        if chunk_size is None:
            chunk_size = max(1, total_pdfs // optimal_workers)

        # 4. 按 chunk_size 把总列表切成若干连续分片（末尾一片可能不足整块）
        chunks = [
            input_doc_paths[i : i + chunk_size]
            for i in range(0, total_pdfs, chunk_size)
        ]

        start_time = time.time()
        processed_count = 0
        
        # ========== 并行执行阶段：提交任务 -> 监听完成 -> 汇总进度 ==========
        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            # 1. 提交所有任务，得到 futures 列表
            futures = [
                executor.submit(
                    _process_chunk,
                    chunk,
                    self.pdf_backend,
                    self.output_dir,
                    self.num_threads,
                    self.metadata_lookup,
                    self.debug_data_path
                )
                for chunk in chunks
            ]

            # 2. 开始监听：哪个 chunk 先解析完，就先进入循环体
            for future in as_completed(futures):
                try:
                    # 3. 获取该特定任务的结果（如果任务出错，这里会抛出异常）
                    result = future.result()

                    # 4. 解析结果字符串，更新全局计数器
                    processed_count += int(result.split()[1])

                    # 5. 立即打印日志，让用户知道进度
                    _log.info(f"{'#'*50}\n{result} ({processed_count}/{total_pdfs} total)\n{'#'*50}")
                except Exception as e:
                    # 6. 一旦有任何一个 chunk 失败，立即记录错误并终止整个并行流程
                    _log.error(f"Error processing chunk: {str(e)}")
                    raise

        # 收尾：能走到这里说明所有 chunk 均已成功，统计并打印总耗时
        elapsed_time = time.time() - start_time
        _log.info(f"Parallel processing completed in {elapsed_time:.2f} seconds.")


class JsonReportProcessor:
    """docling「对象图」-> 下游扁平 JSON 的格式转换器（报告级，无状态可复用）。

    转换要点（产物 schema 即下游全部阶段的输入契约，改这里必须同步改消费方）:
        - metainfo: 文件级统计 + company_name（来自 csv 映射）；
        - content  : 按物理页号排序的 [{page, content:[{type, text_id|table_id|picture_id,...}]}]
                    正文块只保留文本引用，表格/图片以 id 形式挂页，实体本体在下面两个列表；
        - tables  : [{table_id, page, bbox, markdown, html, json}] —— 表格全文另行保存，
                    供合并阶段按 id 取回渲染（避免重复携带大对象）；
        - pictures: [{picture_id, page, bbox, children}] —— 仅文本子块，本身不含图像数据。

    依赖与副作用:
        - assemble_* 系列要求 data 为 docling export_to_dict() 结构（body/children 的
          $ref 指针语义: "/texts/12" -> data['texts'][12]）；
        - debug_data() 会把原始 data 整份落盘到 debug_data_path（体积大）。
    """

    def __init__(self, metadata_lookup: dict = None, debug_data_path: Path = None):
        self.metadata_lookup = metadata_lookup or {}
        self.debug_data_path = debug_data_path

    def assemble_report(self, conv_result, normalized_data=None):
        """组装整份报告 JSON：metainfo + content + tables + pictures，并落 debug 副本。

        Args:
            conv_result: docling ConversionResult（含 document 与 tables 对象）
            normalized_data: 已归一化的 export_to_dict 结果；为 None 时现场再导出

        Returns:
            上文 schema 所描述的扁平报告 dict（供调用方 json.dump 到 01 目录）。
        """
        # 1. 数据源统一：调用方已归一化则直接采用，否则现场导出一次 docling 对象图
        data = normalized_data if normalized_data is not None else conv_result.document.export_to_dict()
        assembled_report = {}
        # 2. 依次组装四大部分：metainfo（统计/身份）-> 按页 content -> tables -> pictures
        assembled_report['metainfo'] = self.assemble_metainfo(data)
        assembled_report['content'] = self.assemble_content(data)
        assembled_report['tables'] = self.assemble_tables(conv_result.document.tables, data)
        assembled_report['pictures'] = self.assemble_pictures(data)
        # 3. 顺带把 docling 原始导出整份落盘到 debug 目录（仅排障用，体积大）
        self.debug_data(data)
        # 4. 返回组装好的扁平报告 dict（供调用方 json.dump 到 01 目录）
        return assembled_report
    
    def assemble_metainfo(self, data):
        """汇出报告级统计信息与身份字段。

        关键约定: sha1_name 取源 PDF 文件名（去扩展名），后续所有阶段（02/03、
        chunked/faiss/bm25 文件命名）都以此为关联键 —— 文件名必须与 subset.csv 的
        sha1 一致，否则 company_name 缺失且库文件对不上号。
        """
        metainfo = {}
        # 1. 关联键 = 源 PDF 文件名去扩展名（即 subset.csv 中的 sha1），
        #    后续 02/03 各阶段的文件命名都以它为键，必须保持一致
        sha1_name = data['origin']['filename'].rsplit('.', 1)[0]
        metainfo['sha1_name'] = sha1_name
        # 2. 汇出报告级统计：页数及各实体数量，供下游了解一份报告的构成
        metainfo['pages_amount'] = len(data.get('pages', []))
        metainfo['text_blocks_amount'] = len(data.get('texts', []))
        metainfo['tables_amount'] = len(data.get('tables', []))
        metainfo['pictures_amount'] = len(data.get('pictures', []))
        metainfo['equations_amount'] = len(data.get('equations', []))
        metainfo['footnotes_amount'] = len([t for t in data.get('texts', []) if t.get('label') == 'footnote'])

        # 3. csv 元数据增强：查找表命中则补 company_name（未命中不报错，保持兼容）
        if self.metadata_lookup and sha1_name in self.metadata_lookup:
            csv_meta = self.metadata_lookup[sha1_name]
            metainfo['company_name'] = csv_meta['company_name']

        return metainfo

    def process_table(self, table_data):
        # 占位钩子（历史遗留）：真实表格处理由 assemble_tables 与 tables_serialization 承担
        return 'processed_table_content'

    def debug_data(self, data):
        """把 docling 原始导出整份落盘（仅排障用，下游无消费者）。"""
        # 1. 未配置 debug 目录则直接跳过（生产模式不落这份体积巨大的原始副本）
        if self.debug_data_path is None:
            return
        # 2. 以文档名命名的输出文件路径；父目录不存在时先递归创建
        doc_name = data['name']
        path = self.debug_data_path / f"{doc_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        # 3. 整份写入 docling 原始导出（indent=2 便于人工排障阅读）
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def expand_groups(self, body_children, groups):
        """把 body 子元素中的 groups 指针展开为组内元素，并把组信息贴到每个元素上。

        背景: docling 用「组」表达列表、段落等聚合结构；直接保留指针会让下游每个
        消费方都重复实现解引用。本函数做一次展开，同时把 group_id/name/label 打到
        子元素上，供下游按组过滤或渲染时使用。

        Args:
            body_children: body.children 原始列表（元素为 $ref dict 或普通 dict）
            groups: data['groups'] 组定义表

        Returns:
            展开后的元素列表（组指针被替换为组内子元素的副本，其余原样透传）。
        """
        expanded_children = []
        # 1. 逐元素处理：只关心 $ref 指针型元素，普通 dict 原样透传
        for item in body_children:
            if isinstance(item, dict) and '$ref' in item:
                # 2. 解析指针（形如 "/groups/12"）：末两段分别是实体类型与编号
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)

                # 3. 命中组指针：把组内每个子元素复制展开，并贴上组信息供下游使用
                if ref_type == 'groups':
                    group = groups[ref_num]
                    group_id = ref_num
                    group_name = group.get('name', '')
                    group_label = group.get('label', '')

                    for child in group['children']:
                        child_copy = child.copy()
                        child_copy['group_id'] = group_id
                        child_copy['group_name'] = group_name
                        child_copy['group_label'] = group_label
                        expanded_children.append(child_copy)
                else:
                    # 4. 非组指针（文本/表格/图片等）：不解引用，原样留给上层继续分类
                    expanded_children.append(item)
            else:
                # 5. 普通元素（无 $ref）：不做处理直接保留
                expanded_children.append(item)

        return expanded_children
    
    def _process_text_reference(self, ref_num, data):
        """Helper method to process text references and create content items.
        
        Args:
            ref_num (int): Reference number for the text item
            data (dict): Document data dictionary
            
        Returns:
            dict: Processed content item with text information
        """
        # 1. 按编号取回文本实体本体，构造下游 content 项（label 即文本类型）
        text_item = data['texts'][ref_num]
        item_type = text_item['label']
        content_item = {
            'text': text_item.get('text', ''),
            'type': item_type,
            'text_id': ref_num
        }

        # 2. 仅当存在与 text 不同的原始文本 'orig' 时才附加该键，避免冗余
        orig_content = text_item.get('orig', '')
        if orig_content != text_item.get('text', ''):
            content_item['orig'] = orig_content

        # 3. 附属字段（列表序号 enumerated / 项目符号 marker）有则携带，
        #    供下游还原列表语境
        if 'enumerated' in text_item:
            content_item['enumerated'] = text_item['enumerated']
        if 'marker' in text_item:
            content_item['marker'] = text_item['marker']

        return content_item
    
    def assemble_content(self, data):
        """按物理页聚合正文块：{page: {page, content: [...], page_dimensions}} 的有序列表。

        页归属取自文本/表格/图片的 prov[0].page_no（1 起始）；同一页多个实体共用一个
        page 字典。返回按页号升序，页号空洞会被压缩（页码值本身不变）。
        """
        pages = {}
        # 1. 先展开 body 中的组指针：组内子元素复制后带上 group_id/name/label 摊平，
        #    后续只需处理「$ref -> 文本/表格/图片」一种形态即可
        body_children = data['body']['children']
        groups = data.get('groups', [])
        expanded_body_children = self.expand_groups(body_children, groups)

        # 2. 对每个展开元素按 $ref 指针分类（指针形如 "/texts/12"）并落到对应页面
        for item in expanded_body_children:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                # 3. 解引用：指针末两段即「实体类型 / 实体编号」
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)

                # —— 分支一：文本块 ——
                if ref_type == 'texts':
                    # 4. 复用文本引用处理逻辑生成 content 项（text/type/text_id/orig）
                    content_item = self._process_text_reference(ref_num, data)

                    # 5. 若该文本属于某个组（列表/段落），把组信息回贴到 content 项上
                    if 'group_id' in item:
                        content_item['group_id'] = item['group_id']
                        content_item['group_name'] = item['group_name']
                        content_item['group_label'] = item['group_label']

                    # 6. 归属页面：取 prov 首个来源的 page_no（页号从 1 起）
                    if 'prov' in text_item and text_item['prov']:
                        page_num = text_item['prov'][0]['page_no']

                        # 7. 该页还没建桶就初始化：同页所有实体共用一个 page 字典
                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': text_item['prov'][0].get('bbox', {})
                            }

                        # 8. 入桶：把文本 content 项追加到所属页面
                        pages[page_num]['content'].append(content_item)

                # —— 分支二：表格块（实体本体在顶层 tables 列表） ——
                elif ref_type == 'tables':
                    table_item = data['tables'][ref_num]
                    # 9. 页内只挂 {type, table_id} 引用，避免重复携带整张表的大对象
                    content_item = {
                        'type': 'table',
                        'table_id': ref_num
                    }

                    if 'prov' in table_item and table_item['prov']:
                        page_num = table_item['prov'][0]['page_no']

                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': table_item['prov'][0].get('bbox', {})
                            }

                        pages[page_num]['content'].append(content_item)

                # —— 分支三：图片块（同上，页内只放引用） ——
                elif ref_type == 'pictures':
                    picture_item = data['pictures'][ref_num]
                    content_item = {
                        'type': 'picture',
                        'picture_id': ref_num
                    }

                    if 'prov' in picture_item and picture_item['prov']:
                        page_num = picture_item['prov'][0]['page_no']

                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': picture_item['prov'][0].get('bbox', {})
                            }

                        pages[page_num]['content'].append(content_item)

        # 10. 按页号升序输出页字典列表（页号数值原样保留，空洞页不会出现在序列里）
        sorted_pages = [pages[page_num] for page_num in sorted(pages.keys())]
        return sorted_pages

    def assemble_tables(self, tables, data):
        """汇出全部表格：每张表同时保留 markdown/html/结构化 JSON 三种表示。

        下游分工: markdown 供合并阶段直接渲染进页面文本；html 供表格序列化阶段
        （LLM 输入）；json 供程序化访问。bbox 统一从 docling 的 {l,t,r,b} 字典
        显式取值转为平面列表 [l, t, r, b]，全链路口径一致（图片同规则）。
        """
        assembled_tables = []
        # 1. 逐张表生成三种表示：md（合并阶段渲染进正文）、html（表格序列化/LLM 输入）、
        #    json（结构化对象，程序化访问）
        for i, table in enumerate(tables):
            table_json_obj = table.model_dump()
            table_md = self._table_to_md(table_json_obj)
            table_html = table.export_to_html()

            # 2. 版位信息取自 docling prov 首个来源：所在页号 page_no 与 bbox
            table_data = data['tables'][i]
            table_page_num = table_data['prov'][0]['page_no']
            table_bbox = table_data['prov'][0]['bbox']
            # 3. bbox 从 {l,t,r,b} 字典拍平成平面列表 [l, t, r, b]，
            #    与全链路图片 bbox 口径一致
            table_bbox = [
                table_bbox['l'],
                table_bbox['t'],
                table_bbox['r'],
                table_bbox['b']
            ]

            # 4. 行/列数取自结构化网格数据（供下游快速判断表格规模）
            nrows = table_data['data']['num_rows']
            ncols = table_data['data']['num_cols']

            # 5. 规范化 table_id：从 self_ref 指针尾部解析出实体编号
            ref_num = table_data['self_ref'].split('/')[-1]
            ref_num = int(ref_num)

            # 6. 汇总成一张表的完整对象并入结果列表
            table_obj = {
                'table_id': ref_num,
                'page': table_page_num,
                'bbox': table_bbox,
                '#-rows': nrows,
                '#-cols': ncols,
                'markdown': table_md,
                'html': table_html,
                'json': table_json_obj
            }
            assembled_tables.append(table_obj)
        return assembled_tables

    def _table_to_md(self, table):
        """把 docling 表格网格转为 github 风格 markdown（首行作为表头）。"""
        # 1. 从 docling 网格逐行取单元格文本，组成二维数组
        table_data = []
        for row in table['data']['grid']:
            table_row = [cell['text'] for cell in row]
            table_data.append(table_row)

        # 2. 满足「既有表头行也有数据行」时，把首行当作表头转 github 风格 markdown
        if len(table_data) > 1 and len(table_data[0]) > 0:
            try:
                md_table = tabulate(
                    table_data[1:], headers=table_data[0], tablefmt="github"
                )
            except ValueError:
                # 3. tabulate 默认会把纯数字单元格数值化（去千分位/科学计数），
                #    年报数字常被误伤；回退为 disable_numparse 保留单元格原文
                md_table = tabulate(
                    table_data[1:],
                    headers=table_data[0],
                    tablefmt="github",
                    disable_numparse=True,
                )
        else:
            # 4. 只有一行（无表头）时：无表头直接输出
            md_table = tabulate(table_data, tablefmt="github")

        return md_table

    def assemble_pictures(self, data):
        """汇出全部图片：记录页号/bbox，并把图中文字（如表格截图被识别为图片时的文本）捞出来。"""
        assembled_pictures = []
        # 1. 逐张图片处理：先解引用 children 指针，捞出图片内的文字子块
        for i, picture in enumerate(data['pictures']):
            children_list = self._process_picture_block(picture, data)

            # 2. 规范化 picture_id：同表格规则，从 self_ref 指针尾部取编号
            ref_num = picture['self_ref'].split('/')[-1]
            ref_num = int(ref_num)

            # 3. 版位信息：页号 + bbox（拍平成 [l, t, r, b]，口径与表格一致）
            picture_page_num = picture['prov'][0]['page_no']
            picture_bbox = picture['prov'][0]['bbox']
            picture_bbox = [
                picture_bbox['l'],
                picture_bbox['t'],
                picture_bbox['r'],
                picture_bbox['b']
            ]

            # 4. 组装单张图片对象（children 只含文本子块，不含图像数据本身）
            picture_obj = {
                'picture_id': ref_num,
                'page': picture_page_num,
                'bbox': picture_bbox,
                'children': children_list,
            }
            assembled_pictures.append(picture_obj)
        return assembled_pictures
    
    def _process_picture_block(self, picture, data):
        """解引用图片的 children 指针，只保留其中的文本子块（图像数据本身不保留）。"""
        children_list = []

        # 1. 只处理 $ref 指针元素；图片子块中的普通字典（非文本引用等）直接跳过
        for item in picture['children']:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)

                # 2. 仅解引用文本引用：复用 _process_text_reference 生成文本 content 项
                if ref_type == 'texts':
                    content_item = self._process_text_reference(ref_num, data)

                    children_list.append(content_item)

        return children_list
