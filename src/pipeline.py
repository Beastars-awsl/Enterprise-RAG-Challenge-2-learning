"""
Pipeline 编排层与「实验配置仓库」，全系统的中枢模块。

职责定位:
    - PipelineConfig : 目录布局与磁盘契约。所有阶段产物的路径、命名后缀规则集中在此，
                       下游模块只认这些路径，不感知彼此存在（依赖倒置点）。
    - RunConfig      : 一次实验的全部策略开关（是否序列化表格、是否父文档召回、是否 LLM
                       重排、模型与采样规模、提交元信息…），模块底部 configs / preprocess_configs
                       即历次实验预设的存档。
    - Pipeline       : 面向 main.py 与模块 __main__ 的粗粒度门面，每个方法 = 一个可独立
                       重跑的流水线阶段。

数据流位置（阶段间通过磁盘 JSON 接力，这是整套代码可断点续跑的设计根基）:
    pdf_reports/*.pdf
      -> [pdf_parsing]      -> debug_data/01_parsed_reports(+_debug)    逐块解析结果
      -> [parsed_reports_merging] -> debug_data/02_merged_reports(_ser_tab) 按页 markdown 文本
      -> [export]           -> debug_data/03_reports_markdown(_ser_tab) 全文 md（审查/全上下文问答）
      -> [text_splitter]    -> databases(_ser_tab)/chunked_reports      检索分块（含 token 数元数据）
      -> [ingestion]        -> databases(_ser_tab)/vector_dbs(.faiss)/bm25_dbs(.pkl)
      -> [questions_processing] -> answers{config_suffix}.json(+_debug) 提交产物
    命名约定: 目录后缀 _ser_tab 标识该产物来自「表格序列化」变体；answers 文件后缀来自
    config_suffix —— 不同策略的产物与答案文件互不覆盖，保证可并排对比（A/B 实验）。

核心依赖与副作用:
    - 构造 Pipeline 时可能改写数据目录（subset.json -> subset.csv）；
    - 各方法为阻塞式串行编排，并发只在方法内部（多进程解析/多线程表格序列化与问答）；
    - 网络依赖: docling 模型（首次）、OpenAI/IBM/Gemini API（密钥来自 .env）。
"""
from dataclasses import dataclass
from pathlib import Path
from pyprojroot import here
import logging
import os
import json
import pandas as pd

from src.pdf_parsing import PDFParser
from src.parsed_reports_merging import PageTextPreparation
from src.text_splitter import TextSplitter
from src.ingestion import VectorDBIngestor
from src.ingestion import BM25Ingestor
from src.questions_processing import QuestionsProcessor
from src.tables_serialization import TableSerializer

# 注：本类无注解字段且已手写 __init__，@dataclass 实际不会生成任何方法（历史遗留装饰器）。
@dataclass
class PipelineConfig:
    """数据目录布局解析器：把 root_path 展开为一棵固定命名的路径树。

    设计意图:
        将「阶段产物放在哪里」固化为唯一事实来源。解析(01)、合并(02/03)、检索库
        (databases) 三组路径的命名规则不同（见 __init__），改目录布局只需动本类。
    """

    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", serialized: bool = False, config_suffix: str = ""):
        self.root_path = root_path
        # 后缀规则一：serialized 变体的「合并产物」与「检索库」目录统一加 _ser_tab，
        # 与共享输入 01_parsed_reports 解耦 —— 同一份解析结果可派生多套下游产物并排运行。
        suffix = "_ser_tab" if serialized else ""

        self.subset_path = root_path / subset_name
        self.questions_file_path = root_path / questions_file_name
        self.pdf_reports_dir = root_path / pdf_reports_dir_name
        
        # 后缀规则二：答案/提交文件按 config_suffix 区分（answers_max_nst_o3m.json）；
        # 同名文件再次运行时不覆盖，由 Pipeline._get_next_available_filename 追加 _NN 编号。
        self.answers_file_path = root_path / f"answers{config_suffix}.json"
        self.debug_data_path = root_path / "debug_data"
        self.databases_path = root_path / f"databases{suffix}"
        
        self.vector_db_dir = self.databases_path / "vector_dbs"
        self.documents_dir = self.databases_path / "chunked_reports"
        self.bm25_db_path = self.databases_path / "bm25_dbs"

        self.parsed_reports_dirname = "01_parsed_reports"
        self.parsed_reports_debug_dirname = "01_parsed_reports_debug"
        self.merged_reports_dirname = f"02_merged_reports{suffix}"
        self.reports_markdown_dirname = f"03_reports_markdown{suffix}"

        self.parsed_reports_path = self.debug_data_path / self.parsed_reports_dirname
        self.parsed_reports_debug_path = self.debug_data_path / self.parsed_reports_debug_dirname
        self.merged_reports_path = self.debug_data_path / self.merged_reports_dirname
        self.reports_markdown_path = self.debug_data_path / self.reports_markdown_dirname

@dataclass
class RunConfig:
    """单次实验的策略预设（纯数据对象，只被 Pipeline 与 QuestionsProcessor 消费）。

    字段组说明:
        - use_serialized_tables: 表格是否走 LLM 序列化变体 —— 会改变 02 合并产物与检索库
          的目录后缀，因此同一配置必须配合已生成对应产物的数据目录使用；
        - parent_document_retrieval / llm_reranking / top_n_retrieval / llm_reranking_sample_size:
          召回链路开关。llm_reranking 开启时 HybridRetriever 先取
          llm_reranking_sample_size 个向量候选再 LLM 打分截断到 top_n_retrieval，
          故要求 sample_size >= top_n；
        - full_context: 跳过检索，把整本报告全部页面塞给模型（Gemini 大上下文配置，
          严格说已不属于 RAG）；
        - api_provider / answering_model: 问答厂商与模型；parallel_requests 为问答线程数
          （API 限流较严的厂商如 IBM/Gemini 会主动调低）；
        - config_suffix / team_email / submission_name / pipeline_details: 影响输出文件名
          与提交文件元信息；submission_file=False 时只写 *_debug.json。
    """

    # 是否使用 LLM 序列化后的表格文本替代原始表格
    use_serialized_tables: bool = False 
    # 是否启用 父文档检索
    parent_document_retrieval: bool = False
    # 是否使用 向量数据库 做语义检索
    use_vector_dbs: bool = True
    # 是否使用 BM25 关键词索引（传统稀疏检索）
    use_bm25_db: bool = False
    # 是否用 LLM 对初步检索结果进行重排序。
    llm_reranking: bool = False
    # 重排序前的 候选池大小
    llm_reranking_sample_size: int = 30
    # 最终喂给 LLM 的 上下文条数
    top_n_retrieval: int = 10
    # 问答阶段的 并发请求数
    parallel_requests: int = 10
    # 提交文件元信息中的联系邮箱
    team_email: str = "79250515615@yandex.com"
    # 提交文件的方案名
    submission_name: str = "Ilia_Ris vDB + SO CoT"
    # 自由描述，写进提交文件
    pipeline_details: str = ""
    # 是否生成 正式提交文件
    submission_file: bool = True
    # 跳过检索，把整本报告所有页面塞给 LLM
    full_context: bool = False
    # 问答调用的 API 厂商
    api_provider: str = "openai"
    # 用于生成最终答案的 LLM 模型
    answering_model: str = "gpt-4o-mini-2024-07-18" #or "gpt-4o-2024-08-06"
    # 输出文件的 自定义后缀
    config_suffix: str = ""

class Pipeline:
    """阶段编排门面：把 解析->合并->切分->建库->问答 拆成可独立调用、可断点续跑的方法。

    设计意图:
        main.py 的每个子命令与本类的公开方法一一对应；所有阶段共享同一份
        self.paths（PipelineConfig），保证任意组合的执行顺序都能落到正确目录。

    前置假定:
        - 各方法只对其所需「上游产物」的存在负责，不校验下游状态；
        - 同一 Pipeline 实例上不要并发调用多个方法（非线程安全）。

    用法: 见本文件底部 __main__（逐个取消注释单阶段执行，或经 main.py CLI）。
    """

    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", run_config: RunConfig = RunConfig()):
        self.run_config = run_config
        self.paths = self._initialize_paths(root_path, subset_name, questions_file_name, pdf_reports_dir_name)
        # 数据目录里可能只有旧版 subset.json —— 进到流程前先归一化成 CSV，
        # 否则下游 pd.read_csv 直接崩；文件已存在时本方法为空操作。
        self._convert_json_to_csv_if_needed()

    def _initialize_paths(self, root_path: Path, subset_name: str, questions_file_name: str, pdf_reports_dir_name: str) -> PipelineConfig:
        """Initialize paths configuration based on run config settings"""
        return PipelineConfig(
            root_path=root_path,
            subset_name=subset_name,
            questions_file_name=questions_file_name,
            pdf_reports_dir_name=pdf_reports_dir_name,
            serialized=self.run_config.use_serialized_tables,
            config_suffix=self.run_config.config_suffix
        )

    def _convert_json_to_csv_if_needed(self):
        """
        Checks if subset.json exists in root dir and subset.csv is absent.
        If so, converts the JSON to CSV format.
        """
        json_path = self.paths.root_path / "subset.json"
        csv_path = self.paths.root_path / "subset.csv"
        
        if json_path.exists() and not csv_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                df = pd.DataFrame(data)
                
                df.to_csv(csv_path, index=False)
                
            except Exception as e:
                print(f"Error converting JSON to CSV: {str(e)}")

# Docling automatically downloads some models from huggingface when first used
# I wanted to download them prior to running the pipeline and created this crutch
    @staticmethod
    def download_docling_models(): 
        logging.basicConfig(level=logging.DEBUG)
        parser = PDFParser(output_dir=here())
        parser.parse_and_export(input_doc_paths=[here() / "src/dummy_report.pdf"])

    def parse_pdf_reports_sequential(self):
        """串行解析 pdf_reports/*.pdf -> 01_parsed_reports（少量 PDF / 排障时使用）。

        csv_metadata_path 提供 sha1 -> 公司名映射，写进每份报告的 metainfo；
        PDFParser 输出两份文件：结构化 JSON（本流程消费）与 docling 原始输出（排障用）。
        """
        logging.basicConfig(level=logging.DEBUG)

        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_reports_path,
            csv_metadata_path=self.paths.subset_path
        )
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path
            
        pdf_parser.parse_and_export(doc_dir=self.paths.pdf_reports_dir)
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

    def parse_pdf_reports_parallel(self, chunk_size: int = 2, max_workers: int = 10):
        """多进程并行解析 pdf_reports/*.pdf -> 01_parsed_reports。

        进程模型: 每个 worker 在独立子进程内自建 PDFParser —— docling 的
        DocumentConverter 不可跨进程 pickle，_process_chunk 因此只传配置参数。
        GPU 场景下 worker 数超过 1 时按进程切分显存/CPU 线程（num_threads 环境变量）。

        Args:
            chunk_size: 每个 worker 一次领取的 PDF 数量（越大则进程切换越少、粒度越粗）
            max_workers: 并行 worker 进程数上限
        """
        logging.basicConfig(level=logging.DEBUG)

        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_reports_path,
            csv_metadata_path=self.paths.subset_path
        )
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path

        input_doc_paths = list(self.paths.pdf_reports_dir.glob("*.pdf"))
        
        pdf_parser.parse_and_export_parallel(
            input_doc_paths=input_doc_paths,
            optimal_workers=max_workers,
            chunk_size=chunk_size
        )
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

    def serialize_tables(self, max_workers: int = 10):
        """就地改写 01_parsed_reports/*.json：为每张表追加 LLM 序列化信息块。

        仅在需要运行 use_serialized_tables 配置（ser_tab 系列）前调用；
        逐表做一次 gpt-4o-mini 调用（温度 0），文件级并行 + 线程内独立事件循环。

        Args:
            max_workers: 并发处理文件的线程数（限流与账单由内部请求队列控制）
        """
        serializer = TableSerializer()
        serializer.process_directory_parallel(
            self.paths.parsed_reports_path,
            max_workers=max_workers
        )

    def merge_reports(self):
        """01 逐块 JSON -> 02_merged_reports(_ser_tab)：按页拼成 markdown 风格长文本。

        消费方契约（PageTextPreparation 保证）:
            - 丢弃页眉页脚与图片引用，表格就地渲染为 markdown（ser_tab 变体下为
              markdown + "Description of the table entities:" 序列化描述，二者叠放保证
              该页文本自含可答）；
            - 页码沿用 docling 的物理页序号（1 起始），全链路内部统一按此口径，
              仅提交文件在 _post_process_submission_answers 中减 1 转 0 起始。
        """
        ptp = PageTextPreparation(use_serialized_tables=self.run_config.use_serialized_tables)
        _ = ptp.process_reports(
            reports_dir=self.paths.parsed_reports_path,
            output_dir=self.paths.merged_reports_path
        )
        print(f"Reports saved to {self.paths.merged_reports_path}")

    def export_reports_to_markdown(self):
        """02 逻辑的纯文本导出 -> 03_reports_markdown(_ser_tab)/*.md（人工审查 / 全上下文问答）。

        与 merge_reports 共享 PageTextPreparation 的清洗与排版逻辑，但每页加
        "# Page N" 分隔标记，输出直接可读的整册 markdown。
        """
        ptp = PageTextPreparation(use_serialized_tables=self.run_config.use_serialized_tables)
        ptp.export_to_markdown(
            reports_dir=self.paths.parsed_reports_path,
            output_dir=self.paths.reports_markdown_path
        )
        print(f"Reports saved to {self.paths.reports_markdown_path}")

    def chunk_reports(self, include_serialized_tables: bool = False):
        """02 整页文本 -> databases(_ser_tab)/chunked_reports：按 300 token 重叠切块。

        切块元数据（id/type/page/length_tokens/text）是后续建库与检索的唯一数据源；
        chunks 顺序即向量库索引顺序（见 VectorDBIngestor / VectorRetriever 的契约注释）。

        Args:
            include_serialized_tables: True 时额外把每页序列化表格作为独立 chunk
                （type=serialized_table）插入该页内容块之后 —— 注意 process_parsed_reports
                未传该参数，当前默认流程靠 02 合并文本已内嵌序列化描述，此开关为休眠特性。
        """
        text_splitter = TextSplitter()

        serialized_tables_dir = None
        if include_serialized_tables:
            serialized_tables_dir = self.paths.parsed_reports_path

        text_splitter.split_all_reports(
            self.paths.merged_reports_path,
            self.paths.documents_dir,
            serialized_tables_dir
        )
        print(f"Chunked reports saved to {self.paths.documents_dir}")

    def create_vector_dbs(self):
        """对 chunked_reports 逐报告生成 FAISS 向量库 -> vector_dbs/{sha1_name}.faiss。

        chunks 顺序即索引行序，行号与 chunk id 一一对应（检索端依赖此契约取回 chunk）；
        embedding 走 OpenAI text-embedding-3-large（按批 1024 条，内置 20s 重试）。
        """
        input_dir = self.paths.documents_dir
        output_dir = self.paths.vector_db_dir

        vdb_ingestor = VectorDBIngestor()
        vdb_ingestor.process_reports(input_dir, output_dir)
        print(f"Vector databases created in {output_dir}")
    
    def create_bm25_db(self):
        """对 chunked_reports 逐报告构建 BM25 索引 -> bm25_dbs/{sha1_name}.pkl。

        目前问答链路（QuestionsProcessor）未挂接 BM25 召回，BM25Retriever 为
        保留的独立检索器；分词沿用英文空格切词（rank_bm25 默认口径）。
        """
        input_dir = self.paths.documents_dir
        output_file = self.paths.bm25_db_path

        bm25_ingestor = BM25Ingestor()
        bm25_ingestor.process_reports(input_dir, output_file)
        print(f"BM25 database created at {output_file}")
    
    def parse_pdf_reports(self, parallel: bool = True, chunk_size: int = 2, max_workers: int = 10):
        """解析入口分发器：parallel 决定走多进程还是单进程路径（两者产物格式完全一致）。"""
        if parallel:
            self.parse_pdf_reports_parallel(chunk_size=chunk_size, max_workers=max_workers)
        else:
            self.parse_pdf_reports_sequential()
    
    def process_parsed_reports(self):
        """在已就绪的 01_parsed_reports 上跑完 02 合并 -> 03 Markdown -> 切分 -> 建向量库。

        完整流程为:
        1. merge_reports      01 逐块 JSON -> 02 按页 markdown 文本（目录后缀随 ser_tab 开关）
        2. export_to_markdown 02 -> 03 整册可读 markdown
        3. chunk_reports      02 整页 -> 300 token 重叠分块（含 token 数元数据）
        4. create_vector_dbs  分块 -> OpenAI embedding + FAISS
        注意:
        - 若 run_config.use_serialized_tables=True 而 01 缺少 serialized 字段（未先跑
          serialize_tables），第 1 步会自动回退为纯 markdown 表格，不报错 —— 需自行核对；
        - 建 BM25 库不在此流程内（见 create_bm25_db 的注释）。
        """
        print("Starting reports processing pipeline...")
        
        print("Step 1: Merging reports...")
        self.merge_reports()
        
        print("Step 2: Exporting reports to markdown...")
        self.export_reports_to_markdown()
        
        print("Step 3: Chunking reports...")
        self.chunk_reports()
        
        print("Step 4: Creating vector databases...")
        self.create_vector_dbs()
        
        print("Reports processing pipeline completed successfully!")
        
    def _get_next_available_filename(self, base_path: Path) -> Path:
        """
        Returns the next available filename by adding a numbered suffix if the file exists.
        Example: If answers.json exists, returns answers_01.json, etc.
        """
        if not base_path.exists():
            return base_path
            
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        
        counter = 1
        while True:
            new_filename = f"{stem}_{counter:02d}{suffix}"
            new_path = parent / new_filename
            
            if not new_path.exists():
                return new_path
            counter += 1

    def process_questions(self):
        """读取 questions.json 逐题执行 检索 -> 重排(可选) -> LLM 生成，产出提交与 debug 两份 JSON。

        前置: 与 run_config 匹配的向量库/目录已构建（库目录随 use_serialized_tables
              取 databases 或 databases_ser_tab）；配置好对应厂商 API key。
        行为: run_config 的全部召回/生成开关透传给 QuestionsProcessor；输出路径先经
              _get_next_available_filename 去重，已存在的旧答案文件不会被覆盖。
        """
        # 1. 实例化问答处理器：把路径契约（库/文档/题目文件）与 run_config 的全部召回
        #    /生成开关一次性透传 —— Pipeline 只做编排不做策略，检索与生成的实际逻辑
        #    全部下沉到 QuestionsProcessor，保证本方法与策略变化解耦
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=self.paths.questions_file_path,
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=self.run_config.parallel_requests,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context
        )

        # 2. 先去重输出路径再开跑：同名 answers 文件已存在时追加 _NN 编号（answers_01.json），
        #    防止重复实验覆盖旧答案 —— 历史提交是不可再生的实验产物，必须保护
        output_path = self._get_next_available_filename(self.paths.answers_file_path)

        # 3. 执行问答主流程（内部并发由 run_config.parallel_requests 控制），产出两份文件：
        #    提交用 answers{suffix}.json 与含检索/中间过程细节的 *_debug.json；
        #    submission_file=False 时只写 debug 版，适合试跑省钱
        _ = processor.process_all_questions(
            output_path=output_path,
            submission_file=self.run_config.submission_file,
            team_email=self.run_config.team_email,
            submission_name=self.run_config.submission_name,
            pipeline_details=self.run_config.pipeline_details
        )
        # 4. 落盘完成后打印实际写入路径（注意是去重后的 output_path，而非配置里的原始路径）
        print(f"Answers saved to {output_path}")


# 建库（预处理）阶段的实验预设：只切换序列化表格开关，决定 02/数据库目录取哪个变体。
preprocess_configs = {"ser_tab": RunConfig(use_serialized_tables=True),
                      "no_ser_tab": RunConfig(use_serialized_tables=False)}

base_config = RunConfig(
    parallel_requests=10,
    submission_name="Ilia Ris v.0",
    pipeline_details="Custom pdf parsing + vDB + Router + SO CoT; llm = GPT-4o-mini",
    config_suffix="_base"
)

parent_document_retrieval_config = RunConfig(
    parent_document_retrieval=True,
    parallel_requests=20,
    submission_name="Ilia Ris v.1",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT; llm = GPT-4o",
    answering_model="gpt-4o-2024-08-06",
    config_suffix="_pdr"
)

max_config = RunConfig(
    use_serialized_tables=True,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=20,
    submission_name="Ilia Ris v.2",
    pipeline_details="Custom pdf parsing + table serialization + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = GPT-4o",
    answering_model="gpt-4o-2024-08-06",
    config_suffix="_max"
)

max_no_ser_tab_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=20,
    submission_name="Ilia Ris v.3",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = GPT-4o",
    answering_model="gpt-4o-2024-08-06",
    config_suffix="_max_no_ser_tab"
)

max_nst_o3m_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=25,
    submission_name="Ilia Ris v.4",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = o3-mini",
    answering_model="o3-mini-2025-01-31",
    config_suffix="_max_nst_o3m"
)

max_st_o3m_config = RunConfig(
    use_serialized_tables=True,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=25,
    submission_name="Ilia Ris v.5",
    pipeline_details="Custom pdf parsing + tables serialization + Router + vDB + Parent Document Retrieval + reranking + SO CoT; llm = o3-mini",
    answering_model="o3-mini-2025-01-31",
    config_suffix="_max_st_o3m"
)

ibm_llama70b_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=False,
    parallel_requests=10,
    submission_name="Ilia Ris v.6",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT + SO reparser; IBM WatsonX llm = llama-3.3-70b-instruct",
    api_provider="ibm",
    answering_model="meta-llama/llama-3-3-70b-instruct",
    config_suffix="_ibm_llama70b"
)

ibm_llama8b_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=False,
    parallel_requests=10,
    submission_name="Ilia Ris v.7",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT + SO reparser; IBM WatsonX llm = llama-3.1-8b-instruct",
    api_provider="ibm",
    answering_model="meta-llama/llama-3-1-8b-instruct",
    config_suffix="_ibm_llama8b"
)

gemini_thinking_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=False,
    parallel_requests=1,
    full_context=True,
    submission_name="Ilia Ris v.8",
    pipeline_details="Custom pdf parsing + Full Context + Router + SO CoT + SO reparser; llm = gemini-2.0-flash-thinking-exp-01-21",
    api_provider="gemini",
    answering_model="gemini-2.0-flash-thinking-exp-01-21",
    config_suffix="_gemini_thinking_fc"
)

gemini_flash_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=False,
    parallel_requests=1,
    full_context=True,
    submission_name="Ilia Ris v.9",
    pipeline_details="Custom pdf parsing + Full Context + Router + SO CoT + SO reparser; llm = gemini-2.0-flash",
    api_provider="gemini",
    answering_model="gemini-2.0-flash",
    config_suffix="_gemini_flash_fc"
)

max_nst_o3m_config_big_context = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=5,
    llm_reranking_sample_size=36,
    top_n_retrieval=14,
    submission_name="Ilia Ris v.10",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = o3-mini; top_n = 14; topn for rerank = 36",
    answering_model="o3-mini-2025-01-31",
    config_suffix="_max_nst_o3m_bc"
)

ibm_llama70b_config_big_context = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=5,
    llm_reranking_sample_size=36,
    top_n_retrieval=14,
    submission_name="Ilia Ris v.11",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = llama-3.3-70b-instruct; top_n = 14; topn for rerank = 36",
    api_provider="ibm",
    answering_model="meta-llama/llama-3-3-70b-instruct",
    config_suffix="_ibm_llama70b_bc"
)

gemini_thinking_config_big_context = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    parallel_requests=1,
    top_n_retrieval=30,
    submission_name="Ilia Ris v.12",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT; llm = gemini-2.0-flash-thinking-exp-01-21; top_n = 30;",
    api_provider="gemini",
    answering_model="gemini-2.0-flash-thinking-exp-01-21",
    config_suffix="_gemini_thinking_bc"
)

configs = {"base": base_config,
           "pdr": parent_document_retrieval_config,
           "max": max_config, 
           "max_no_ser_tab": max_no_ser_tab_config,
           "max_nst_o3m": max_nst_o3m_config, # This configuration returned the best results
           "max_st_o3m": max_st_o3m_config,
           "ibm_llama70b": ibm_llama70b_config, # This one won't work, because ibm api was avaliable only while contest was running
           "ibm_llama8b": ibm_llama8b_config, # This one won't work, because ibm api was avaliable only while contest was running
           "gemini_thinking": gemini_thinking_config}


# You can run any method right from this file with 
# python .\src\pipeline.py
# Just uncomment the method you want to run
# You can also change the run_config to try out different configurations
if __name__ == "__main__":
    root_path = here() / "data" / "test_set"
    pipeline = Pipeline(root_path, run_config=max_nst_o3m_config)
    
    
    # This method parses pdf reports into a jsons. It creates jsons in the debug/data_01_parsed_reports. These jsons used in the next steps. 
    # It also stores raw output of docling in debug/data_01_parsed_reports_debug, these jsons contain a LOT of metadata, and not used anywhere
    # pipeline.parse_pdf_reports_sequential() 
    
    
    # This method should be called only if you want run configs with serialized tables
    # It modifies the jsons in the debug/data_01_parsed_reports, adding a new field "serialized_table" to each table
    # pipeline.serialize_tables(max_workers=5) 
    
    
    # This method converts jsons from the debug/data_01_parsed_reports into much simpler jsons, that is a list of pages in markdown
    # New jsons can be found in debug/data_02_merged_reports
    # pipeline.merge_reports() 


    # This method exports the reports into plain markdown format. They used only for review and for full text search config: gemini_thinking_config
    # New files can be found in debug/data_03_reports_markdown
    # pipeline.export_reports_to_markdown() 
    

    # This method splits the reports into chunks, that are used for vectorization
    # New jsons can be found in databases/chunked_reports
    # pipeline.chunk_reports() 
    
    
    # This method creates vector databases from the chunked reports
    # New files can be found in databases/vector_dbs
    # pipeline.create_vector_dbs() 
    
    
    # This method processes the questions and answers
    # Questions processing logic depends on the run_config
    # pipeline.process_questions() 