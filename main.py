"""
RAG Challenge 管道 CLI 入口：把 src/pipeline.py 的各离线/在线阶段暴露为一组 click 子命令。

职责定位:
    本文件只做「命令解析 + 当前目录解析 + 委托调用」，不含任何业务逻辑；
    业务编排全部在 src/pipeline.py 的 Pipeline 中。

数据流位置（磁盘文件即阶段间接口）:
    必须以「数据目录」为当前工作目录运行（例如 cd data/test_set 后执行），目录契约:
        - subset.csv / subset.json   报告元信息（sha1 -> company_name），parse-pdfs 消费
        - pdf_reports/*.pdf          原始年报，parse-pdfs 消费
        - questions.json             待回答问题集，process-questions 消费
    产物:
        - debug_data/ 下各阶段 JSON（01_parsed_reports / 02_merged_reports / 03_reports_markdown）
        - databases/ 下 chunked_reports 与 vector_dbs
        - 当前目录的 answers{config_suffix}.json（提交文件）及同名 *_debug.json

核心依赖与副作用:
    - 每个子命令在单个进程内执行一个阶段，阶段间通过磁盘接力，可中断后按序重跑；
    - 网络 I/O 集中在被调模块内（docling 模型下载、各 LLM 厂商 API），密钥由 .env 注入；
    - 无并发（并发发生在 Pipeline 内部的多进程/多线程）。
"""
import click
from pathlib import Path
from src.pipeline import Pipeline, configs, preprocess_configs

@click.group()
def cli():
    """RAG 管道命令行组：每个子命令对应一个可独立重跑的离线/在线阶段。

    各阶段通过磁盘文件接力（见 Pipeline 各方法注释），因此命令可以
    任意顺序单独执行，例如解析中断后可直接重跑 parse-pdfs。
    """
    pass

@cli.command()
def download_models():
    """预下载 docling 所需模型（一次性暖机）。

    docling 在首次解析时才从 HuggingFace 拉模型，把下载前置到本命令，
    避免正式跑批时因网络抖动卡死整个解析流程。
    """
    click.echo("Downloading docling models...")
    Pipeline.download_docling_models()

@cli.command()
@click.option('--parallel/--sequential', default=True, help='Run parsing in parallel or sequential mode')
@click.option('--chunk-size', default=2, help='Number of PDFs to process in each worker')
@click.option('--max-workers', default=10, help='Number of parallel worker processes')
def parse_pdfs(parallel, chunk_size, max_workers):
    """将 pdf_reports/ 下的年报解析为结构化 JSON（离线阶段 01）。

    前置: 当前目录含 subset.csv（用于 sha1 -> 公司名映射）与 pdf_reports/*.pdf；
          首次运行需先执行 download-models 完成 docling 模型暖机。
    产物: debug_data/01_parsed_reports/*.json（供后续所有阶段消费），
          原始 docling 输出存于 debug_data/01_parsed_reports_debug（仅排障用）。
    并行模式按 chunk 切分 PDF 后使用多进程，适合 GPU 单机高吞吐；
    失败计数大于 0 时整个命令抛 RuntimeError 失败退出。
    """
    root_path = Path.cwd()
    pipeline = Pipeline(root_path)
    
    click.echo(f"Parsing PDFs (parallel={parallel}, chunk_size={chunk_size}, max_workers={max_workers})")
    pipeline.parse_pdf_reports(parallel=parallel, chunk_size=chunk_size, max_workers=max_workers)

@cli.command()
@click.option('--max-workers', default=10, help='Number of workers for table serialization')
def serialize_tables(max_workers):
    """对 01_parsed_reports 中每张表格调用 LLM 做上下文无关化序列化（可选阶段）。

    前置: 已完成 parse-pdfs 且配置了 OPENAI_API_KEY（固定使用 gpt-4o-mini，温度 0）。
    效果: 就地改写 01 目录下的 JSON，为每个 table 增加 serialized 字段
          （信息块列表，供 ser_tab 系列配置的建库流程使用）；
          不含 serialized 字段的解析结果在后续 merge 阶段会静默回退为纯 markdown 表格。
    副作用: 临时请求/结果文件写入 ./temp/ 与 cwd（线程私有命名，处理后删除）。
    """
    root_path = Path.cwd()
    pipeline = Pipeline(root_path)
    
    click.echo(f"Serializing tables (max_workers={max_workers})...")
    pipeline.serialize_tables(max_workers=max_workers)

@cli.command()
@click.option('--config', type=click.Choice(['ser_tab', 'no_ser_tab']), default='no_ser_tab', help='Configuration preset to use')
def process_reports(config):
    """在 01 解析结果上执行 合并 -> Markdown 导出 -> 切分 -> 建向量库（离线阶段 02-05）。

    前置: debug_data/01_parsed_reports 已就绪；
          config=ser_tab 时要求已执行 serialize-tables（缺 serialized 字段会回退为纯表格 markdown）。
    产物: debug_data/02_merged_reports(_ser_tab)、03_reports_markdown(_ser_tab)、
          databases(_ser_tab)/chunked_reports 与 vector_dbs。
    注意: 本流程不建 BM25 库（create_bm25_db 为独立步骤，当前问答链路未启用 BM25 召回）。
    """
    root_path = Path.cwd()
    run_config = preprocess_configs[config]
    pipeline = Pipeline(root_path, run_config=run_config)
    
    click.echo(f"Processing parsed reports (config={config})...")
    pipeline.process_parsed_reports()

@cli.command()
@click.option('--config', type=click.Choice(['base', 'pdr', 'max', 'max_no_ser_tab', 'max_nst_o3m', 'max_st_o3m', 'ibm_llama70b', 'ibm_llama8b', 'gemini_thinking']), default='base', help='Configuration preset to use')
def process_questions(config):
    """读取 questions.json，基于指定实验预设逐题检索 + 生成答案（在线阶段）。

    前置: 与 config 匹配的向量库已构建（库目录按 ser_tab 开关取 databases 或 databases_ser_tab），
          配置了对应厂商的 API key；configs 键见 src/pipeline.py 底部。
    产物: answers{config_suffix}.json（提交格式）与 answers{config_suffix}_debug.json
          （含逐题推理过程与 token 统计）；同名文件已存在时自动追加 _NN 编号，绝不覆盖旧结果。
    """
    root_path = Path.cwd()
    run_config = configs[config]
    pipeline = Pipeline(root_path, run_config=run_config)
    
    click.echo(f"Processing questions (config={config})...")
    pipeline.process_questions()

if __name__ == '__main__':
    cli()