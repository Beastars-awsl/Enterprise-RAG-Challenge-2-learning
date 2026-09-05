"""
表格序列化阶段（可选离线增强）：把年报表格交给 LLM 转成「上下文无关信息块」。

职责定位:
    TableSerializer  逐表调用 gpt-4o-mini（温度 0），要求模型结合表格 HTML 与其所在页
                     上下文，产出 subject_core_entity + information_block 形式的自包含
                     文本块 —— 解决纯 markdown 表格切块后「只有数字没有表头/单位」导致
                     检索与问答失配的问题（比赛胜出的关键技巧之一）。
    TableSerialization 声明该任务的 system prompt 与结构化输出 schema（即信息块的业务语义）。

数据流位置:
    输入: debug_data/01_parsed_reports/*.json（须先跑过 parse-pdfs；表格含 html 表示）；
    输出: 就地改写同一批 JSON —— 每个 table 增加 serialized 字段
          {subject_core_entities_list, relevant_headers_list, information_blocks[]}，
          供 parsed_reports_merging 在 use_serialized_tables=True 时渲染进 02 页面文本。

核心依赖与副作用:
    - 每张表一次 OpenAI 结构化输出调用，费用随表数线性增长，且逐文件落盘（可断点重跑）；
    - 并发模型: 多线程 x 每线程独立 asyncio 事件循环，批量请求经 AsyncOpenaiProcessor
      （api_request_parallel_processor）按限流窗口调度；
    - 临时 jsonl（请求/结果）落在 ./temp/ 与 cwd（以线程 id 隔离命名，跑完即删）；
    - 日志经 TqdmLoggingHandler 桥接进 message_queue，由主线程在 tqdm 刷新间隙打印，
      避免并发日志把进度条刷烂（线程安全约定: 日志只入队，打印只发生在主线程轮询处）。
"""
import os
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, List, Union, Literal
from pydantic import BaseModel, Field
from openai import OpenAI
from src.api_requests import BaseOpenaiProcessor, AsyncOpenaiProcessor
import tiktoken
from tqdm import tqdm
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import time

message_queue = Queue()

class TqdmLoggingHandler(logging.Handler):
    """把 logging 记录转发到线程安全的 Queue，而非直接打印。

    并发线程直接 print 会与 tqdm 进度条交错输出；本 handler 只负责入队，
    由主线程在 process_messages()（进度条刷新间隙）统一取出打印。
    """

    def emit(self, record):
        try:
            msg = self.format(record)
            message_queue.put((record.levelno, msg))
        except Exception:
            self.handleError(record)

def process_messages():
    """主线程专用：排空队列中的日志并经由 tqdm.write 输出（不污染进度条）。"""
    while not message_queue.empty():
        level, msg = message_queue.get_nowait()
        tqdm.write(msg)

class TableSerializer(BaseOpenaiProcessor):
    """表格序列化器：目录级并行入口 + 单表序列化 + 结果回写。

    并发层次:
        文件间并行（ThreadPoolExecutor，max_workers 个文件同时跑）；
        文件内并行（每线程建独立事件循环跑 AsyncOpenaiProcessor 批量请求）。
        临时 jsonl 以 thread_id 命名，避免多线程写同一请求文件的竞态。
    """

    def __init__(self, preserve_temp_files: bool = True):
        super().__init__()
        self.preserve_temp_files = preserve_temp_files
        os.makedirs('./temp', exist_ok=True)
        
        self.logger = logging.getLogger('TableSerializer')
        self.logger.setLevel(logging.INFO)
        
        self.logger.handlers.clear()
        
        handler = TqdmLoggingHandler()
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        self.logger.addHandler(handler)
        
        self.logger.propagate = False

    def _get_table_context(self, json_report, target_table_index):
        """取目标表所在页的邻近文本块，作为 LLM 消歧的上下文（不跨页）。

        设计意图: 表格上方通常是标题/说明，下方常跟脚注或注释；把这些上下文随表发给
        模型，可让序列化块带上「这张表统计什么、单位是什么」等必要语境。取不到时返回
        空串，调用方仍会只拿表格本身序列化（容错而非报错）。

        Returns:
            (context_before, context_after)：表前/表后拼好的纯文本；任一侧不存在则为 ""。
        """
        table_info = next(table for table in json_report["tables"] if table["table_id"] == target_table_index)
        page_num = table_info["page"]

        page_content = next(
            (page["content"] for page in json_report["content"] if page["page"] == page_num),
            []
        )

        if not page_content:
            self.logger.warning(f"Page {page_num} not found for table {target_table_index}")
            return "", ""

        # Find position of target table in page_content
        current_table_position = -1
        for i, block in enumerate(page_content):
            if block["type"] == "table" and block.get("table_id") == target_table_index:
                current_table_position = i
                break

        # Find position of previous table if exists
        previous_table_position = -1
        for i in range(current_table_position-1, -1, -1):
            if page_content[i]["type"] == "table":
                previous_table_position = i
                break

        # Find position of next table if exists
        next_table_position = -1
        for i in range(current_table_position + 1, len(page_content)):
            if page_content[i]["type"] == "table":
                next_table_position = i
                break

        # Get blocks above current table
        start_position = previous_table_position + 1 if previous_table_position != -1 else 0
        context_before = page_content[start_position:current_table_position]

        # Get blocks after current table
        context_after = []
        if next_table_position == -1:
            # If no next table, take up to 3 blocks until end of page
            context_after = page_content[current_table_position + 1:current_table_position + 4]
        else:
            # If next table exists, take up to 3 blocks before it
            blocks_between = next_table_position - (current_table_position + 1)
            if blocks_between > 3:
                context_after = page_content[current_table_position + 1:current_table_position + 4]
            elif blocks_between > 1:
                context_after = page_content[current_table_position + 1:current_table_position + blocks_between]
            # blocks_between <= 1 时故意取空：紧贴下一张表的那一两个文本块很可能是
            # 下一张表的说明/表头前导，混入会污染当前表的序列化上下文。

        context_before = "\n".join(block.get("text", "") for block in context_before if "text" in block)
        context_after = "\n".join(block.get("text", "") for block in context_after if "text" in block)

        return context_before, context_after

    def _send_serialization_request(self, table, context_before, context_after):
        """对单张表发起一次结构化输出调用（同步版，串行/调试路径使用）。

        提示语把「上下文可能相关也可能无关」写明白，防止模型把表旁文本硬塞进块里；
        输入/输出 token 数按 gpt-4o 词表(o200k)自算并随响应一起统计，供成本核算。
        """
        user_prompt = ""
        
        if context_before:
            user_prompt += f'Here is additional text before the table that might be relevant (or not):\n"""{context_before}"""\n\n'
        
        user_prompt += f'Here is a table in HTML format:\n"""{table}"""'
        
        if context_after:
            user_prompt += f'\n\nHere is additional text after the table that might be relevant (or not):\n"""{context_after}"""'
        
        system_prompt = TableSerialization.system_prompt
        reponse_schema = TableSerialization.TableBlocksCollection

        answer_dict = self.send_message(
            model='gpt-4o-mini-2024-07-18',
            temperature=0,
            system_content=system_prompt,
            human_content=user_prompt,
            is_structured=True,
            response_format=reponse_schema
        )

        input_message = user_prompt + system_prompt + str(reponse_schema.schema())
        input_tokens = self.count_tokens(input_message)
        output_tokens = self.count_tokens(str(answer_dict))

        result = answer_dict
        return result
    
    def _serialize_table(self, json_report: dict, target_table_index: int) -> dict:
        """单表同步序列化：取邻近上下文 -> 组装请求 -> 返回结构化结果 dict。"""
        # Get the context surrounding the table
        context_before, context_after = self._get_table_context(json_report, target_table_index)
        
        # Get the table content
        table_info = next(table for table in json_report["tables"] if table["table_id"] == target_table_index)
        table_content = table_info["html"]
        
        # Serialize the table with its context
        result = self._send_serialization_request(
            table=table_content,
            context_before=context_before,
            context_after=context_after
        )
        
        return result

    def serialize_tables(self, json_report: dict) -> dict:
        """同步串行序列化整份报告的表格并就地写回 serialized 字段（供调试/小规模用）。

        Args:
            json_report: 01 解析 JSON（内存中的 dict，tables 会被就地修改）

        Returns:
            修改后的同一份 dict。
        """

        for table in json_report["tables"]:
            table_index = table["table_id"]
            
            # Get serialization results for current table
            serialization_result = self._serialize_table(
                json_report=json_report,
                target_table_index=table_index
            )
            
            # Add serialization results to the table info
            table["serialized"] = serialization_result
        
        return json_report

    async def async_serialize_tables(
        self, 
        json_report: dict,
        requests_filepath: str = './temp_async_llm_requests.jsonl',
        results_filepath: str = './temp_async_llm_results.jsonl'
    ) -> dict:
        """Process all tables in the report asynchronously"""
        queries = []
        table_indices = []
        
        for table in json_report["tables"]:
            table_index = table["table_id"]
            table_indices.append(table_index)
            
            context_before, context_after = self._get_table_context(json_report, table_index)
            table_info = next(table for table in json_report["tables"] if table["table_id"] == table_index)
            table_content = table_info["html"]
            
            # Construct the query
            query = ""
            if context_before:
                query += f'Here is additional text before the table that might be relevant (or not):\n"""{context_before}"""\n\n'
            query += f'Here is a table in HTML format:\n"""{table_content}"""'
            if context_after:
                query += f'\n\nHere is additional text after the table that might be relevant (or not):\n"""{context_after}"""'
            
            queries.append(query)

        results = await AsyncOpenaiProcessor().process_structured_ouputs_requests(
            model='gpt-4o-mini-2024-07-18',
            temperature=0,
            system_content=TableSerialization.system_prompt,
            queries=queries,
            response_format=TableSerialization.TableBlocksCollection,
            preserve_requests=False,
            preserve_results=False,
            logging_level=20,
            requests_filepath=requests_filepath,
            save_filepath=results_filepath,
        )

        # 结果的顺序严格对齐 table_indices（AsyncOpenaiProcessor 内部按 original_index
        # 重排后返回），因此可以放心按 zip 回填，不会错位。
        # Add results back to json_report
        for table_index, result in zip(table_indices, results):
            table_info = next(table for table in json_report["tables"] if table["table_id"] == table_index)

            new_table = {}
            for key, value in table_info.items():
                new_table[key] = value
                if key == "html":
                    new_table["serialized"] = result["answer"]

            for i, table in enumerate(json_report["tables"]):
                if table["table_id"] == table_index:
                    json_report["tables"][i] = new_table

        return json_report

    def process_file(self, json_path: Path) -> None:
        """处理单个文件：读 JSON -> 异步序列化全部表格 -> 回写（线程池的任务单元）。

        每个线程自建事件循环 —— asyncio 事件循环不能跨线程复用，ThreadPoolExecutor
        的 worker 线程不能共享主线程（或无）循环，故在此用 new_event_loop 局部起停。
        临时 jsonl 文件路径带线程 id，多线程并发处理不同文件时互不覆盖。

        Raises:
            JSONDecodeError / 其他异常: 均向上抛，由 process_directory_parallel 汇总裁决。
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_report = json.load(f)

            thread_id = threading.get_ident()
            requests_filepath = f'./temp/async_llm_requests_{thread_id}.jsonl'
            results_filepath = f'./temp/async_llm_results_{thread_id}.jsonl'

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                updated_report = loop.run_until_complete(self.async_serialize_tables(
                    json_report,
                    requests_filepath=requests_filepath,
                    results_filepath=results_filepath
                ))
            finally:
                loop.close()
                try:
                    os.remove(requests_filepath)
                    os.remove(results_filepath)
                except FileNotFoundError:
                    pass
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(updated_report, f, indent=2, ensure_ascii=False)
                
        except json.JSONDecodeError as e:
            self.logger.error("JSON Error in %s: %s", json_path.name, str(e))
            raise
        except Exception as e:
            self.logger.error("Error processing %s: %s", json_path.name, str(e))
            raise

    def process_directory_parallel(self, input_dir: Path, max_workers: int = 5):
        """线程池并行处理整个目录（Pipeline.serialize_tables 的实际入口）。

        Args:
            input_dir: 01_parsed_reports 目录（就地改写其中全部 *.json）
            max_workers: 并发文件数 —— 建议按 API 限流与单文件表数权衡，
                表多的文件内部还有文件级并发请求在跑

        Notes:
            主线程不阻塞在 future 上，而是轮询已完成任务 + 顺带冲刷日志队列
            （process_messages），保证 tqdm 进度条不被 worker 日志打断。
        """
        self.logger.info("Starting parallel table serialization...")
        
        json_files = list(input_dir.glob("*.json"))
        
        if not json_files:
            self.logger.warning("No JSON files found in %s", input_dir)
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(
                total=len(json_files),
                desc="Processing files",
                mininterval=1.0,
                maxinterval=5.0,
                smoothing=0.3
            ) as pbar:
                futures = []
                for json_file in json_files:
                    future = executor.submit(self.process_file, json_file)
                    future.add_done_callback(lambda p: pbar.update(1))
                    futures.append(future)
                
                while futures:
                    process_messages()
                    
                    done_futures = []
                    for future in futures:
                        if future.done():
                            done_futures.append(future)
                            try:
                                future.result()
                            except Exception as e:
                                self.logger.error(str(e))
                    
                    for future in done_futures:
                        futures.remove(future)
                    
                    time.sleep(0.1)

        process_messages()
        self.logger.info("Table serialization completed!")


class TableSerialization:
    """表格序列化任务的「提示词 + 输出 schema」静态仓库。

    信息块设计约束（语义即任务契约，改动需评估对下游检索的影响）:
        - SerializedInformationBlock 是最终入库的检索单元，因此要求 subject 单一、
          信息完全自含（含表名/脚注/币种/数值呈现方式等「周边」说明）；
        - TableBlocksCollection 额外让模型显式列出核心实体与表头 —— 这两列是消歧用的
          「显式索引」，便于人工抽查模型是否漏块；
        - 文档说 SKIPPING ANY VALUABLE INFORMATION WILL BE HEAVILY PENALIZED，
          用强措辞对抗模型漏行漏列（表是检索密集区，漏一行=丢一个可答点）。
    """

    system_prompt = (
        "You are a table serialization agent.\n"
        "Your task is to create a set of contextually independent blocks of information based on the provided table and surrounding text.\n"
        "These blocks must be totally context-independent because they will be used as separate chunk to populate database."
    )

    class SerializedInformationBlock(BaseModel):
        "A single self-contained information block enriched with comprehensive context"

        subject_core_entity: str = Field(description="A primary focus of what this block is about. Usually located in a row header. If one row in the table doesn't make sense without neighboring rows, you can merge information from neighboring rows into one block")
        information_block: str = Field(description=(
    "Detailed information about the chosen core subject from tables and additional texts. Information SHOULD include:\n"
    "1. All related header information\n"
    "2. All related units and their descriptions\n"
    "    2.1. If header is Total, always write additional context about what this total represents in this block!\n"
    "3. All additional info for context enrichment to make ensure complete context-independency if it present in whole table. This can include:\n"
    "    - The name of the table\n"
    "    - Additional footnotes\n"
    "    - The currency used\n"
    "    - The way amounts are presented\n"
    "    - Anything else that can make context even slightly richer\n"
    "SKIPPING ANY VALUABLE INFORMATION WILL BE HEAVILY PENALIZED!"
    ))

    class TableBlocksCollection(BaseModel):
        """Collection of serialized table blocks with their core entities and header relationships"""

        subject_core_entities_list: List[str] = Field(
            description="A complete list of core entities. Keep in mind, empty headers are possible - they should also be interpreted and listed (Usually it's a total or something similar). In most cases each row header represents a core entity")
        relevant_headers_list: List[str] = Field(description="A list of ALL headers relevant to the subject. These headers will serve as keys in each information block. In most cases each column header represents a core entity")
        information_blocks: List["TableSerialization.SerializedInformationBlock"] = Field(description="Complete list of fully described context-independent information blocks")
