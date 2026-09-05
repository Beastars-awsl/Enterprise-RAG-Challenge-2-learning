"""
问答阶段（在线推理）编排：题目 -> 公司定位 -> 检索取上下文 -> LLM 结构化回答
-> 页码引用校验/回填 -> 统计与落盘（调试格式 + 比赛提交格式）。

职责定位:
    QuestionsProcessor 是 main.py process-questions / pipeline.process_questions
    的执行体（流水线「最后一公里」）。它与召回/生成层解耦：只持有两个数据目录
    （vector_db_dir、documents_dir）与一个 provider 处理器（APIProcessor），
    具体检索与重排逻辑在 retrieval.py，prompt 模板在 prompts.py。

数据流位置:
    输入: 问题清单 JSON（list[dict]；键族由 new_challenge_pipeline 决定 ——
        新管线 text/kind，legacy question/schema）+ chunked_reports 与
        vector_dbs 目录（经 retriever 载入）+ subset.csv（新管线用它把
        校验后的页码补上 pdf_sha1 以构造提交引用）;
    输出: {stem}_debug.json（questions + answer_details + statistics 全现场）
        与（可选）提交格式文件 —— 提交中页码为 0 起始，内部全链路 1 起始。

关键契约:
    - answer_details 是定长数组，槽位 == 题目注入的 _question_index；主条目
      只带结果与 $ref（"#/answer_details/{i}"）指针，明细（分析/摘要/引用页/
      response_data）集中存放 —— 两处必须同下标读写，任何乱序/拷贝都会错位;
    - 线程模型: parallel_requests 线程池问答，槽位写入在 self._lock 内做；
      self.response_data 是无锁的 last-writer-wins 快照（并行下存在竞态，
      详见 get_answer_for_company），只作诊断/计费统计，不能与题目严格对应;
    - 页码语义: 检索、引用校验、references 构造一律用 1 起始物理页号，
      只在 _post_process_submission_answers 统一转 0 起始（评审格式要求）;
    - 错误语义: 单题失败不中断批次 —— 收敛为带 "error" 字段的结果条目，
      出错题目在提交层以 "N/A" 呈现（不向评审暴露内部错误）;
    - 兼容两代题库: 同一套执行逻辑，仅字段命名与引用校验开关不同。

副作用:
    过程信息（统计、幻觉页码告警、错误 traceback）直接 print 到 stdout;
    带 output_path 时每处理一批（桶）就覆盖写盘一次 —— 长任务中断后
    已完成批次可从文件找回，不必等整批结束。
"""
import json
from typing import Union, Dict, List, Optional
import re
from pathlib import Path
from src.retrieval import VectorRetriever, HybridRetriever
from src.api_requests import APIProcessor
from tqdm import tqdm
import pandas as pd
import threading
import concurrent.futures


class QuestionsProcessor:
    """问答执行器：一个实例承载整批题目的串行/线程池并行问答。

    构造即做 I/O（载入问题文件）并创建 provider 处理器；并行问答时多线程
    共享同一实例 —— 线程安全边界仅限 answer_details 槽位写入（self._lock），
    新增任何被并发写的实例状态都需自行加锁（self.response_data 即反例，
    见模块头竞态说明）。self.detail_counter 无任何读写方（历史遗留字段）。

    关键配置矩阵（仅切换字段族与引用链路，检索/回答核心逻辑共用）:
        new_challenge_pipeline=True : 从 subset.csv 抽取公司名（_extract_）
            答案键 question_text/kind/value，页码引用需 _validate_ + 回填 sha1;
        False                     : 引号正则抽公司名（legacy），键 question/
            schema/answer，不做引用校验与回填;
        llm_reranking             : True 走 HybridRetriever（向量粗召 + LLM 精排），
            False 走 VectorRetriever;
        full_context              : 不召回，整本报告全部页直接进上下文
            （retrieve_all）;注意它与 llm_reranking 不应同时开启 ——
            HybridRetriever 未实现 retrieve_all（见 get_answer_for_company）;
        parent_document_retrieval : 以整页为检索粒度（父文档检索）。
    """

    def __init__(
        self,
        vector_db_dir: Union[str, Path] = './vector_dbs',
        documents_dir: Union[str, Path] = './documents',
        questions_file_path: Optional[Union[str, Path]] = None,
        new_challenge_pipeline: bool = False,
        subset_path: Optional[Union[str, Path]] = None,
        parent_document_retrieval: bool = False,
        llm_reranking: bool = False,
        llm_reranking_sample_size: int = 20,
        top_n_retrieval: int = 10,
        parallel_requests: int = 10,
        api_provider: str = "openai",
        answering_model: str = "gpt-4o-2024-08-06",
        full_context: bool = False
    ):
        self.questions = self._load_questions(questions_file_path)
        self.documents_dir = Path(documents_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.subset_path = Path(subset_path) if subset_path else None
        
        self.new_challenge_pipeline = new_challenge_pipeline
        self.return_parent_pages = parent_document_retrieval
        self.llm_reranking = llm_reranking
        self.llm_reranking_sample_size = llm_reranking_sample_size
        self.top_n_retrieval = top_n_retrieval
        self.answering_model = answering_model
        self.parallel_requests = parallel_requests
        self.api_provider = api_provider
        self.openai_processor = APIProcessor(provider=api_provider)
        self.full_context = full_context

        self.answer_details = []
        self.detail_counter = 0
        self._lock = threading.Lock()

    def _load_questions(self, questions_file_path: Optional[Union[str, Path]]) -> List[Dict[str, str]]:
        """读取题目清单文件（list[dict]，条目键族见类 docstring）。

        Notes:
            路径为 None 时返回空表（等价于「没有题可处理」）—— process_all_questions
            空跑不会崩溃;文件缺失或非法 JSON 则原样抛错，不静默降级。
        """
        if questions_file_path is None:
            return []
        with open(questions_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _format_retrieval_results(self, retrieval_results) -> str:
        """把召回结果拼成 RAG 上下文文本：每页带显式页号标记，页间用分隔线隔开。

        输出形如 'Text retrieved from page {n}: \\"\\"\\"{text}\\"\\"\\"'，页间以
        "\\n\\n---\\n\\n" 分隔 —— 与 prompts.build_system_prompt 的分隔符约定一致，
        且页号被显式写进文本：LLM 声称的引用页码（relevant_pages）正是从这里来，
        随后由 _validate_page_references 与召回结果交叉核验。

        Args:
            retrieval_results: [{page, text, ...}]（按相关度降序；page 为 1 起始物理页号）

        Returns:
            空列表返回空串 —— 不产出无内容上下文（调用方已保证非空才走到这里）。
        """
        if not retrieval_results:
            return ""
        
        context_parts = []
        for result in retrieval_results:
            page_number = result['page']
            text = result['text']
            context_parts.append(f'Text retrieved from page {page_number}: \n"""\n{text}\n"""')
            
        return "\n\n---\n\n".join(context_parts)

    def _extract_references(self, pages_list: list, company_name: str) -> list:
        """把校验后的页号清单包装成提交引用结构：[{"pdf_sha1", "page_index"}, ...]。

        每次调用都重读 subset.csv（文件小、调用频度低，未做缓存）并按
        company_name 精确匹配取 sha1 —— 无匹配行时以空串占位：产出的引用在
        提交端无法指向真实报告（正常配置下 subset 与报告目录一致，不会发生）。

        Notes:
            page_index 此刻仍是 1 起始的内部页号；统一转 0 起始发生在
            _post_process_submission_answers（提交格式要求）—— 不要在此处转换，
            否则内部其他消费方拿到的页码会错位。
        """
        # Load companies data
        if self.subset_path is None:
            raise ValueError("subset_path is required for new challenge pipeline when processing references.")
        self.companies_df = pd.read_csv(self.subset_path)

        # Find the company's SHA1 from the subset CSV
        matching_rows = self.companies_df[self.companies_df['company_name'] == company_name]
        if matching_rows.empty:
            company_sha1 = ""
        else:
            company_sha1 = matching_rows.iloc[0]['sha1']

        refs = []
        for page in pages_list:
            refs.append({"pdf_sha1": company_sha1, "page_index": page})
        return refs

    def _validate_page_references(self, claimed_pages: list, retrieval_results: list, min_pages: int = 2, max_pages: int = 8) -> list:
        """清洗 LLM 声称的引用页码：剔除幻觉页号，不足时从召回结果回填，超限截断。

        为什么需要校验: LLM 可能凭空写出上下文里不存在的页（幻觉引用）；
        竞赛评审按提交引用的 (sha1, 页号) 核验答案，幻觉页 = 扣分项。
        三段式防御:
            1) 只保留出现在 retrieval_results 中的页号（对召回结果做成员判断，
               类型必须一致 —— 两侧都是 1 起始 int，任何一侧改了口径会让
               全部声称页被误判为幻觉）;
            2) 有效页不足 min_pages（默认 2）时按召回顺序（相关度降序）补足
               「最相关」的候选页 —— 保证即便 LLM 只给了一个页，也有最少
               可核验引用;
            3) 超过 max_pages（默认 8）截到前 8 条 —— 保持「LLM 声称页在前、
               回填页在后」的相对顺序（回填是 append 追加）。
        每一步修正都 print 告警（幻觉页明细、截断条数），不静默 —— 出现
        高频幻觉说明 prompt 侧页号口径或召回质量有问题，日志用于定位根因。

        Args:
            claimed_pages: 答案 dict 的 relevant_pages（可能缺失 -> None）
            retrieval_results: 本次召回结果（[{page, ...}]，顺序 = 相关度排序）

        Returns:
            清洗后的页号列表（int，1 起始，长度 <= max_pages）
        """
        if claimed_pages is None:
            claimed_pages = []
        
        retrieved_pages = [result['page'] for result in retrieval_results]
        
        validated_pages = [page for page in claimed_pages if page in retrieved_pages]
        
        if len(validated_pages) < len(claimed_pages):
            removed_pages = set(claimed_pages) - set(validated_pages)
            print(f"Warning: Removed {len(removed_pages)} hallucinated page references: {removed_pages}")
        
        if len(validated_pages) < min_pages and retrieval_results:
            existing_pages = set(validated_pages)
            
            for result in retrieval_results:
                page = result['page']
                if page not in existing_pages:
                    validated_pages.append(page)
                    existing_pages.add(page)
                    
                    if len(validated_pages) >= min_pages:
                        break
        
        if len(validated_pages) > max_pages:
            print(f"Trimming references from {len(validated_pages)} to {max_pages} pages")
            validated_pages = validated_pages[:max_pages]
        
        return validated_pages

    def get_answer_for_company(self, company_name: str, question: str, schema: str) -> dict:
        """单公司问答闭环：选检索器 -> 召回/取全页 -> LLM 结构化回答 -> 引用后处理。

        执行顺序与开关语义:
            1. llm_reranking=True 建 HybridRetriever（向量粗召 + LLM 精排），
               False 建 VectorRetriever —— 每次调用都新建实例（重载全部索引），
               高频路径有 I/O 成本;
            2. full_context=True 跳过召回取整本报告全部页（retrieve_all）。
               注意: 该分支需要 VectorRetriever.retrieve_all —— 配置层不应同时
               开启 llm_reranking 与 full_context（HybridRetriever 没有 retrieve_all）;
            3. 召回为空 -> ValueError（宁可报错也不把空上下文喂给模型硬答）;
            4. 回答后把 provider 侧的 token 统计快照拷到 self.response_data
               —— 无锁赋值：并行问答下可能已被其他线程覆盖（见模块头），
               仅作诊断/计费用途;
            5. new_challenge_pipeline 下: relevant_pages 先经 _validate_page_references
               剔除幻觉页，再经 _extract_references 带上 pdf_sha1 组成提交引用。

        Args:
            company_name: subset 公司全名（与 metainfo.company_name 精确匹配）
            question: 问题原文 —— 单公司题即原题；比较题流程中为改写后的子问题
            schema: 答案契约类型（name/number/boolean/names，见 prompts.py；
                比较题子问题恒为 "number"）

        Returns:
            answer dict: step_by_step_analysis / reasoning_summary /
                relevant_pages / final_answer，+（新管线）references

        Raises:
            ValueError: 检索结果为空，或目录中不存在该公司的报告/索引
        """
        if self.llm_reranking:
            retriever = HybridRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )
        else:
            retriever = VectorRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )

        if self.full_context:
            retrieval_results = retriever.retrieve_all(company_name)
        else:           
            retrieval_results = retriever.retrieve_by_company_name(
                company_name=company_name,
                query=question,
                llm_reranking_sample_size=self.llm_reranking_sample_size,
                top_n=self.top_n_retrieval,
                return_parent_pages=self.return_parent_pages
            )
        
        if not retrieval_results:
            raise ValueError("No relevant context found")
        
        rag_context = self._format_retrieval_results(retrieval_results)
        answer_dict = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema=schema,
            model=self.answering_model
        )
        # 无锁快照（诊断用）—— 并行线程池下可能已反映别题的统计，见模块头竞态说明
        self.response_data = self.openai_processor.response_data
        if self.new_challenge_pipeline:
            pages = answer_dict.get("relevant_pages", [])
            validated_pages = self._validate_page_references(pages, retrieval_results)
            answer_dict["relevant_pages"] = validated_pages
            answer_dict["references"] = self._extract_references(validated_pages, company_name)
        return answer_dict

    def _extract_companies_from_subset(self, question_text: str) -> list[str]:
        """从题目文本中抽取 subset 里被问到的公司名（new challenge 管线用）。

        匹配顺序与剔除策略共同保证不重不漏:
            - 公司名按长度降序尝试: 名字常互为子串（如 "Bank" 之于 "Bank of X"），
              先试短名会把长名的一部分当公司吃掉，导致长名失去匹配机会;
            - 边界 (?:\\W|$) 只在「名字后紧跟非词字符或文本结尾」时命中 ——
              挡住 "PETRA" 命中 "PETRAS" 这类名字是前缀词的伪命中;
            - 命中即从待匹配文本中剔除（且大小写不敏感）: 比较题以分隔符罗列
              多家公司，同公司也可能被提两次 —— 剔除保证每家只计一次;
              也避免已命中文本再次被更短的名字命中。
        副作用: companies_df 惰性加载到实例（_extract_references 每次重读 CSV，
        本方法只在缺失时读一次后缓存 —— 两者并存，勿混用约定）。
        """
        if not hasattr(self, 'companies_df'):
            if self.subset_path is None:
                raise ValueError("subset_path must be provided to use subset extraction")
            self.companies_df = pd.read_csv(self.subset_path)
        
        found_companies = []
        company_names = sorted(self.companies_df['company_name'].unique(), key=len, reverse=True)
        
        for company in company_names:
            escaped_company = re.escape(company)
            
            pattern = rf'{escaped_company}(?:\W|$)'
            
            if re.search(pattern, question_text, re.IGNORECASE):
                found_companies.append(company)
                question_text = re.sub(pattern, '', question_text, flags=re.IGNORECASE)
        
        return found_companies

    def process_question(self, question: str, schema: str):
        """单题路由：定位题目涉及的公司 -> 单公司直答 / 多公司比较流程。

        公司名抽取策略随管线版本切换:
            new_challenge_pipeline=True -> 与 subset 公司全名精确匹配（_extract_），
            公司名在正文中自然出现（模板生成题）;
            False                     -> 引号内容正则 "([^"]*)"（legacy 题库以
            引号显式标注公司名）。

        路由分支:
            0 家公司 -> ValueError（题目无法定位报告，宁可报错也不硬答）;
            1 家     -> get_answer_for_company 直接回答;
            >=2 家   -> process_comparative_question（先拆子问题再并行回答，
            最后汇总成比较答案）。

        Returns:
            单公司: answer dict; 多公司: final_answer 为公司名或 "N/A" 的比较 dict
        """
        if self.new_challenge_pipeline:
            extracted_companies = self._extract_companies_from_subset(question)
        else:
            extracted_companies = re.findall(r'"([^"]*)"', question)
        
        if len(extracted_companies) == 0:
            raise ValueError("No company name found in the question.")
        
        if len(extracted_companies) == 1:
            company_name = extracted_companies[0]
            answer_dict = self.get_answer_for_company(company_name=company_name, question=question, schema=schema)
            return answer_dict
        else:
            return self.process_comparative_question(question, extracted_companies, schema)
    
    def _create_answer_detail_ref(self, answer_dict: dict, question_index: int) -> str:
        """把该题的推理明细写入预分配的 answer_details 槽位，返回 $ref 指针。

        $ref = "#/answer_details/{i}": 主条目（questions 数组里该题的结果 dict）
        只带轻量字段 + 这个指针；step_by_step_analysis / reasoning_summary /
        relevant_pages / response_data 等长文本集中存到顶层 answer_details 数组
        —— 提交文件通过 $ref 消费，debug 文件则两者同存（_save_progress）。
        槽位必须已由 process_questions_list 预分配（[None] * total），否则越界。

        线程安全: 整槽替换写在 self._lock 临界区内 —— 单条赋值在 CPython 下
        本身原子，锁的意义在于把「槽位定位 + 赋值」与错误路径的同类写入圈成
        互斥区，并显式声明该共享可变状态的访问纪律; 注意槽内 response_data
        来自无锁共享属性（见 get_answer_for_company），此处只保证写入原子，
        不保证内容与该题严格对应。
        """
        ref_id = f"#/answer_details/{question_index}"
        with self._lock:
            self.answer_details[question_index] = {
                "step_by_step_analysis": answer_dict['step_by_step_analysis'],
                "reasoning_summary": answer_dict['reasoning_summary'],
                "relevant_pages": answer_dict['relevant_pages'],
                "response_data": self.response_data,
                "self": ref_id
            }
        return ref_id

    def _calculate_statistics(self, processed_questions: List[dict], print_stats: bool = False) -> dict:
        """汇总批处理统计：总数 / 错误 / N/A / 成功，可选打印（print_stats=True）。

        判定口径与提交侧一致:
            - 错误: 结果 dict 含 "error" 键（LLM 侧错误或执行期异常都收敛为此）;
            - N/A: 无 error 且 value 字段 == "N/A"（键名随管线字段族取
              value 或 answer）—— 出错条目的 value 是 None，不重复计入 N/A;
            - 成功 = 总数 - 错误 - N/A。

        Notes:
            print_stats=True 时若 processed_questions 为空会 ZeroDivisionError
            （直接除 total_questions）—— process_questions_list 恒以 True 调用，
            空题单不应走到该函数（见模块头「路径为 None -> 空表」的关联前提）。
        """
        total_questions = len(processed_questions)
        error_count = sum(1 for q in processed_questions if "error" in q)
        na_count = sum(1 for q in processed_questions if (q.get("value") if "value" in q else q.get("answer")) == "N/A")
        success_count = total_questions - error_count - na_count
        if print_stats:
            print(f"\nFinal Processing Statistics:")
            print(f"Total questions: {total_questions}")
            print(f"Errors: {error_count} ({(error_count/total_questions)*100:.1f}%)")
            print(f"N/A answers: {na_count} ({(na_count/total_questions)*100:.1f}%)")
            print(f"Successfully answered: {success_count} ({(success_count/total_questions)*100:.1f}%)\n")
        
        return {
            "total_questions": total_questions,
            "error_count": error_count,
            "na_count": na_count,
            "success_count": success_count
        }

    def process_questions_list(self, questions_list: List[dict], output_path: str = None, submission_file: bool = False, team_email: str = "", submission_name: str = "", pipeline_details: str = "") -> dict:
        """批量执行题目：串行/线程池两种调度，分桶断点落盘。

        执行模型:
            parallel_threads <= 1 -> 逐题串行（tqdm），每题后落盘一次;
            > 1 -> 按 parallel_threads 分桶，桶内 ThreadPoolExecutor 并行、
            桶间串行，每桶结束后落盘一次 —— 长任务中断时已完成部分已持久化。

        顺序与一致性的依赖链（并行下也不允许错位）:
            每题的 dict 在分桶前被注入临时键 "_question_index"（仅内部消费，
            不会出现在输出条目里）; self.answer_details 同步重置为定长
            [None] * total 预分配数组; 线程内经 _create_answer_detail_ref /
            _handle_processing_error 按该下标在锁内原位写入。
            于是无论各题耗时如何错落，questions[i] 的 $ref 永远指向属于
            该题的明细槽位 —— 桶内顺序由 executor.map 的保序语义保证
            （map 按输入顺序产出结果，不是按完成时间）。

        Args:
            questions_list: 题目列表（条目键族见类 docstring）
            output_path: 非空则逐批写 {stem}_debug.json（必写）与提交文件（可选）
            submission_file: True 时额外把提交格式（0 起始页码）写到 output_path
            team_email / submission_name / pipeline_details: 提交文件元信息

        Returns:
            {"questions": 按输入顺序的结果列表, "answer_details": 明细数组,
             "statistics": 统计 dict} —— 与 _save_progress 写的 debug 现场同构
        """
        total_questions = len(questions_list)
        # 注入批内序号：answer_details 槽位/统计/提交侧全部以此对齐（见模块头契约）
        questions_with_index = [{**q, "_question_index": i} for i, q in enumerate(questions_list)]
        self.answer_details = [None] * total_questions  # 定长预分配：槽位写入而非 append
        processed_questions = []
        parallel_threads = self.parallel_requests

        if parallel_threads <= 1:
            for question_data in tqdm(questions_with_index, desc="Processing questions"):
                processed_question = self._process_single_question(question_data)
                processed_questions.append(processed_question)
                if output_path:
                    self._save_progress(processed_questions, output_path, submission_file=submission_file, team_email=team_email, submission_name=submission_name, pipeline_details=pipeline_details)
        else:
            with tqdm(total=total_questions, desc="Processing questions") as pbar:
                for i in range(0, total_questions, parallel_threads):
                    batch = questions_with_index[i : i + parallel_threads]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                        # executor.map will return results in the same order as the input list.
                        batch_results = list(executor.map(self._process_single_question, batch))
                    processed_questions.extend(batch_results)
                    
                    if output_path:
                        self._save_progress(processed_questions, output_path, submission_file=submission_file, team_email=team_email, submission_name=submission_name, pipeline_details=pipeline_details)
                    pbar.update(len(batch_results))
        
        statistics = self._calculate_statistics(processed_questions, print_stats = True)
        
        return {
            "questions": processed_questions,
            "answer_details": self.answer_details,
            "statistics": statistics
        }

    def _process_single_question(self, question_data: dict) -> dict:
        """单题 worker 入口（executor.map 的任务函数）：路由并包装成统一结果形态。

        本方法不抛异常 —— 任何失败都被 except 收敛为带 "error" 字段的结果条目
        （traceback 由 _handle_processing_error 存入明细槽位），这样单个坏题
        不会拖垮整个 executor 批次。
        字段族切换: new_challenge_pipeline 用 text/kind 读题、产出
        question_text/kind/value/references; legacy 用 question/schema、
        产出 question/schema/answer —— 两族在统计与提交转换处再次归一。

        Notes:
            "error" in answer_dict（LLM/API 层把失败放进返回值）与执行期异常
            是两条不同路径: 前者槽位里 step_by_step_analysis 等置 None
            （保留 response_data），后者槽位只有 traceback —— 提交侧对两者
            都表现为 N/A，区分度保留在 debug 文件里。
        """
        # 缺 _question_index（绕过 process_questions_list 的单题调用）时回退 0 号槽 ——
        # 独立单题场景无并发、槽位不会被别的题覆盖，回退安全
        question_index = question_data.get("_question_index", 0)

        if self.new_challenge_pipeline:
            question_text = question_data.get("text")
            schema = question_data.get("kind")
        else:
            question_text = question_data.get("question")
            schema = question_data.get("schema")
        try:
            answer_dict = self.process_question(question_text, schema)
            
            if "error" in answer_dict:
                detail_ref = self._create_answer_detail_ref({
                    "step_by_step_analysis": None,
                    "reasoning_summary": None,
                    "relevant_pages": None
                }, question_index)
                if self.new_challenge_pipeline:
                    return {
                        "question_text": question_text,
                        "kind": schema,
                        "value": None,
                        "references": [],
                        "error": answer_dict["error"],
                        "answer_details": {"$ref": detail_ref}
                    }
                else:
                    return {
                        "question": question_text,
                        "schema": schema,
                        "answer": None,
                        "error": answer_dict["error"],
                        "answer_details": {"$ref": detail_ref},
                    }
            detail_ref = self._create_answer_detail_ref(answer_dict, question_index)
            if self.new_challenge_pipeline:
                return {
                    "question_text": question_text,
                    "kind": schema,
                    "value": answer_dict.get("final_answer"),
                    "references": answer_dict.get("references", []),
                    "answer_details": {"$ref": detail_ref}
                }
            else:
                return {
                    "question": question_text,
                    "schema": schema,
                    "answer": answer_dict.get("final_answer"),
                    "answer_details": {"$ref": detail_ref},
                }
        except Exception as err:
            return self._handle_processing_error(question_text, schema, err, question_index)

    def _handle_processing_error(self, question_text: str, schema: str, err: Exception, question_index: int) -> dict:
        """异常兜底：完整 traceback 落槽 + stdout 打印，返回带 "error" 的结果条目。

        设计意图: 线程池 worker 内不允许未捕获异常冒泡（executor.map 会把
        异常放大成整个批次失败），因此每题异常就地收敛为错误结果 dict——
        类型与消息进 "error" 字段（进入统计与提交层），完整 traceback 存入
        对应 answer_details 槽位供赛后排查，stdout 同步打印三行概览。

        Notes:
            traceback 依赖「异常仍在线程栈上」: 本方法必须在 except 块内
            同步调用（_process_single_question 正是如此），异步/延迟调用
            会拿到过期的栈。槽位写入与 _create_answer_detail_ref 同一把锁，
            同一 _question_index 两条写入路径互斥 —— 每题只可能走其中一条。
        """
        import traceback
        error_message = str(err)
        tb = traceback.format_exc()
        error_ref = f"#/answer_details/{question_index}"
        error_detail = {
            "error_traceback": tb,
            "self": error_ref
        }
        
        with self._lock:
            self.answer_details[question_index] = error_detail
        
        print(f"Error encountered processing question: {question_text}")
        print(f"Error type: {type(err).__name__}")
        print(f"Error message: {error_message}")
        print(f"Full traceback:\n{tb}\n")
        
        if self.new_challenge_pipeline:
            return {
                "question_text": question_text,
                "kind": schema,
                "value": None,
                "references": [],
                "error": f"{type(err).__name__}: {error_message}",
                "answer_details": {"$ref": error_ref}
            }
        else:
            return {
                "question": question_text,
                "schema": schema,
                "answer": None,
                "error": f"{type(err).__name__}: {error_message}",
                "answer_details": {"$ref": error_ref},
            }

    def _post_process_submission_answers(self, processed_questions: List[dict]) -> List[dict]:
        """把内部结果转成比赛提交格式（answers 列表），含页码/引用/字段族归一。

        四项转换:
            1. 页码 1 起始 -> 0 起始 —— 评审格式要求 0-based；内部全链路用
               1-based 是为了对照 PDF 物理页调试，转换收敛在此一处;
            2. value 为 "N/A" 时清空 references —— 出错条目在此层同样呈现为
               "N/A"（见 value 三元式），N/A 答案不携带任何引用;
            3. 题目文本与类型键名按管线字段族归一成 question_text/kind
               （新管线本身即是，legacy 的 question/schema 在此改名）;
            4. 按 $ref 从 answer_details 槽位取 step_by_step_analysis 回填
               reasoning_process（仅当非空 —— 错误/异常槽位没有该键，
               自然不携带推理过程）。

        $ref 解析带边界防御: 指针格式不符 / 下标越界 / 槽位为 None 都静默跳过
        （一条坏引用不应拖垮整份提交文件）。

        Returns:
            提交格式的 answers 列表（含 question_text/kind/value/references，
            及可选 reasoning_process）
        """
        submission_answers = []
        
        for q in processed_questions:
            question_text = q.get("question_text") or q.get("question")
            kind = q.get("kind") or q.get("schema")
            value = "N/A" if "error" in q else (q.get("value") if "value" in q else q.get("answer"))
            references = q.get("references", [])
            
            answer_details_ref = q.get("answer_details", {}).get("$ref", "")
            step_by_step_analysis = None
            if answer_details_ref and answer_details_ref.startswith("#/answer_details/"):
                try:
                    index = int(answer_details_ref.split("/")[-1])
                    if 0 <= index < len(self.answer_details) and self.answer_details[index]:
                        step_by_step_analysis = self.answer_details[index].get("step_by_step_analysis")
                except (ValueError, IndexError):
                    pass
            
            # Clear references if value is N/A
            if value == "N/A":
                references = []
            else:
                # Convert page indices from one-based to zero-based (competition requires 0-based page indices, but for debugging it is easier to use 1-based)
                references = [
                    {
                        "pdf_sha1": ref["pdf_sha1"],
                        "page_index": ref["page_index"] - 1
                    }
                    for ref in references
                ]
            
            submission_answer = {
                "question_text": question_text,
                "kind": kind,
                "value": value,
                "references": references,
            }
            
            if step_by_step_analysis:
                submission_answer["reasoning_process"] = step_by_step_analysis
            
            submission_answers.append(submission_answer)
        
        return submission_answers

    def _save_progress(self, processed_questions: List[dict], output_path: Optional[str], submission_file: bool = False, team_email: str = "", submission_name: str = "", pipeline_details: str = ""):
        """落盘当前进度：调试 JSON 恒写，提交格式文件按 submission_file 开关写。

        debug 文件 = 输出名加 "_debug" 后缀（{stem}_debug{suffix}）: 内容是
        {questions, answer_details, statistics} 完整现场 —— 含每题推理明细与
        response_data，供复盘/计费审计; 提交文件写 output_path 本体，内容为
        _post_process_submission_answers 的产物（0 起始页码） + 队伍元信息。
        两者区分的原因: 调试现场任意时刻可被下一批覆盖（本方法每批调一次），
        提交文件同样按批覆盖 —— 因此文件里永远只有「已处理完成的题」，
        重启续跑场景下未处理的题不会出现在任何文件中（调用方自行保证
        questions_list 全量重跑即可补全）。

        Notes:
            output_path 为 None 时整体 no-op（process_questions_list 已按
            该条件决定是否调用本方法，此处再守一次作为防御）。
        """
        if output_path:
            statistics = self._calculate_statistics(processed_questions)
            
            # Prepare debug content
            result = {
                "questions": processed_questions,
                "answer_details": self.answer_details,
                "statistics": statistics
            }
            output_file = Path(output_path)
            debug_file = output_file.with_name(output_file.stem + "_debug" + output_file.suffix)
            with open(debug_file, 'w', encoding='utf-8') as file:
                json.dump(result, file, ensure_ascii=False, indent=2)
            
            if submission_file:
                # Post-process answers for submission
                submission_answers = self._post_process_submission_answers(processed_questions)
                submission = {
                    "answers": submission_answers,
                    "team_email": team_email,
                    "submission_name": submission_name,
                    "details": pipeline_details
                }
                with open(output_file, 'w', encoding='utf-8') as file:
                    json.dump(submission, file, ensure_ascii=False, indent=2)

    def process_all_questions(self, output_path: str = 'questions_with_answers.json', team_email: str = "79250515615@yandex.com", submission_name: str = "Ilia_Ris SO CoT + Parent Document Retrieval", submission_file: bool = False, pipeline_details: str = ""):
        """整库入口：处理 __init__ 载入的全部题目（self.questions）并落盘。

        方法参数同时充当「一键提交」默认值 —— 直接调用即得到带固定署名的
        标准提交文件（pipeline.process_questions 也经此路径，显式传参覆盖）。
        委托 process_questions_list 后原样返回其结果 dict（结构见彼处）。

        Args:
            output_path: 提交/调试文件的输出根（默认 questions_with_answers.json
                —— 见 main.py 的 answers_file 命名约定，实际比赛路径由 config 覆盖）
            submission_file: 是否额外写比赛提交格式
            team_email / submission_name / pipeline_details: 提交文件元信息
        """
        result = self.process_questions_list(
            self.questions,
            output_path,
            submission_file=submission_file,
            team_email=team_email,
            submission_name=submission_name,
            pipeline_details=pipeline_details
        )
        return result

    def process_comparative_question(self, question: str, companies: List[str], schema: str) -> dict:
        """比较题（涉及 >=2 家公司）处理：拆子问 -> 并行单答 -> 汇总出比较答案。

        三步流水线:
            1. 拆解: get_rephrased_questions 把比较题改写成每家自洽的子问题
               （{公司: 子题} 映射，prompt 强约束子题与母题同指标同口径 ——
               口径漂移会让比较失去可比性）;
            2. 并行单答: 每家一个 future 调 get_answer_for_company，schema 硬编码
               "number"（比较题衡量的都是数值口径，子题只负责出数 —— 外层传入
               的 schema 参数在此不参与选模板，见下方行内注释）;各子答的
               references 先汇聚成全量引用列表;
            3. 汇总: 以 schema="comparative" 再走一次问答 —— rag_context 是
               {公司名: 答案 dict} 而非页文本;ComparativeAnswerPrompt 的
               final_answer 合约是「公司名或 N/A」（含币种不符剔除公司等
               排除规则，语义全在 prompt 侧）。

        引用去重: 不同公司可能引用同一 (pdf_sha1, page_index)，按二元组去重、
        保留首次出现 —— 提交引用是集合语义，重复只会放大噪音（dict 保持
        插入序，去重后顺序 == 首见顺序）。

        异常语义: 任一公司失败（子题缺失 ValueError 或执行异常）都会打印后
        re-raise —— 比较题拒绝「缺腿答案」：少一家公司的比较结果是误导性的，
        宁可使整题失败、进入错误条目。
        """
        # Step 1: Rephrase the comparative question
        rephrased_questions = self.openai_processor.get_rephrased_questions(
            original_question=question,
            companies=companies
        )
        
        individual_answers = {}
        aggregated_references = []
        
        # Step 2: Process each individual question in parallel
        def process_company_question(company: str) -> tuple[str, dict]:
            """单公司子题的执行体（每个 future 的任务）: 取改写子题 -> 出数。

            子题缺失时抛 ValueError —— 上游改写漏了某公司，属于流程缺陷，
            不静默跳过（见外层 docstring 的「缺腿答案」语义）。
            """
            sub_question = rephrased_questions.get(company)
            if not sub_question:
                raise ValueError(f"Could not generate sub-question for company: {company}")

            answer_dict = self.get_answer_for_company(
                company_name=company,
                question=sub_question,
                # 硬编码 "number": 比较题的子题衡量的都是同一数值口径（改写 prompt
                # 保证与母题同指标），比较/排除语义留给最后的 comparative 汇总步;
                # 外层 process_question 传入的 schema 到本方法为止，不再下传
                schema="number"
            )
            return company, answer_dict

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_company = {
                executor.submit(process_company_question, company): company 
                for company in companies
            }
            
            for future in concurrent.futures.as_completed(future_to_company):
                try:
                    company, answer_dict = future.result()
                    individual_answers[company] = answer_dict
                    
                    company_references = answer_dict.get("references", [])
                    aggregated_references.extend(company_references)
                except Exception as e:
                    company = future_to_company[future]
                    print(f"Error processing company {company}: {str(e)}")
                    raise
        
        # 按 (pdf_sha1, page_index) 二元组去重 —— dict 键覆盖天然「保首见」、
        # 结果顺序 == 各公司首次出现的顺序（提交端引用是集合语义，重复无意义）
        unique_refs = {}
        for ref in aggregated_references:
            key = (ref.get("pdf_sha1"), ref.get("page_index"))
            unique_refs[key] = ref
        aggregated_references = list(unique_refs.values())
        
        # Step 3: Get the comparative answer using all individual answers
        comparative_answer = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=individual_answers,
            schema="comparative",
            model=self.answering_model
        )
        # 无锁快照（诊断用）—— 并行下的竞态语义同 get_answer_for_company 处
        self.response_data = self.openai_processor.response_data

        # 引用页 = 各公司子答引用的去重并集（比较题自身不给页 —— Comparative
        # AnswerSchema 的 relevant_pages 约定留空，页引用只能来自各公司单答）
        comparative_answer["references"] = aggregated_references
        return comparative_answer
    