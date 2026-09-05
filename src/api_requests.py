"""
LLM 厂商适配层：统一 chat / 结构化输出 / 失败修复三套能力，向上屏蔽厂商差异。

职责定位:
    - Base*Processor（OpenAI / IBM / Gemini）: 各自的请求发送 + 结构化输出实现 +
      「脏响应修复链」（repair_json 修补 -> 让模型按 schema 重写）；
    - APIProcessor: 按 api_provider 选择实现的门面，并按题目类型(schema:
      name/number/boolean/names/comparative)路由到 prompts 仓库中的对应模板；
    - AsyncOpenaiProcessor: 基于「JSONL 文件队列」的异步批量结构化请求
      （表格序列化等大批量场景），限流由 api_request_parallel_processor 负责。

数据流位置:
    上游调用方: questions_processing（单题问答/比较题拆解）、tables_serialization、
    reranking（绕过本模块直接用 OpenAI 客户端）。
    输出: dict 形式的结构化答案（统一经 response_format 模型校验/清洗），
    以及每次调用的 token 统计 self.response_data（供 *_debug.json 计费审计）。

厂商差异（本层要消灭的三个不一致）:
    1. 结构化输出: OpenAI 原生支持（response_format=… 走 beta.parse）；IBM/Gemini
       无等价能力 -> 改用「system prompt 内嵌 JSON schema + 文本生成 + 客户端修复」；
    2. 请求参数: reasoning 模型(o3-mini)拒绝 temperature；IBM 用 random_seed 字段
       名与 max_new_tokens/min_new_tokens 参数体系；Gemini 需 generation_config；
    3. 失败模式: IBM/Gemini 文本响应带 ```json 围栏/前后缀文本 -> json_repair 打底。

核心依赖与副作用:
    - 网络阻塞 I/O（同步 requests/OpenAI SDK），无重试上限场景可能挂起，调用方超时自理；
    - Gemini 路径包了 tenacity 重试（3 次 x 20s 冷却）; IBM 端点已随比赛下线（README 声明）；
    - self.response_data 是实例级共享属性 —— 多线程并发调用时它记录的是
      「最后一次调用」的统计，逐题统计请从返回值链路取用（已知取舍，勿改）。
"""
import os
import json
from dotenv import load_dotenv
from typing import Union, List, Dict, Type, Optional, Literal
from openai import OpenAI
import asyncio
from src.api_request_parallel_processor import process_api_requests_from_file
from openai.lib._parsing import type_to_response_format_param
import tiktoken
import src.prompts as prompts
import requests
from json_repair import repair_json
from pydantic import BaseModel
import google.generativeai as genai
from copy import deepcopy
from tenacity import retry, stop_after_attempt, wait_fixed



class BaseOpenaiProcessor:
    """OpenAI 实现：原生结构化输出 + seed 复现 + 无本地修复链（API 层已保证 schema）。

    Notes:
        - timeout=None 交由 SDK 默认策略，max_retries=2 兜网络瞬断；
        - seed 参数保持输出可复现（竞赛实验对照需要），但注意 temperature>0 时
          seed 只是尽力而为，不承诺确定性；
        - count_tokens 用 o200k_base（gpt-4o 系列词表），仅作近似统计。
    """

    def __init__(self):
        self.llm = self.set_up_llm()
        self.default_model = 'gpt-4o-2024-08-06'
        # self.default_model = 'gpt-4o-mini-2024-07-18',

    def set_up_llm(self):
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
            )
        return llm

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None, # For deterministic ouptputs
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None
        ):
        """发送一次 chat/结构化请求（同步阻塞），并把 token 统计写入 self.response_data。

        Args:
            model: 显式模型名；None 时用 self.default_model
            temperature: 采样温度（OpenAI 体系）
            seed: 复现种子；None 表示随机 —— 固定 seed 是实验可复现的关键约定
            system_content / human_content: 即 system/user 消息正文
            is_structured: True 时走 beta.chat.completions.parse，按
                response_format（pydantic 类）返回其 model_dump()；False 返回纯文本
            response_format: 结构化输出所需的 pydantic schema（is_structured=True 必填）

        Returns:
            str（非结构化）或 dict（结构化）

        Notes:
            - 结构化路径若抛异常会直接上抛（OpenAI 端 schema 校验失败通常意味着
              prompt/schema 定义错误，属调用方 bug，不应静默降级）；
            - 本方法不是线程安全的统计源（见模块头 response_data 约定）。
        """
        if model is None:
            model = self.default_model
        params = {
            "model": model,
            "seed": seed,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": human_content}
            ]
        }

        # Reasoning models do not support temperature
        # o3-mini 等推理模型只吃 reasoning_effort，传 temperature 会被 API 直接拒绝；
        # 因此对所有含 o3-mini 的模型名跳过该参数（推理模型本身对温度不敏感）
        if "o3-mini" not in model:
            params["temperature"] = temperature
            
        if not is_structured:
            completion = self.llm.chat.completions.create(**params)
            content = completion.choices[0].message.content

        elif is_structured:
            params["response_format"] = response_format
            completion = self.llm.beta.chat.completions.parse(**params)

            response = completion.choices[0].message.parsed
            content = response.dict()

        self.response_data = {"model": completion.model, "input_tokens": completion.usage.prompt_tokens, "output_tokens": completion.usage.completion_tokens}
        print(self.response_data)

        return content

    @staticmethod
    def count_tokens(string, encoding_name="o200k_base"):
        """按 o200k_base 词表近似统计 token 数（成本核算/预算检查用，非精确计费口径）。"""
        encoding = tiktoken.get_encoding(encoding_name)

        # Encode the string and count the tokens
        tokens = encoding.encode(string)
        token_count = len(tokens)

        return token_count


class BaseIBMAPIProcessor:
    """IBM WatsonX 网关实现（经比赛代理端点 rag.timetoact.at/ibm 访问）。

    注意: 代理端点仅在赛期内可用 —— 本类代码保留作历史对照与离线评估，
    ibm_* 系列 RunConfig 现无法跑通（README 已声明）。

    结构化输出降级策略: 无原生 JSON mode，文本生成后经
        json_repair 修补 -> pydantic 校验 -> 失败则让模型按
        AnswerSchemaFixPrompt 重写响应（_reparse_response，至多重试一层）。
    """

    def __init__(self):
        load_dotenv()
        self.api_token = os.getenv("IBM_API_KEY")
        self.base_url = "https://rag.timetoact.at/ibm"
        self.default_model = 'meta-llama/llama-3-3-70b-instruct'
    def check_balance(self):
        """Check the current balance for the provided token."""
        balance_url = f"{self.base_url}/balance"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        try:
            response = requests.get(balance_url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error checking balance: {err}")
            return None
    
    def get_available_models(self):
        """Get a list of available foundation models."""
        models_url = f"{self.base_url}/foundation_model_specs"
        
        try:
            response = requests.get(models_url)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting available models: {err}")
            return None
    
    def get_embeddings(self, texts, model_id="ibm/granite-embedding-278m-multilingual"):
        """Get vector embeddings for the provided text inputs."""
        embeddings_url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": texts,
            "model_id": model_id
        }
        
        try:
            response = requests.post(embeddings_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting embeddings: {err}")
            return None
    
    def send_message(
        self,
        # model='meta-llama/llama-3-1-8b-instruct',
        model=None,
        temperature=0.5,
        seed=None,  # For deterministic outputs
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        max_new_tokens=5000,
        min_new_tokens=1,
        **kwargs
    ):
        """IBM 文本生成调用（同步），含结构化响应修复链与 HTTP 错误降级。

        结构化响应链路（本方法内部）:
            首次解析 -> 校验失败 -> _reparse_response（AskSchemaFixPrompt 重写）
            -> 重写结果再 repair+校验；重写仍不合 schema 时返回「未校验 dict」——
            即调用方可能拿到非契约数据，需靠下游容错（这是没有原生 JSON mode
            的妥协，见类 docstring）。

        Args:
            model: 网关模型名（IBM 全名格式 meta-llama/...），None 用默认
            seed: 经 random_seed 参数传给网关（IBM 命名习惯，非 OpenAI 语义）
            max_new_tokens/min_new_tokens: 生成长度约束（网关特有参数体系）
            is_structured/response_format: 语义同 BaseOpenaiProcessor，但
                response_format 必须是 pydantic 类（用于本地 model_validate）

        Returns:
            结构化时返回清洗后 dict；HTTP 错误返回 None（调用方需判空）。
        """
        if model is None:
            model = self.default_model
        text_generation_url = f"{self.base_url}/text_generation"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        # Prepare the input messages
        input_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": human_content}
        ]

        # Prepare parameters with defaults and any additional parameters
        parameters = {
            "temperature": temperature,
            "random_seed": seed,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            **kwargs
        }
        
        payload = {
            "input": input_messages,
            "model_id": model,
            "parameters": parameters
        }
        
        try:
            response = requests.post(text_generation_url, headers=headers, json=payload)
            response.raise_for_status()
            completion = response.json()

            content = completion.get("results")[0].get("generated_text")
            self.response_data = {"model": completion.get("model_id"), "input_tokens": completion.get("results")[0].get("input_token_count"), "output_tokens": completion.get("results")[0].get("generated_token_count")}
            print(self.response_data)
            if is_structured and response_format is not None:
                try:
                    repaired_json = repair_json(content)
                    parsed_dict = json.loads(repaired_json)
                    validated_data = response_format.model_validate(parsed_dict)
                    content = validated_data.model_dump()
                    return content
                
                except Exception as err:
                    print("Error processing structured response, attempting to reparse the response...")
                    reparsed = self._reparse_response(content, system_content)
                    try:
                        repaired_json = repair_json(reparsed)
                        reparsed_dict = json.loads(repaired_json)
                        try:
                            validated_data = response_format.model_validate(reparsed_dict)
                            print("Reparsing successful!")
                            content = validated_data.model_dump()
                            return content
                        
                        except Exception:
                            return reparsed_dict
                        
                    except Exception as reparse_err:
                        print(f"Reparse failed with error: {reparse_err}")
                        print(f"Reparsed response: {reparsed}")
                        return content
            
            return content

        except requests.HTTPError as err:
            print(f"Error generating text: {err}")
            return None

    def _reparse_response(self, response, system_content):
        """把不合 schema 的文本响应交给「JSON 格式化员」模型重写（修复链第二环）。

        注意这里递归风险: send_message(is_structured=False) 不会再次进入修复链，
        递归深度至多 1 —— 若重写结果仍不合法，由上层 try/except 兜底返回原文，
        保证链路不会无限重试烧钱。
        """
        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=system_content,
            response=response
        )

        reparsed_response = self.send_message(
            system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
            human_content=user_prompt,
            is_structured=False
        )

        return reparsed_response

     
class BaseGeminiProcessor:
    """Gemini 实现：system+user 拼为单 prompt，结构化为「本地修复链」模式。

    差异点:
        - Gemini 无 role 分隔的 system message，正文以分隔行拼进 user prompt；
        - thinking 系列模型通过模型名启用（gemini-2.0-flash-thinking-*）；
        - send_message 外层 tenacity 重试（20s x 3）只覆盖 _generate_with_retry，
          修复链的二次调用不参与重试，避免叠加重试成本失控。
    """

    def __init__(self):
        self.llm = self._set_up_llm()
        self.default_model = 'gemini-2.0-flash-001'
        # self.default_model = "gemini-2.0-flash-thinking-exp-01-21",
        
    def _set_up_llm(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        return genai

    def list_available_models(self) -> None:
        """
        Prints available Gemini models that support text generation.
        """
        print("Available models for text generation:")
        for model in self.llm.list_models():
            if "generateContent" in model.supported_generation_methods:
                print(f"- {model.name}")
                print(f"  Input token limit: {model.input_token_limit}")
                print(f"  Output token limit: {model.output_token_limit}")
                print()

    def _log_retry_attempt(retry_state):
        """Print information about the retry attempt"""
        exception = retry_state.outcome.exception()
        print(f"\nAPI Error encountered: {str(exception)}")
        print("Waiting 20 seconds before retry...\n")

    @retry(
        wait=wait_fixed(20),
        stop=stop_after_attempt(3),
        before_sleep=_log_retry_attempt,
    )
    def _generate_with_retry(self, model, human_content, generation_config):
        """generate_content 的限次重试包装（20s 固定退避 x 3 次）。

        重试策略: 等待 20 秒给厂商限流窗口让路（Gemini 429/5xx 恢复通常秒级~十秒级）；
        before_sleep 回调负责打印失败原因与剩余等待，避免黑盒等待。
        """
        try:
            return model.generate_content(
                human_content,
                generation_config=generation_config
            )
        except Exception as e:
            if getattr(e, '_attempt_number', 0) == 3:
                print(f"\nRetry failed. Error: {str(e)}\n")
            raise

    def _parse_structured_response(self, response_text, response_format):
        try:
            repaired_json = repair_json(response_text)
            parsed_dict = json.loads(repaired_json)
            validated_data = response_format.model_validate(parsed_dict)
            return validated_data.model_dump()
        except Exception as err:
            print(f"Error parsing structured response: {err}")
            print("Attempting to reparse the response...")
            reparsed = self._reparse_response(response_text, response_format)
            return reparsed

    def _reparse_response(self, response, response_format):
        """Reparse invalid JSON responses using the model itself."""
        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=prompts.AnswerSchemaFixPrompt.system_prompt,
            response=response
        )
        
        try:
            reparsed_response = self.send_message(
                model="gemini-2.0-flash-001",
                system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
                human_content=user_prompt,
                is_structured=False
            )
            
            try:
                repaired_json = repair_json(reparsed_response)
                reparsed_dict = json.loads(repaired_json)
                try:
                    validated_data = response_format.model_validate(reparsed_dict)
                    print("Reparsing successful!")
                    return validated_data.model_dump()
                except Exception:
                    return reparsed_dict
            except Exception as reparse_err:
                print(f"Reparse failed with error: {reparse_err}")
                print(f"Reparsed response: {reparsed_response}")
                return response
        except Exception as e:
            print(f"Reparse attempt failed: {e}")
            return response

    def send_message(
        self,
        model=None,
        temperature: float = 0.5,
        seed=12345,  # For back compatibility
        system_content: str = "You are a helpful assistant.",
        human_content: str = "Hello!",
        is_structured: bool = False,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> Union[str, Dict, None]:
        if model is None:
            model = self.default_model

        generation_config = {"temperature": temperature}
        
        prompt = f"{system_content}\n\n---\n\n{human_content}"

        model_instance = self.llm.GenerativeModel(
            model_name=model,
            generation_config=generation_config
        )

        try:
            response = self._generate_with_retry(model_instance, prompt, generation_config)

            self.response_data = {
                "model": response.model_version,
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            }
            print(self.response_data)
            
            if is_structured and response_format is not None:
                return self._parse_structured_response(response.text, response_format)
            
            return response.text
        except Exception as e:
            raise Exception(f"API request failed after retries: {str(e)}")


class APIProcessor:
    """厂商无关的问答门面：按 schema 路由 prompt 模板 + 委托 send_message。

    使用方（QuestionsProcessor）只依赖本类三个方法，不感知 provider 细节；
    换厂商 = 换 RunConfig.api_provider，代码零改动。
    """

    def __init__(self, provider: Literal["openai", "ibm", "gemini"] ="openai"):
        self.provider = provider.lower()
        if self.provider == "openai":
            self.processor = BaseOpenaiProcessor()
        elif self.provider == "ibm":
            self.processor = BaseIBMAPIProcessor()
        elif self.provider == "gemini":
            self.processor = BaseGeminiProcessor()

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        human_content="Hello!",
        is_structured=False,
        response_format=None,
        **kwargs
    ):
        """
        Routes the send_message call to the appropriate processor.
        The underlying processor's send_message method is responsible for handling the parameters.
        """
        if model is None:
            model = self.processor.default_model
        return self.processor.send_message(
            model=model,
            temperature=temperature,
            seed=seed,
            system_content=system_content,
            human_content=human_content,
            is_structured=is_structured,
            response_format=response_format,
            **kwargs
        )

    def get_answer_from_rag_context(self, question, rag_context, schema, model):
        """基于 RAG 上下文的单次结构化问答（全链路最热调用点，串接召回与生成）。

        Args:
            question: 题目原文（进入 user prompt）
            rag_context: 检索结果文本 —— 单公司题为字符串（页文本拼接），
                比较题为「公司名 -> 单公司答案 dict」的序列化（见 questions_processing）
            schema: 答案契约类型 —— name/number/boolean/names/comparative，
                决定使用哪套 system prompt + 输出 schema
            model: 生成模型（来自 RunConfig.answering_model）

        Returns:
            结构化答案 dict（含 step_by_step_analysis / reasoning_summary /
            relevant_pages / final_answer，具体字段随 schema 而异）

        Raises:
            ValueError: schema 不在支持集合内（prompt 仓库不同步的编程错误）
        """
        system_prompt, response_format, user_prompt = self._build_rag_context_prompts(schema)

        answer_dict = self.processor.send_message(
            model=model,
            system_content=system_prompt,
            human_content=user_prompt.format(context=rag_context, question=question),
            is_structured=True,
            response_format=response_format
        )
        self.response_data = self.processor.response_data
        return answer_dict


    def _build_rag_context_prompts(self, schema):
        """按 schema 查 prompt 模板表，返回 (system_prompt, response_format, user_prompt)。

        双 system prompt 体系: 无原生结构化输出的厂商（ibm/gemini）使用
        *_with_schema 变体 —— 把 pydantic 类源码文本直接嵌进 system prompt，
        让模型「照着 schema 写 JSON」；OpenAI 走原生 response_format，无需内嵌。

        Raises:
            ValueError: 未知 schema（新增题目类型时必须先在 prompts 仓库注册）。
        """
        use_schema_prompt = True if self.provider == "ibm" or self.provider == "gemini" else False
        
        if schema == "name":
            system_prompt = (prompts.AnswerWithRAGContextNamePrompt.system_prompt_with_schema 
                            if use_schema_prompt else prompts.AnswerWithRAGContextNamePrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamePrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamePrompt.user_prompt
        elif schema == "number":
            system_prompt = (prompts.AnswerWithRAGContextNumberPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextNumberPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNumberPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNumberPrompt.user_prompt
        elif schema == "boolean":
            system_prompt = (prompts.AnswerWithRAGContextBooleanPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextBooleanPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextBooleanPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextBooleanPrompt.user_prompt
        elif schema == "names":
            system_prompt = (prompts.AnswerWithRAGContextNamesPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextNamesPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamesPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamesPrompt.user_prompt
        elif schema == "comparative":
            system_prompt = (prompts.ComparativeAnswerPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.ComparativeAnswerPrompt.system_prompt)
            response_format = prompts.ComparativeAnswerPrompt.AnswerSchema
            user_prompt = prompts.ComparativeAnswerPrompt.user_prompt
        else:
            raise ValueError(f"Unsupported schema: {schema}")
        return system_prompt, response_format, user_prompt

    def get_rephrased_questions(self, original_question: str, companies: List[str]) -> Dict[str, str]:
        """把比较题拆解为每家公司的独立子问题（比较题问答的前置步骤）。

        产物契约: {公司名: 该公司专用子问题} —— 子问题必须与母题同意图同口径
        （prompt 中强约束），否则各公司答案口径不一致，比较阶段将失去可比性。
        """
        answer_dict = self.processor.send_message(
            system_content=prompts.RephrasedQuestionsPrompt.system_prompt,
            human_content=prompts.RephrasedQuestionsPrompt.user_prompt.format(
                question=original_question,
                companies=", ".join([f'"{company}"' for company in companies])
            ),
            is_structured=True,
            response_format=prompts.RephrasedQuestionsPrompt.RephrasedQuestions
        )
        
        # Convert the answer_dict to the desired format
        questions_dict = {item["company_name"]: item["question"] for item in answer_dict["questions"]}
        
        return questions_dict


class AsyncOpenaiProcessor:
    """文件队列式异步批量结构化请求处理器（面向大批量：表格序列化等）。

    设计要点:
        - 请求先整体落盘成 JSONL（requests_filepath），由 api_request_parallel_processor
          的协程调度器按「每分钟请求数/每分钟 token 数」双限流窗口并发消费 ——
          内存占用与文件行数解耦，海量请求也不会 OOM；
        - 结果文件每行 [request, response, metadata]，行顺序不保证；
          靠请求内 metadata.original_index 重排回原始顺序（本类收尾时也修正文件本身）；
        - 逐行容错解析: 坏 JSON / 内容不合 schema 的行记日志并降级为空答案
          （批量任务容忍个别失败，不做全量重跑）。
    """

    def _get_unique_filepath(self, base_filepath):
        """返回不冲突的文件路径（已存在则依次追加 _1/_2/... 后缀）。

        防御: 并发/重复调用时临时文件互相覆盖会静默丢数据，宁可换名也绝不截断旧文件。
        """
        if not os.path.exists(base_filepath):
            return base_filepath
        
        base, ext = os.path.splitext(base_filepath)
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        return f"{base}_{counter}{ext}"

    async def process_structured_ouputs_requests(
        self,
        model="gpt-4o-mini-2024-07-18",
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        queries=None,
        response_format=None,
        requests_filepath='./temp_async_llm_requests.jsonl',
        save_filepath='./temp_async_llm_results.jsonl',
        preserve_requests=False,
        preserve_results=True,
        request_url="https://api.openai.com/v1/chat/completions",
        max_requests_per_minute=3_500,
        max_tokens_per_minute=3_500_000,
        token_encoding_name="o200k_base",
        max_attempts=5,
        logging_level=20,
        progress_callback=None
    ):
        """批量结构化请求主流程: 落盘请求 -> 限流并发消费 -> 按序收集与校验。

        进度回调机制: monitor_progress 协程轮询结果文件行数（0.1s 间隔），
        新完成的行逐条触发 progress_callback —— 让调用方在无共享内存跨进程的
        条件下拿到准实时进度（tqdm 更新即由此驱动）。

        Args:
            queries: 一批 user 内容（每条 = 一次独立 chat 请求）
            response_format: pydantic 类，逐条校验并 model_dump 结构化答案
            requests_filepath/save_filepath: JSONL 中转文件（_get_unique_filepath 防覆盖；
                preserve_requests/preserve_results 控制跑完是否清理）
            max_requests_per_minute/max_tokens_per_minute: 限流目标（建议留 25%~50% 余量）
            max_attempts: 单请求最大重试次数
            progress_callback: 每条请求成功落盘时回调一次（可为 None）

        Returns:
            [{question: 消息列表, answer: 结构化 dict 或 ""}]，按原始输入顺序对齐
            （解析失败的行为降级为空串 answer，不中断整体）。
        """
        # Create requests for jsonl
        jsonl_requests = []
        for idx, query in enumerate(queries):
            request = {
                "model": model,
                "temperature": temperature,
                "seed": seed,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": query},
                ],
                'response_format': type_to_response_format_param(response_format),
                'metadata': {'original_index': idx}
            }
            jsonl_requests.append(request)
            
        # Get unique filepaths if files already exist
        requests_filepath = self._get_unique_filepath(requests_filepath)
        save_filepath = self._get_unique_filepath(save_filepath)

        # Write requests to JSONL file
        with open(requests_filepath, "w") as f:
            for request in jsonl_requests:
                json_string = json.dumps(request)
                f.write(json_string + "\n")

        # Process API requests
        total_requests = len(jsonl_requests)

        async def monitor_progress():
            last_count = 0
            while True:
                try:
                    with open(save_filepath, 'r') as f:
                        current_count = sum(1 for _ in f)
                        if current_count > last_count:
                            if progress_callback:
                                for _ in range(current_count - last_count):
                                    progress_callback()
                            last_count = current_count
                        if current_count >= total_requests:
                            break
                except FileNotFoundError:
                    pass
                await asyncio.sleep(0.1)

        async def process_with_progress():
            await asyncio.gather(
                process_api_requests_from_file(
                    requests_filepath=requests_filepath,
                    save_filepath=save_filepath,
                    request_url=request_url,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    max_requests_per_minute=max_requests_per_minute,
                    max_tokens_per_minute=max_tokens_per_minute,
                    token_encoding_name=token_encoding_name,
                    max_attempts=max_attempts,
                    logging_level=logging_level
                ),
                monitor_progress()
            )

        await process_with_progress()

        with open(save_filepath, "r") as f:
            validated_data_list = []
            results = []
            for line_number, line in enumerate(f, start=1):
                raw_line = line.strip()
                try:
                    result = json.loads(raw_line)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Line {line_number}: Failed to load JSON from line: {raw_line}")
                    continue

                # Check finish_reason in the API response
                finish_reason = result[1]['choices'][0].get('finish_reason', '')
                if finish_reason != "stop":
                    print(f"[WARNING] Line {line_number}: finish_reason is '{finish_reason}' (expected 'stop').")

                # Safely parse answer; if it fails, leave answer empty and report the error.
                try:
                    answer_content = result[1]['choices'][0]['message']['content']
                    answer_parsed = json.loads(answer_content)
                    answer = response_format(**answer_parsed).model_dump()
                except Exception as e:
                    print(f"[ERROR] Line {line_number}: Failed to parse answer JSON. Error: {e}.")
                    answer = ""

                results.append({
                    'index': result[2],
                    'question': result[0]['messages'],
                    'answer': answer
                })
            
            # Sort by original index and build final list
            validated_data_list = [
                {'question': r['question'], 'answer': r['answer']} 
                for r in sorted(results, key=lambda x: x['index']['original_index'])
            ]

        if not preserve_requests:
            os.remove(requests_filepath)

        if not preserve_results:
            os.remove(save_filepath)
        else:  # Fix requests order
            with open(save_filepath, "r") as f:
                results = [json.loads(line) for line in f]
            
            sorted_results = sorted(results, key=lambda x: x[2]['original_index'])
            
            with open(save_filepath, "w") as f:
                for result in sorted_results:
                    json_string = json.dumps(result)
                    f.write(json_string + "\n")
            
        return validated_data_list
