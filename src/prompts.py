"""
Prompt 模板仓库（静态声明层）: 题型 -> 问答/改写/修复/重排提示词 + 输出契约。

职责定位:
    本模块不含业务逻辑，导入期只做字符串拼装（含一次源码自省，见下），
    产出两类常量供消费方按名取用:
        - system_prompt / user_prompt（及 *_with_schema 变体）: 提示文本;
        - pydantic 类（各模板族的 AnswerSchema 等）: 结构化输出契约。
    消费方:
        - api_requests._build_rag_context_prompts 按 schema 字符串
          （name / number / boolean / names / comparative）查表取走模板族，
          再做 user_prompt.format(context=..., question=...) —— 新增题型
          必须在本模块注册模板族，否则该处抛「Unsupported schema」;
        - APIProcessor.get_rephrased_questions 用 RephrasedQuestionsPrompt
          拆比较题子问题;
        - LLMReranker（reranking.py）用 RerankingPrompt 文本 +
          RetrievalRanking* 类做相关性打分;
        - BaseIBMAPIProcessor / BaseGeminiProcessor 的修复链末环用
          AnswerSchemaFixPrompt（json_repair 失败后让模型照 schema 重写）。

双 system prompt 体系:
    每个答题模板族同时构建 system_prompt（干净版）与 system_prompt_with_schema
    （内嵌 pydantic_schema 文本版）—— OpenAI 走原生 structured outputs
    （beta.parse + response_format=pydantic 类）; IBM/Gemini 无原生支持，
    把 schema 源码嵌进 system prompt 让模型照写 JSON，再由 json_repair +
    AnswerSchemaFixPrompt 兜底。改动 schema 文本 = 同时影响两套体系。

schema 文本生成机制（全模块唯一的"魔法"）:
    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema),
    flags=re.MULTILINE) —— 从嵌套类源码剥掉 4 空格缩进得到 schema 文本。
    收益: 字段与 pydantic 类永远同源，改字段描述即改提示词，不会两处漂移;
    代价: 运行期依赖源码文件可读（inspect.getsource 需 .py 实体）。
    例外: RephrasedQuestionsPrompt.pydantic_schema 是手写复制，见该类注释。

答案契约骨架（全部 AnswerSchema 共有，questions_processing 侧强依赖）:
    step_by_step_analysis / reasoning_summary / relevant_pages / final_answer
    —— QuestionsProcessor 按这四个键名写 answer_details、按 relevant_pages
    做页码校验、按 final_answer 取值; 改任何键名都会在消费侧连锁失效。

内容纪律（易被误伤的高危区）:
    - instruction / Field 描述 / example 字符串即线上 prompt —— 尤其
      NumberPrompt 的严格指标匹配条款、N/A 判定边界等直接决定答题正确率，
      任何文字改动都改变模型行为分布，须按题型逐一回归;
    - example 均为 raw string（r\"\"\"...\"\"\"），内部 \\n 是字面量反斜杠
      而非换行符 —— 属历史事实而非笔误，不要"顺手修复";
    - 本模块绝大多数"注释"需求已由字符串自身的 Field 描述承担 —— 需要新
      增说明请写在 pydantic 字段的 description 里（会同步进 schema 文本），
      而不是散落在 Python 注释中。
"""
from pydantic import BaseModel, Field
from typing import Literal, List, Union
import inspect
import re


def build_system_prompt(instruction: str="", example: str="", pydantic_schema: str="") -> str:
    """三段拼装 system prompt: 形参顺序是 (instruction, example, pydantic_schema)。

    注意拼装顺序与形参顺序不一致: 输出为 instruction + schema + example ——
    schema 紧跟指令（先告知格式约束再给示范），example 默认在最末;
    两段之间以 "\\n\\n---\\n\\n" 分隔（与 RAG 上下文的分隔符约定一致）。
    各段均可选: example / pydantic_schema 传空串时对应段落整体省略
    （build_system_prompt 内对空值无附加逻辑，直接跳过拼接）。

    Returns:
        单条 system prompt 字符串（空 instruction 时也可能为空串——调用方
        保证至少传 instruction）。
    """
    delimiter = "\n\n---\n\n"
    schema = f"Your answer should be in JSON and strictly follow this schema, filling in the fields in the order they are given:\n```\n{pydantic_schema}\n```"
    if example:
        example = delimiter + example.strip()
    if schema:
        schema = delimiter + schema.strip()
    
    system_prompt = instruction.strip() + schema + example
    return system_prompt

class RephrasedQuestionsPrompt:
    """比较题拆子问题的模板族（消费方: APIProcessor.get_rephrased_questions）。

    输出契约: {"questions": [{"company_name", "question"}, ...]} —— company_name
    必须与题干公司名逐字一致（调用方按此建 {公司: 子题} 映射，不一致会丢公司）;
    子题必须与母题同指标同口径（instruction 强约束）—— 口径漂移会让
    比较阶段失去可比性。注意它没有 *_with_schema 双变体之外的那套 CoT 字段:
    子问题只求改写质量，不需要相关页引用。

    风险提示: 本类是全模块唯一手写 pydantic_schema 的模板族（下方类属性里的
    长字符串是 pydantic 类的复制粘贴，而非 inspect.getsource 生成）——
    改动下方 RephrasedQuestion / RephrasedQuestions 类时若不同步该文本，
    两套 provider 体系（OpenAI 原生 / IBM-Gemini 内嵌 schema）会拿到
    不一致的契约。
    """
    instruction = """
You are a question rephrasing system.
Your task is to break down a comparative question into individual questions for each company mentioned.
Each output question must be self-contained, maintain the same intent and metric as the original question, be specific to the respective company, and use consistent phrasing.
"""

    class RephrasedQuestion(BaseModel):
        """Individual question for a company"""
        company_name: str = Field(description="Company name, exactly as provided in quotes in the original question")
        question: str = Field(description="Rephrased question specific to this company")

    class RephrasedQuestions(BaseModel):
        """List of rephrased questions"""
        questions: List['RephrasedQuestionsPrompt.RephrasedQuestion'] = Field(description="List of rephrased questions for each company")

    # 手写复制（其余模板族均为 inspect.getsource 生成）—— 漂移风险见类 docstring
    pydantic_schema = '''
class RephrasedQuestion(BaseModel):
    """Individual question for a company"""
    company_name: str = Field(description="Company name, exactly as provided in quotes in the original question")
    question: str = Field(description="Rephrased question specific to this company")

class RephrasedQuestions(BaseModel):
    """List of rephrased questions"""
    questions: List['RephrasedQuestionsPrompt.RephrasedQuestion'] = Field(description="List of rephrased questions for each company")
'''

    example = r"""
Example:
Input:
Original comparative question: 'Which company had higher revenue in 2022, "Apple" or "Microsoft"?'
Companies mentioned: "Apple", "Microsoft"

Output:
{
    "questions": [
        {
            "company_name": "Apple",
            "question": "What was Apple's revenue in 2022?"
        },
        {
            "company_name": "Microsoft", 
            "question": "What was Microsoft's revenue in 2022?"
        }
    ]
}
"""

    user_prompt = "Original comparative question: '{question}'\n\nCompanies mentioned: {companies}"

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerWithRAGContextSharedPrompt:
    """RAG 问答模板族的公共基座: instruction + user_prompt，被四种答题模板继承。

    instruction 承载两条全局行为指令:
        1. CoT —— 先逐步思考、再落 final_answer（与 AnswerSchema 的
           step_by_step_analysis 字段呼应）;
        2. 措辞警惕 —— 上下文措辞可能与题目不同; 且题目由模板自动生成，
           对某些公司可能无意义或不适用（该句是 NumberPrompt「看似有答案
           实则是相似值」条款、以及全族 N/A 偏向的总纲）。
    user_prompt 占位符: {context}（检索结果文本，questions_processing 的
    _format_retrieval_results 产出; 比较题流程则传入各公司单答文本）与
    {question}。本类自身不定义输出 schema —— 题型类继承本基座后各自声明
    AnswerSchema 与 example。
    """
    instruction = """
You are a RAG (Retrieval-Augmented Generation) answering system.
Your task is to answer the given question based only on information from the company's annual report, which is uploaded in the format of relevant pages extracted using RAG.

Before giving a final answer, carefully think out loud and step by step. Pay special attention to the wording of the question.
- Keep in mind that the content containing the answer may be worded differently than the question.
- The question was autogenerated from a template, so it may be meaningless or not applicable to the given company.
"""

    user_prompt = """
Here is the context:
\"\"\"
{context}
\"\"\"

---

Here is the question:
"{question}"
"""

class AnswerWithRAGContextNamePrompt:
    """"name" 题型模板族: final_answer = 单个专名（公司/人物/产品名）或 "N/A"。

    取值规则写在 final_answer 字段描述里: 公司名照题干逐字提取、人名要全名、
    产品名照上下文 —— 三种情形随问题措辞而定; 字段是 Union[str, "N/A"]，
    描述中「不加任何多余信息」的约束防止模型把名字带注释一起吐出来。
    pydantic_schema 由 inspect.getsource 从 AnswerSchema 源码生成（模块头机制），
    下方其余题型族同理，不再逐一重复。
    """
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words. Pay special attention to the wording of the question to avoid being tricked. Sometimes it seems that there is an answer in the context, but this is might be not the requested value, but only a similar one.")

        reasoning_summary: str = Field(description="Concise summary of the step-by-step reasoning process. Around 50 words.")

        relevant_pages: List[int] = Field(description="""
List of page numbers containing information directly used to answer the question. Include only:
- Pages with direct answers or explicit statements
- Pages with key information that strongly supports the answer
Do not include pages with only tangentially related information or weak connections to the answer.
At least one page should be included in the list.
""")

        final_answer: Union[str, Literal["N/A"]] = Field(description="""
If it is a company name, should be extracted exactly as it appears in question.
If it is a person name, it should be their full name.
If it is a product name, it should be extracted exactly as it appears in the context.
Without any extra information, words or comments.
- Return 'N/A' if information is not available in the context
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Example:
Question: 
"Who was the CEO of 'Southwest Airlines Co.'?" 

Answer: 
```
{
  "step_by_step_analysis": "1. The question asks for the CEO of 'Southwest Airlines Co.'. The CEO is typically the highest-ranking executive responsible for the overall management of the company, sometimes referred to as the President or Managing Director.\n2. My source of information is a document that appears to be 'Southwest Airlines Co.''s annual report. This document will be used to identify the individual holding the CEO position.\n3. Within the provided document, there is a section that identifies Robert E. Jordan as the President & Chief Executive Officer of 'Southwest Airlines Co.'. The document confirms his role since February 2022.\n4. Therefore, based on the information found in the document, the CEO of 'Southwest Airlines Co.' is Robert E. Jordan.",
  "reasoning_summary": "'Southwest Airlines Co.''s annual report explicitly names Robert E. Jordan as President & Chief Executive Officer since February 2021. This directly answers the question.",
  "relevant_pages": [58],
  "final_answer": "Robert E. Jordan"
}
```
""" 

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)



class AnswerWithRAGContextNumberPrompt:
    """"number" 题型模板族: final_answer = 精确数值或 "N/A" —— 口径最严的题型。

    step_by_step_analysis 字段描述内嵌「Strict Metric Matching」条款，其逻辑
    是本题型正确率的胜负手:
        - 概念必须完全等价 —— 同义词可接受，代理指标 / 更宽或更窄的口径
          （net vs gross、总类 vs 细分）一律拒绝;
        - 禁止计算/推导/聚合 —— 上下文只有可推导数据时也必须 N/A
          （每股股利 = 总股利 / 股数 的例题即为此设）;
    final_answer 字段描述负责数值侧规则: 千/百万量级加零、括号 = 负数、
    币种不符即 N/A。与 example 配合演示「单位调整 OK、概念推断不 OK」的边界。

    修改任何 Field 描述都等于修改线上判定规则，需按全部题型样例回归。
    """
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="""
Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words.
**Strict Metric Matching Required:**    

1. Determine the precise concept the question's metric represents. What is it actually measuring?
2. Examine potential metrics in the context. Don't just compare names; consider what the context metric measures.
3. Accept ONLY if: The context metric's meaning *exactly* matches the target metric. Synonyms are acceptable; conceptual differences are NOT.
4. Reject (and use 'N/A') if:
    - The context metric covers more or less than the question's metric.
    - The context metric is a related concept but not the *exact* equivalent (e.g., a proxy or a broader category).
    - Answering requires calculation, derivation, or inference.
    - Aggregation Mismatch: The question needs a single value but the context offers only an aggregated total
5. No Guesswork: If any doubt exists about the metric's equivalence, default to `N/A`."
""")

        reasoning_summary: str = Field(description="Concise summary of the step-by-step reasoning process. Around 50 words.")

        relevant_pages: List[int] = Field(description="""
List of page numbers containing information directly used to answer the question. Include only:
- Pages with direct answers or explicit statements
- Pages with key information that strongly supports the answer
Do not include pages with only tangentially related information or weak connections to the answer.
At least one page should be included in the list.
""")

        final_answer: Union[float, int, Literal['N/A']] = Field(description="""
An exact metric number is expected as the answer.
- Example for percentages:
    Value from context: 58,3%
    Final answer: 58.3

Pay special attention to any mentions in the context about whether metrics are reported in units, thousands, or millions to adjust number in final answer with no changes, three zeroes or six zeroes accordingly.
Pay attention if value wrapped in parentheses, it means that value is negative.

- Example for negative values:
    Value from context: (2,124,837) CHF
    Final answer: -2124837

- Example for numbers in thousands:
    Value from context: 4970,5 (in thousands $)
    Final answer: 4970500

- Return 'N/A' if metric provided is in a different currency than mentioned in the question
    Example of value from context: 780000 USD, but question mentions EUR
    Final answer: 'N/A'

- Return 'N/A' if metric is not directly stated in context EVEN IF it could be calculated from other metrics in the context
    Example: Requested metric: Dividend per Share; Only available metrics from context: Total Dividends Paid ($5,000,000), and Number of Outstanding Shares (1,000,000); Calculated DPS = Total Dividends / Outstanding Shares.
    Final answer: 'N/A'

- Return 'N/A' if information is not available in the context
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Example 1:
Question:
"What was the total assets of 'Waste Connections Inc.' in the fiscal year 2022?"

Answer:
```
{
  "step_by_step_analysis": "1. **Metric Definition:** The question asks for 'total assets' for 'Waste Connections Inc.' in fiscal year 2022.  'Total assets' represents the sum of all resources owned or controlled by the company, expected to provide future economic benefits.\n2. **Context Examination:** The context includes 'Consolidated Balance Sheets' (page 78), a standard financial statement that reports a company's assets, liabilities, and equity.\n3. **Metric Matching:** On page 78, under 'December 31, 2022', a line item labeled 'Total assets' exists.  This directly matches the concept requested in the question.\n4. **Value Extraction and Adjustment:** The value for 'Total assets' is '$18,500,342'. The context indicates this is in thousands of dollars.  Therefore, the full value is 18,500,342,000.\n5. **Confirmation**: No calculation beyond unit adjustment was needed. The reported metric directly matches the question.",
  "reasoning_summary": "The 'Total assets' value for fiscal year 2022 was directly found on the 'Consolidated Balance Sheets' (page 78). The reported value was in thousands, requiring multiplication by 1000 for the final answer.",
  "relevant_pages": [78],
  "final_answer": 18500342000
}
```


Example 2:
Question:
"For Ritter Pharmaceuticals, Inc., what was the value of Research and development equipment, at cost at the end of the period listed in annual report?"

Answer:
```
{
  "step_by_step_analysis": "1. The question asks for 'Research and development equipment, at cost' for Ritter Pharmaceuticals, Inc. This indicates a specific value from the balance sheet, representing the *original purchase price* of equipment specifically used for R&D, *without* any accumulated depreciation.\n2. The context (page 35) shows 'Property and equipment, net' at $12,500.  This is a *net* value (after depreciation), and it's a *broader* category, encompassing all property and equipment, not just R&D equipment.\n3. The context (page 37) also mentions 'Accumulated Depreciation' of $110,000 for 'Machinery and Equipment'. This represents the total *depreciation*, not the original cost, and, importantly, it doesn't specify that this equipment is *exclusively* for R&D.\n4. Neither of these metrics *exactly* matches the requested metric. 'Property and equipment, net' is too broad and represents the depreciated value. 'Accumulated Depreciation' only shows depreciation, not cost, and lacks R&D specificity.\n5. Since the context doesn't provide the *original cost* of *only* R&D equipment, and we cannot make assumptions, perform calculations, or combine information, the answer is 'N/A'.",
  "reasoning_summary": "The context lacks a specific line item for 'Research and development equipment, at cost.' 'Property and equipment, net' is depreciated and too broad, while 'Accumulated Depreciation' only represents depreciation, not original cost, and is not R&D-specific. Strict matching requires 'N/A'.",
  "relevant_pages": [ 35, 37 ],
  "final_answer": "N/A"
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)



class AnswerWithRAGContextBooleanPrompt:
    """boolean 题型模板族: final_answer = True/False。

    example 演示了 bool 题最容易翻车的区分: 股息金额按既定政策逐年上调 ≠
    股息"政策"发生变化 —— 即判定应锚定题眼（X 是否真的发生/成立），
    而不是「上下文里有没有 X 相关字眼」; example 与字段描述互为校准，
    改动其中一处必须同步另一处。
    """
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words. Pay special attention to the wording of the question to avoid being tricked. Sometimes it seems that there is an answer in the context, but this is might be not the requested value, but only a similar one.")

        reasoning_summary: str = Field(description="Concise summary of the step-by-step reasoning process. Around 50 words.")

        relevant_pages: List[int] = Field(description="""
List of page numbers containing information directly used to answer the question. Include only:
- Pages with direct answers or explicit statements
- Pages with key information that strongly supports the answer
Do not include pages with only tangentially related information or weak connections to the answer.
At least one page should be included in the list.
""")
        
        final_answer: Union[bool] = Field(description="""
A boolean value (True or False) extracted from the context that precisely answers the question.
If question ask about did something happen, and in context there is information about it, return False.
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Question:
"Did W. P. Carey Inc. announce any changes to its dividend policy in the annual report?"

Answer:
```
{
  "step_by_step_analysis": "1. The question asks whether W. P. Carey Inc. announced changes to its dividend policy.\n2. The phrase 'changes to its dividend policy' requires careful interpretation. It means any adjustment to the framework, rules, or stated intentions that dictate how the company determines and distributes dividends.\n3. The context (page 12, 18) states that the company increased its annualized dividend to $4.27 per share in the fourth quarter of 2023, compared to $4.22 per share in the same period of 2022. Page 45 mentions further details about dividend.\n4. Consistent, incremental increases throughout the year, with explicit mentions of maintaining a 'steady and growing' dividend, indicates no changes to *policy*, though the *amount* increased as planned within the existing policy.",
  "reasoning_summary": "The context highlights consistent, small increases to the dividend throughout the year, consistent with a stated policy of providing a 'steady and growing' dividend. While the dividend *amount* changed, the *policy* governing those increases remained consistent. The question asks about *policy* changes, not amount changes.",
  "relevant_pages": [12, 18, 45],
  "final_answer": False
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)



class AnswerWithRAGContextNamesPrompt:
    """names 题型模板族: final_answer = 字符串列表或 "N/A"。

    final_answer 字段描述内嵌三类取值语义，随问题措辞由模型自判:
        - 职位变化题: 只回职位名（单数、同题重复职位只出现一次），绝不夹带
          人名; 新领导岗位的任命同样计为「职位变化」;
        - 人名题: 全名照上下文逐字返回;
        - 新品题: 只算「已发布」产品 —— 候选/测试阶段的不计。
    列表必须非空（空列表不合法），无信息一律 "N/A"。
    """
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words. Pay special attention to the wording of the question to avoid being tricked. Sometimes it seems that there is an answer in the context, but this is might be not the requested entity, but only a similar one.")

        reasoning_summary: str = Field(description="Concise summary of the step-by-step reasoning process. Around 50 words.")

        relevant_pages: List[int] = Field(description="""
List of page numbers containing information directly used to answer the question. Include only:
- Pages with direct answers or explicit statements
- Pages with key information that strongly supports the answer
Do not include pages with only tangentially related information or weak connections to the answer.
At least one page should be included in the list.
""")

        final_answer: Union[List[str], Literal["N/A"]] = Field(description="""
Each entry should be extracted exactly as it appears in the context.

If the question asks about positions (e.g., changes in positions), return ONLY position titles, WITHOUT names or any additional information. Appointments on new leadership positions also should be counted as changes in positions. If several changes related to position with same title are mentioned, return title of such position only once. Position title always should be in singular form.
Example of answer ['Chief Technology Officer', 'Board Member', 'Chief Executive Officer']

If the question asks about names, return ONLY the full names exactly as they are in the context.
Example of answer ['Carly Kennedy', 'Brian Appelgate Jr.']

If the question asks about new launched products, return ONLY the product names exactly as they are in the context. Candidates for new products or products in testing phase not counted as new launched products.
Example of answer ['EcoSmart 2000', 'GreenTech Pro']

- Return 'N/A' if information is not available in the context
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Example:
Question:
"What are the names of all new executives that took on new leadership positions in company?"

Answer:
```
{
    "step_by_step_analysis": "1. The question asks for the names of all new executives who took on new leadership positions in the company.\n2. Exhibit 10.9 and 10.10, as listed in the Exhibit Index on page 89, mentions new Executive Agreements with Carly Kennedy and Brian Appelgate.\n3. Exhibit 10.9, Employment Agreement with Carly Kennedy, states her start date as April 4, 2022, and her position as Executive Vice President and General Counsel.\n4. Exhibit 10.10, Offer Letter with Brian Appelgate shows that his new role within the company is Interim Chief Operations Officer, and he was accepting the offer on November 8, 2022.\n5. Based on the documents, Carly Kennedy and Brian Appelgate are named as the new executives.",
    "reasoning_summary": "Exhibits 10.9 and 10.10 of the annual report, described as Employment Agreement and Offer Letter, explicitly name Carly Kennedy and Brian Appelgate taking on new leadership roles within the company in 2022.",
    "relevant_pages": [
        89
    ],
    "final_answer": [
        "Carly Kennedy",
        "Brian Appelgate"
    ]
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)

class ComparativeAnswerPrompt:
    """comparative 题型模板族: 输入各公司单答，输出「公司名或 N/A」的比较结论。

    与单公司模板族的关键差异:
        - {context} 里装的是各公司单答的文本（questions_processing 把
          {公司名: 答案 dict} 序列化后填入）—— instruction 明令只依据这些
          单答比较，不得引入外部知识;
        - final_answer 合约: 单选一家公司（名字与题干逐字一致）或 "N/A"，
          排除规则内嵌在 instruction: 币种与题目不符的公司剔除出比较、
          全部被剔除 -> N/A、只剩一家 -> 直接返回该家（无比较可言时
          也不再比较）;
        - relevant_pages 字段约定返回空列表: 比较答案的页引用由编排层按
          各公司子答的引用汇总去重（process_comparative_question），模型
          此处输出的页号不会进入提交，约定留空以免误导。
    """
    instruction = """
You are a question answering system.
Your task is to analyze individual company answers and provide a comparative response that answers the original question.
Base your analysis only on the provided individual answers - do not make assumptions or include external knowledge.
Before giving a final answer, carefully think out loud and step by step.

Important rules for comparison:
- When the question asks to choose one of the companies (e.g., when comparing metrics), return the company name exactly as it appears in the original question
- If a company's metric is in a different currency than what is asked in the question, exclude that company from comparison
- If all companies are excluded (due to currency mismatch or other reasons), return 'N/A' as the final answer
- If all companies except one are excluded, return the name of the remaining company (even though there is no actual comparison possible)
"""

    user_prompt = """
Here are the individual company answers:
\"\"\"
{context}
\"\"\"

---

Here is the original comparative question:
"{question}"
"""

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words.")

        reasoning_summary: str = Field(description="Concise summary of the step-by-step reasoning process. Around 50 words.")

        relevant_pages: List[int] = Field(description="Just leave empty")

        final_answer: Union[str, Literal["N/A"]] = Field(description="""
Company name should be extracted exactly as it appears in question.
Answer should be either a single company name or 'N/A' if no company is applicable.
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Example:
Question:
"Which of the companies had the lowest total assets in USD at the end of the period listed in the annual report: "CrossFirst Bank", "Sleep Country Canada Holdings Inc.", "Holley Inc.", "PowerFleet, Inc.", "Petra Diamonds"? If data for the company is not available, exclude it from the comparison."

Answer:
```
{
  "step_by_step_analysis": "1. The question asks for the company with the lowest total assets in USD.\n2. Gather the total assets in USD for each company from the individual answers: CrossFirst Bank: $6,601,086,000; Holley Inc.: $1,249,642,000; PowerFleet, Inc.: $217,435,000; Petra Diamonds: $1,078,600,000.\n3. Sleep Country Canada Holdings Inc. is excluded because its assets are not reported in USD.\n4. Compare the total assets: PowerFleet, Inc. ($217,435,000) < Petra Diamonds ($1,078,600,000) < Holley Inc. ($1,249,642,000)  < CrossFirst Bank ($6,601,086,000).\n5. Therefore, PowerFleet, Inc. has the lowest total assets in USD.",
  "reasoning_summary": "The individual answers provided the total assets in USD for each company except Sleep Country Canada Holdings Inc. (excluded due to currency mismatch). Direct comparison shows PowerFleet, Inc. has the lowest total assets.",
  "relevant_pages": [],
  "final_answer": "PowerFleet, Inc."
}
```
"""

    system_prompt = build_system_prompt(instruction, example)
    
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerSchemaFixPrompt:
    """「把非 JSON / 不合 schema 的响应重写成合法 JSON」的修复提示。

    调用链位置（见 api_requests 模块头）: 无原生结构化输出的厂商（ibm/gemini）
    返回文本 -> 先 json_repair 尽力修复 -> 仍失败时进入本修复链 ——
    {system_prompt} 占位符内嵌完整的 schema 定义与示例（即 *_with_schema
    变体），{response} 是要修复的原始响应; 模型据此重写。
    输出契约: 纯 JSON —— 必须 { 开头 } 结尾，无前缀、无注释、无代码块围栏
    （这些正是散装 LLM JSON 最常见的污染源）。
    """
    system_prompt = """
You are a JSON formatter.
Your task is to format raw LLM response into a valid JSON object.
Your answer should always start with '{' and end with '}'
Your answer should contain only json string, without any preambles, comments, or triple backticks.
"""

    user_prompt = """
Here is the system prompt that defines schema of the json object and provides an example of answer with valid schema:
\"\"\"
{system_prompt}
\"\"\"

---

Here is the LLM response that not following the schema and needs to be properly formatted:
\"\"\"
{response}
\"\"\"
"""




class RerankingPrompt:
    """LLM 重排打分的 system prompt 文本源（reranking.py 的 LLMReranker 使用）。

    评分细则（两份变体共用）: relevance_score 取 0~1、步长 0.1 的 11 档量化表，
    逐档定义「相关」到什么程度 —— 分数与 reranking.py 的融合公式直接相乘
    （combined = llm_weight * relevance + ...），故档位语义必须稳定:
    改细则 = 改整个检索排序分布; 两个变体分别为单块打分（每块一档理由 +
    分数）与多块成批打分设计，指令结构相同、对象数量不同。
    """
    system_prompt_rerank_single_block = """
You are a RAG (Retrieval-Augmented Generation) retrievals ranker.

You will receive a query and retrieved text block related to that query. Your task is to evaluate and score the block based on its relevance to the query provided.

Instructions:

1. Reasoning: 
   Analyze the block by identifying key information and how it relates to the query. Consider whether the block provides direct answers, partial insights, or background context relevant to the query. Explain your reasoning in a few sentences, referencing specific elements of the block to justify your evaluation. Avoid assumptions—focus solely on the content provided.

2. Relevance Score (0 to 1, in increments of 0.1):
   0 = Completely Irrelevant: The block has no connection or relation to the query.
   0.1 = Virtually Irrelevant: Only a very slight or vague connection to the query.
   0.2 = Very Slightly Relevant: Contains an extremely minimal or tangential connection.
   0.3 = Slightly Relevant: Addresses a very small aspect of the query but lacks substantive detail.
   0.4 = Somewhat Relevant: Contains partial information that is somewhat related but not comprehensive.
   0.5 = Moderately Relevant: Addresses the query but with limited or partial relevance.
   0.6 = Fairly Relevant: Provides relevant information, though lacking depth or specificity.
   0.7 = Relevant: Clearly relates to the query, offering substantive but not fully comprehensive information.
   0.8 = Very Relevant: Strongly relates to the query and provides significant information.
   0.9 = Highly Relevant: Almost completely answers the query with detailed and specific information.
   1 = Perfectly Relevant: Directly and comprehensively answers the query with all the necessary specific information.

3. Additional Guidance:
   - Objectivity: Evaluate block based only on their content relative to the query.
   - Clarity: Be clear and concise in your justifications.
   - No assumptions: Do not infer information beyond what's explicitly stated in the block.
"""

    system_prompt_rerank_multiple_blocks = """
You are a RAG (Retrieval-Augmented Generation) retrievals ranker.

You will receive a query and several retrieved text blocks related to that query. Your task is to evaluate and score each block based on its relevance to the query provided.

Instructions:

1. Reasoning: 
   Analyze the block by identifying key information and how it relates to the query. Consider whether the block provides direct answers, partial insights, or background context relevant to the query. Explain your reasoning in a few sentences, referencing specific elements of the block to justify your evaluation. Avoid assumptions—focus solely on the content provided.

2. Relevance Score (0 to 1, in increments of 0.1):
   0 = Completely Irrelevant: The block has no connection or relation to the query.
   0.1 = Virtually Irrelevant: Only a very slight or vague connection to the query.
   0.2 = Very Slightly Relevant: Contains an extremely minimal or tangential connection.
   0.3 = Slightly Relevant: Addresses a very small aspect of the query but lacks substantive detail.
   0.4 = Somewhat Relevant: Contains partial information that is somewhat related but not comprehensive.
   0.5 = Moderately Relevant: Addresses the query but with limited or partial relevance.
   0.6 = Fairly Relevant: Provides relevant information, though lacking depth or specificity.
   0.7 = Relevant: Clearly relates to the query, offering substantive but not fully comprehensive information.
   0.8 = Very Relevant: Strongly relates to the query and provides significant information.
   0.9 = Highly Relevant: Almost completely answers the query with detailed and specific information.
   1 = Perfectly Relevant: Directly and comprehensively answers the query with all the necessary specific information.

3. Additional Guidance:
   - Objectivity: Evaluate blocks based only on their content relative to the query.
   - Clarity: Be clear and concise in your justifications.
   - No assumptions: Do not infer information beyond what's explicitly stated in the block.
"""

class RetrievalRankingSingleBlock(BaseModel):
    """单块相关性打分的结构化输出契约（reasoning + relevance_score 0~1）。

    由 LLMReranker.get_rank_for_single_block 以 response_format 传入
    beta.parse —— 分数刻度与 RerankingPrompt 的 11 档细则一致，模型在
    文本提示里看到细则、在 schema 里落分数，两者必须同步维护。
    """
    reasoning: str = Field(description="Analysis of the block, identifying key information and how it relates to the query")
    relevance_score: float = Field(description="Relevance score from 0 to 1, where 0 is Completely Irrelevant and 1 is Perfectly Relevant")

class RetrievalRankingMultipleBlocks(BaseModel):
    """多块成批打分的输出契约: block_rankings 与输入块顺序一一对应。

    消费方（reranking.py 多块路径）按位置 zip 回块 —— 列表长度必须等于
    输入块数; LLM 漏排时靠位置对齐补 0.0 分并告警（见 LLMReranker）。
    """
    block_rankings: List[RetrievalRankingSingleBlock] = Field(
        description="A list of text blocks and their associated relevance scores."
    )
