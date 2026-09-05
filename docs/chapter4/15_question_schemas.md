# 第 15 章 五种题型契约

> **本章目标**：逐个理解五种 `AnswerSchema` 的差异，重点体会 boolean 题
> "政策未变 vs 金额上调"的判定哲学、number 题"相似值陷阱"、以及全族 N/A 偏向。
> 分治策略：把"回答年报问题"按**答案的形状**分成五个子问题，
> 每个形状有自己的判定细则与陷阱。

## 15.1 公共字段

五族模板的 `AnswerSchema` 都含四个公共字段：`step_by_step_analysis`
（≥5 步、≥150 词 CoT）、`reasoning_summary`、`relevant_pages`、`final_answer`。
题型差异全部体现在 `final_answer` 的类型与描述上。

## 15.2 各题型要点

| 题型 | final_answer 契约 | 关键防坑条款 |
|---|---|---|
| name | 单个专名，逐字提取 | 从**题目**回抄公司名，不从上下文提取 |
| names | 专名列表或 N/A | N/A 偏向：上下文无直接信息就 N/A，宁可 N/A 不可编造 |
| number | 数值 | 「看似有答案实则是相似值」条款：问 2023 值、上下文给的是 2022 值——相似值≠答案 |
| boolean | True/False | 「did something happen」条款：问某事是否发生，上下文只有相关描述而无该事件 → False |
| comparative | 公司名或 N/A | 含排除规则（币种不符剔除公司等），语义全在 prompt 侧 |

## 15.3 boolean 题的判定哲学（源码内置 example 精读）

`prompts.py` 的 `AnswerWithRAGContextBooleanPrompt.example`（第 382 行）演示了
bool 题最容易翻车的区分：**股息金额按既定政策逐年上调 ≠ 股息"政策"发生变化**——
判定应锚定题眼（X 是否真的发生/成立），而不是"上下文里有没有 X 相关字眼"。
docstring 明言：**example 与字段描述互为校准，改动其中一处必须同步另一处**。

## 15.4 number 题的相似值陷阱

公共 instruction 的措辞警惕条款 + number 模板条款共同防"相似值"：
上下文里出现了一个数字，但它答非所问（错年份、错指标、错公司）。
这也是为什么 CoT 要求"Pay special attention to the wording of the question"。

> **动手**
> 1. 从 questions.json 每种题型各找一道题，人工作答后与系统答案对照；
>    错的题标注"错在检索（relevant_pages 没含正确页）还是错在生成"。
>    这批标注就是第 21 章误差分析的原始素材。

> **自测（合并问题）**
> 1. 五种题型不共用一个 schema 的收益是"防坑条款可以题型定制"，
>    代价是什么？（提示：模板维护成本、example 与描述互为校准的同步负担）
> 2. boolean 判定锚定题眼而非字面匹配——这一原则如何反哺第 14 章的公共 instruction？
