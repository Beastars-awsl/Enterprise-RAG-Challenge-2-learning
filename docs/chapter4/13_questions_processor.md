# 第 13 章 QuestionsProcessor 总控流程

> **本章目标**：掌握问答层"总指挥"的完整职责面，以及它如何用分治把
> "一次问答"组织成 **路由 → 检索 → 生成 → 校验 → 落盘** 的流水线。
> 这是生成篇的骨架章，14~18 章都是它的某个环节的展开。

## 13.1 类的职责面

`src/questions_processing.py` 的 `QuestionsProcessor.__init__`（第 49 行起）：
构造参数即配置面——questions 文件、目录、检索器开关（llm_reranking）、
schema 体系、并发开关、api_provider、answering_model、full_context 等，
全部由 `RunConfig` 映射而来（第 19 章展开）。

主入口：

- `process_all_questions`（第 709 行）：整库入口，处理 `__init__` 载入的全部题目
- `process_questions_list`：批处理 + 进度保存
- `process_question`（第 329 行）：单题路由——**公司数 >1 走比较题流水线（第 17 章），
  否则单公司流程**

## 13.2 单公司流程

`get_answer_for_company`（第 225 行）：

```text
选检索器（llm_reranking 开 → HybridRetriever；否则 VectorRetriever）
  → retrieve_by_company_name / retrieve_all
  → _format_retrieval_results（拼接上下文）
  → get_answer_from_rag_context（第 16 章 API 层）
  → 引用提取与校验（第 18 章）
```

`_format_retrieval_results` 拼接出 prompt 模板里的 `{context}` 占位符——
检索层输出的 `{distance, page, text}` 列表在此定型为文本。

## 13.3 工程细节：进度保存与断点续跑

- `_save_progress`：批量处理时周期落盘，中断后不从头再来
- `_calculate_statistics`：错误条目（error entries）机制——**单题失败不终止整批**，
  记为 error entry 继续跑（与第 3 章解析阶段 fail-fast 对照：
  解析失败会污染所有下游，单题失败只损失一题）
- 输出双文件：提交格式 + debug 格式（含 step_by_step_analysis 与 token 统计）

> **动手**
> 1. 跟踪一道题从 questions.json 到 answers.json 的完整生命周期，
>    列出途经的每个方法。
> 2. 观察 `_save_progress` 的触发时机：中断进程后重跑，确认断点续跑生效。

> **自测（合并问题）**
> 1. 这个类的职责是否过重？如果重写，你会把哪些职责拆出去？
>    （路由/检索/校验/落盘都是候选）
> 2. "解析 fail-fast、单题容错"的不对称容错设计，判定标准是什么？
>    （提示：下游污染面）
