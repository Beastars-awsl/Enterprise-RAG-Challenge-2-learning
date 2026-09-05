# 第 12 章 full_context 大上下文模式

> **本章目标**：理解"不做检索"的对照组——整本报告直接塞给 Gemini。
> README 的原话："It is not RAG, actually"。
> 分治策略：对照实验思想——理解检索组件的价值，最有效的方法是拿掉它看损失了什么。

## 12.1 底层支撑：retrieve_all

`src/retrieval.py:303` 的 `retrieve_all`：返回整本报告全部页面（按页码升序），
`distance` 固定 0.5 只是占位。**页面按页码排序而非相似度排序**，
消费方不应对 distance 做任何语义假设。

## 12.2 full_context 在哪里生效

`src/questions_processing.py:225` 的 `get_answer_for_company`：

```python
if self.full_context:
    retrieval_results = retriever.retrieve_all(company_name)
else:
    retrieval_results = retriever.retrieve_by_company_name(...)
```

`RunConfig.full_context=True` 的 `gemini_thinking` 配置正是 README 说的：
"Full context answering with using enormous context window of Gemini. It is not RAG, actually"。
`_format_retrieval_results` 对 full_context 与检索结果一视同仁地拼接——
**生成端无感知**，切换只动检索端，这就是分治边界清晰的体现。

生成端无感知的代价：上下文格式化对"distance 无语义"的两种来源（full_context
占位 0.5、检索真分数）不做区分，可读性优化只能作用于两边共同的结构。

## 12.3 RAG vs 长上下文的权衡

| 维度 | RAG（检索式） | 长上下文（full_context） |
|---|---|---|
| 漏检风险 | 有（召回窗口、重排误差） | 无 |
| 噪声 | 有（无关页挤占上下文） | **大**（百页无关内容稀释注意力） |
| token 成本 | 低 | 极高（每题整本报告） |
| 时延 | 低 | 高 |
| 引用页校验 | 依赖模型自报 | 依赖模型自报（更不可靠） |

结论：两者是光谱两端，`gemini_thinking` 作为对照组的价值在于
**给 RAG 管线一个性能天花板参照**——若 RAG 不显著优于 full_context，
说明检索/重排环节没做好。

> **动手**
> 1. 用 test_set 跑一次 `gemini_thinking`（注意成本！只跑几道题即可），
>    对比 RAG 模式的正确率与开销。
> 2. 读 `_format_retrieval_results`，确认它如何把页列表拼成 {context}。

> **自测（合并问题）**
> 1. 什么样的题目长上下文必赢、RAG 必输？反之呢？
> 2. 如果预算只允许保留一种模式上场比赛，你怎么用第 21 章的误差分析决定？
