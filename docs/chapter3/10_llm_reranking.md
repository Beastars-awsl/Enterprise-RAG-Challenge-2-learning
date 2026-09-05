# 第 10 章 LLM 重排

> **本章目标**：理解 `LLMReranker` 的双路径打分与融合分设计——LLM 相关性分
> 如何与向量分融合，以及为什么需要两条路径。
> 分治策略：重排器把"精排"拆成 **单块打分（并行、贵）** 与
> **多块成批打分（省调用、可能漏块）** 两条路径，按 `documents_batch_size` 分发。

## 10.1 为什么向量分不够用

向量相似度是"表面语义相关"，但年报问答需要"能回答本题"——一道比较毛利率的题，
上下文里出现 "revenue" 的段落未必含毛利率数字。LLM 打分提供第二信号：
**按题目的真实信息需求判断块的相关性**。

## 10.2 双路径打分

`src/reranking.py` 的 `LLMReranker`：

- **单块路径**（`get_rank_for_single_block`，第 89 行）：每块一次结构化打分请求
  （`RetrievalRankingSingleBlock`：reasoning + relevance_score 0~1），
  `ThreadPoolExecutor` 并行执行——贵但评分精细
- **多块路径**（`get_rank_for_multiple_blocks`，第 113 行）：多个块拼进一个
  prompt，一次返回 `block_rankings` 列表——省调用但**可能漏块**，
  靠位置对齐补 0.0 分并告警（缺多少补多少，绝不让漏块静默消失）
- 输出契约：`prompts.py` 的 `RetrievalRankingSingleBlock` /
  `RetrievalRankingMultipleBlocks`——分数刻度与 `RerankingPrompt` 的 11 档
  细则必须同步维护（文本细则与 schema 同源）

> **注意**：prompt 里作为分隔符的 `/n` 是字面量而非换行符（历史笔误，未修以避免
> 扰动线上行为）——比赛代码的"不改、只加"哲学。

## 10.3 融合分：两个信号如何合成一个排序

```python
combined_score = llm_weight * relevance_score + (1 - llm_weight) * vector_distance
```

默认 `llm_weight=0.7`：LLM 分主导排序，向量分做稳定性补偿。两者可直接加权
的前提是**刻度对齐**——relevance_score 定义在 0~1，向量分是单位化向量的
内积（余弦），也近似落在 0~1。

## 10.4 备用重排器：JinaReranker

`src/reranking.py:35` 的 `JinaReranker`（主链路未引用）：
服务端 `top_n` 截断——与本地"先全量打分再本地截断"语义不同，
接入时需注意候选数不可控的场景。

> **动手**
> 1. 改 `llm_weight`（0.5 / 0.7 / 0.9），观察同一题的排序变化。
> 2. 找一道向量检索排第一但答错的题，看 LLM 重排能否把它修正。

> **自测（合并问题）**
> 1. 为什么 LLM 分与向量分可以直接加权平均？如果 LLM 分定义在 0~10，
>    改动最小的方式是什么？
> 2. 单块/多块路径的失效模式不同（贵 vs 漏块），为什么漏块用"补 0"而不是"重试"？
> 3. 重排用 `gpt-4o-mini`、作答用 o3-mini——两处模型的分工逻辑是什么？
