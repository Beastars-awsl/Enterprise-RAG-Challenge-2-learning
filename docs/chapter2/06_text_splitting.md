# 第 6 章 文本切分：chunk 与父页双层结构

> **本章目标**：理解本项目最关键的设计之一——**子块（chunk）用于精准向量匹配，
> 父页（page）用于提供完整上下文**。这是理解"父文档检索"（第 9 章）的前提。
> 分治策略：把"既精准又完整"这个矛盾需求拆给两个粒度层，各司其职。

## 6.1 粒度的矛盾与化解

切小块：向量匹配精准（短文本 embedding 语义集中），但单块缺上下文，LLM 看不懂。
切大块：语义完整，但向量匹配精度下降。

父文档检索（Parent Document Retrieval, PDR）的解法：**索引小块，返回大块**。

```text
pages（父层，完整页文本）     ──▶ 拼进 prompt 的上下文
chunks（子层，300 token 小块） ──▶ 建向量索引，被检索命中
```

## 6.2 实现导读：TextSplitter

`src/text_splitter.py`：

- `_split_page`（第 115 行）：`RecursiveCharacterTextSplitter.from_tiktoken_encoder`，
  **300 token 目标块、50 token 重叠**。注意按 tiktoken 计数而非字符数——
  token 才是 embedding 模型的真实口径。
- `_split_report`：逐页切块 + 统一分配 `id` 与 `type='content'`；`o200k_base`
  编码统计 `length_tokens`
- **表格独立通道**：`ser_tab` 配置下，序列化表格信息块作为独立表块加入
  `chunks`，`type` 区分；序列化文件缺失时仅警告继续（容错降级）
- `split_all_reports`（第 141 行）：02 目录 → `databases/chunked_reports/`，
  逐报告同名写出

产物契约（`ingestion.py` 模块 docstring 点明）：**chunk 数组下标 == FAISS 索引行号**，
因此切分与建库必须共享同一份 chunked JSON、顺序不可变。

> **动手**
> 1. 对一页含表格的页手动跑切分，画出 chunk↔page 的父子映射。
> 2. 验证 chunk 与父页的 token 分布：每页切几个 chunk？重叠 50 token 占比多少？

> **自测（合并问题）**
> 1. 为什么不直接对整页做 embedding？（想想 chunk 命中后反正要返回整页，
>    直接索引整页不是更省事吗？）
> 2. 300 token 这个值是怎么来的？如果把 chunk_size 改成 100 或 1000，
>    检索和生成两端分别会发生什么？
