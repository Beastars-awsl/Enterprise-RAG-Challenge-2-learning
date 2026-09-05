# 第 5 章 表格序列化

> **本章目标**：理解为什么"表格"是年报 RAG 的胜负手，以及 LLM 序列化如何
> 把 HTML 表格变成自解释的信息块。
> 分治策略：把"让 LLM 看懂表格"拆成 **上下文化 → 逐表改写 → 回写原地** 三步。

## 5.1 为什么表格是胜负手

年报问答里大量题目是数字题（number 题型），答案几乎总在财务表格里。但表格
直接线性化为文本后：

- 表头与数据行的关联可能被切断（"2023 vs 2022"列对不上）
- 合并单元格、多级表头丢失语义

这就是 `ser_tab` 系列配置存在的原因：与其让检索和生成猜表格语义，
不如**建库前就请 LLM 把表格改写成自解释的信息块**。

## 5.2 实现导读：TableSerializer

`src/tables_serialization.py` 的流程（`_serialize_table`，第 191 行）：

1. **取上下文**：`_get_table_context` 抓取表格前后的正文（表格通常不能脱离
   上下文自解释——"Net revenues"指哪家公司哪个口径？）
2. **组装请求**：system prompt（`prompts.TableSerialization`）+ 表格 HTML +
   前后文，`gpt-4o-mini`、temperature=0、结构化输出
   （`TableSerialization.TableBlocksCollection` schema）
3. **回写原地**：为每张表追加 `serialized` 字段（信息块列表），
   **就地改写 01 目录的 JSON**

并发模型：`process_directory_parallel` 文件级并行 + 线程内独立事件循环；
临时文件写入 `./temp/`（线程私有命名，处理后删除）。

## 5.3 下游衔接与降级链

序列化结果有两个消费点，且都有降级路径：

```text
01 JSON（tables[].serialized 字段）
  ├─▶ PageTextPreparation（第 4 章）：无 serialized 字段 → 静默回退纯 Markdown 表格
  └─▶ text_splitter（第 6 章）：序列化表格走独立表块通道，缺失时仅警告、正文块照常产出
```

CLI 层：`serialize-tables --max-workers 10`，固定 `gpt-4o-mini`、温度 0。
注意它是**可选阶段**——最小闭环（第 2 章）跳过它。

> **动手**
> 1. 取一张年报里的真实财务表格，分别用 Markdown 线性化和"信息块列表"两种
>    形式表达，自测哪种更容易让 LLM 答对一道数字题。
> 2. （可选）跑 `serialize-tables` 后 diff 一份 01 JSON，观察 `serialized`
>    字段长什么样。

> **自测（合并问题）**
> 1. 序列化发生在"建库前"而不是"检索后"，这个时机选择有什么利弊？
>    （提示：离线成本 vs 在线成本、索引一致性）
> 2. 表格序列化和第 6 章的 chunk 切分是什么关系？表格信息块为什么需要
>    独立的切分通道？
