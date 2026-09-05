# 第 4 章 产物合并与 Markdown 导出

> **本章目标**：理解解析产物 → 干净整页文本的演进链：01 解析 JSON → 02 合并
> JSON → 03 整册 Markdown；体会"中间产物多级保存"的工程价值。
> 分治策略：单份报告的处理再拆成 **逐页排版 → 文本清洗 → 表格渲染** 三个子步骤。

## 4.1 三级中间产物

| 级别 | 目录 | 解决什么问题 |
|---|---|---|
| 01 | `debug_data/01_parsed_reports/` | docling 原始结构（页块 + 表 HTML） |
| 02 | `debug_data/02_merged_reports/` | 每页一段干净文本 + 空 chunks 占位 |
| 03 | `debug_data/03_reports_markdown/` | 整册可读 Markdown（人工审查/全上下文） |

**磁盘文件即接口**：02 是 `text_splitter` 的输入契约
（`{"chunks": None, "pages": [...]}`），chunks 占位由第 6 章的切分阶段填充；
03 服务于人工检查与第 12 章 full_context 模式。

## 4.2 单页处理：排版与清洗

`src/parsed_reports_merging.py` 的核心是 `PageTextPreparation`：

- `process_report`（第 89 行）：逐页调 `prepare_page_text`（第 132 行）组装页文本，
  按 `table_id` 渲染表格为 Markdown 表格；产出
  `{"chunks": None, "pages": [{"page", "text"}]}`
- `_clean_text`：修正 docling 的常见输出瑕疵，并打印修正数量与前 30 条明细
  （语料质量巡检）——**清洗规则是可观察的**，不是黑箱

涉及两个开关（`__init__` 第 54 行）：

- `use_serialized_tables`：是否用 LLM 序列化的表格信息块（第 5 章）
- `serialized_tables_instead_of_markdown`：序列化结果**替换**而非追加 Markdown 表格

## 4.3 Markdown 导出：同一逻辑的第二消费方

`export_to_markdown`（`parsed_reports_merging.py:499`）与合并共用
`PageTextPreparation` 的清洗排版，仅多加 `# Page N` 分隔标记。pipeline 的
`export_reports_to_markdown`（`src/pipeline.py:274`）注解：

> 与 merge_reports 共享 PageTextPreparation 的清洗与排版逻辑

这就是分治的好处：**排版逻辑一处定义，两个消费方（02 JSON 与 03 Markdown）共享**，
不会漂移。

> **动手**
> 1. 对比同一页在 01 / 02 / 03 中的三种形态，各找出 1 处"只有当前形态才有"的信息。
> 2. 在 03 Markdown 里找一页含表格的页，检查表格是否可读、数字是否对齐。

> **自测（合并问题）**
> 1. 为什么中间产物要多存几级？如果磁盘受限，哪级最不该删？
> 2. `_clean_text` 的修正明细为什么打印而不静默？和第 3 章 fail-fast 对照，
>    两处的容错哲学有何不同？
