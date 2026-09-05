# 第 3 章 PDF 解析与并行化

> **本章目标**：理解 `PDFParser` 如何用 Docling 把年报 PDF 转成结构化 JSON，
> 以及串行/多进程两条路径的设计取舍。
> 分治策略：解析器自身也分治——**转换（convert）与落盘（export）分离**，
> 惰性转换流 + 逐份消费，让失败处理集中在消费端。

## 3.1 为什么选 Docling

年报是解析难度最高的文档类型之一：多栏排版、跨页表格、页眉页脚、财务数字的
对齐结构。Docling 的优势：

- 输出结构化文档模型（页/表/文本块，保留页码与表格 HTML）
- 支持 GPU 加速（原作者用 4090，README 明言"GPU helps a lot"）
- 表格识别质量高，而**财务表格正是年报问答的数据核心**（第 5 章）

备选方案对比（自学拓展）：PyMuPDF 直抽文本（快但丢结构）、Unstructured、
Marker、Mathpix。选型逻辑：**下游任务（表格问答）决定解析精度预算**。

## 3.2 解析器的分治结构

`src/pdf_parsing.py` 的 `PDFParser` 关键方法：

- `convert_documents`（惰性转换流）与 `process_documents`（逐份消费落盘）分离，
  两者通过 docling 的 `DocumentConverter` 产物衔接
- `parse_and_export`（`src/pdf_parsing.py:268`）：串行入口，也是每个并行子进程
  内部执行的函数
- `parse_and_export_parallel`：多进程分发——**每个 worker 在独立子进程内自建
  `PDFParser`**，因为 docling 的 `DocumentConverter` 不可跨进程 pickle，
  进程间只传配置参数不传对象（`_process_chunk`）

产物：

| 文件 | 用途 |
|---|---|
| `debug_data/01_parsed_reports/*.json` | 结构化 JSON，供后续所有阶段消费 |
| `debug_data/01_parsed_reports_debug/` | docling 原始输出，仅排障用 |

## 3.3 fail-fast 语义

`parse_and_export` 存在失败文档时列出明细并抛 `RuntimeError`，保持整批失败：

> 意图：避免带着残缺语料继续跑下游、白白消耗 API 额度。

对照思考：哪些场景 fail-fast 是错的？（如长跑批中 1/100 的失败值得跳过并记录）
这里选择 fail-fast 是因为语料规模小（比赛 20 份）、单份失败往往意味着系统性问题
（模型/内存/GPU OOM）。

> **动手**
> 1. 解析 test_set 中 1 份年报（把其余 PDF 移走），观察 JSON 的
>    `metainfo` / `content`（页块列表）/ `tables`（html 字段）结构。
> 2. 对比 `01_parsed_reports` 与 `01_parsed_reports_debug` 同名文件，
>    理解"业务格式 vs 原始格式"为什么要分开存。

> **自测（合并问题）**
> 1. 并行路径为什么不能在进程间传递 `DocumentConverter` 对象？除了 pickle
>    限制还有什么原因？
> 2. 如果让你给解析阶段加断点续跑（已解析的 PDF 跳过），改动最小的方式是什么？
