# 第 2 章 跑通最小闭环

> **本章目标**：用 `data/test_set`（5 份年报）跑通全链路，并产出一张
> "阶段→命令→产物目录"导航卡。
> 分治策略：把"跑通全链路"按磁盘产物的接力顺序拆成 **准备 → 离线 A → 离线 B →
> 在线 → 验证固化** 五段。每段以"上一段产物是否就绪"为唯一前置检查——这正是
> `main.py:8` 的目录契约：**磁盘文件即阶段间接口**。

## 2.1 准备：环境与密钥

三件事：

1. **环境**：项目根目录建 venv（Python 3.12），`pip install -e . -r requirements.txt`
2. **密钥**：`env` 重命名为 `.env`，填入 `OPENAI_API_KEY`（Gemini 可选）
3. **目录契约**：数据目录里必须有 `subset.csv`、`pdf_reports/*.pdf`、`questions.json`

> **动手**：`cd data/test_set` 后运行 `python ../../main.py --help`，确认 5 个子命令可见。

## 2.2 离线段 A：解析（download-models → parse-pdfs)

```bash
python main.py download-models          # 一次性暖机 docling 模型，避免跑批中卡网络
python main.py parse-pdfs --parallel    # 默认 chunk_size=2, max_workers=10
```

产物：`debug_data/01_parsed_reports/*.json`；`01_parsed_reports_debug/` 存 docling
原始输出，仅排障用。解析失败 1 份即整批抛 `RuntimeError`（fail-fast，
见 `pdf_parsing.py:268` 的 `parse_and_export`）——宁可中断也不带着残缺语料
跑下游浪费 API 额度。

> **动手**：跑完后任取一份 JSON，定位 `metainfo`（公司名/sha1）、`content`（页块列表）、
> `tables`（含 `html` 字段）三块结构。

## 2.3 离线段 B：切分建库（跳过 serialize-tables → process-reports）

最小闭环**跳过** `serialize-tables`——它是 `ser_tab` 系列配置的可选增强，要花 LLM 调用。

```bash
python main.py process-reports --config no_ser_tab
```

一条命令完成：**合并(02) → Markdown 导出(03) → 切分 → 建向量库**，产物落在：

| 产物目录 | 内容 |
|---|---|
| `debug_data/02_merged_reports/` | 清洗排版后的整页文本 JSON |
| `debug_data/03_reports_markdown/` | 整册 Markdown（人工审查用） |
| `databases/chunked_reports/` | chunk + 父页双层 JSON（第 6 章） |
| `databases/vector_dbs/{sha1}.faiss` | 每公司一份 FAISS 索引（第 7 章） |

注意 `_ser_tab` 后缀规则：同一份 01 解析结果可派生多套下游产物并排隔离
（`02_merged_reports_ser_tab` 与 `databases_ser_tab`），互不覆盖。

> **动手**：逐目录检查 02/03/chunked_reports/vector_dbs 四级产物；数一数某公司的
> chunk 数与页数，验证每页平均产出不止 1 个 chunk。

## 2.4 在线段：问答（process-questions)

```bash
python main.py process-questions --config base
```

`base` 是最小配置：向量召回、无重排、无父文档检索、`gpt-4o` 作答。
产物：

- `answers_base.json` —— 提交格式
- `answers_base_debug.json` —— 逐题推理过程 + token 统计

同名文件再次运行**不覆盖**，自动追加 `_NN` 编号（`_get_next_available_filename`）。
在 debug 文件里挑一道题，读它的 `step_by_step_analysis` 与 `relevant_pages`。

> **动手**：跑完后在 debug 文件里挑一道题，读它的推理链与引用页，感受 CoT 质量。

## 2.5 验证与固化：闭环闭合

验证闭环 = 拿一道题在原 PDF 里找到 `relevant_pages` 对应页，人工判断答案是否有据。

固化产出——**阶段→命令→产物目录对照表**（后续每章实验的导航卡）：

| 阶段 | 命令 | 产物目录 |
|---|---|---|
| 暖机 | `download-models` | docling 模型缓存 |
| 解析 | `parse-pdfs` | `debug_data/01_parsed_reports/` |
| 表格序列化（可选） | `serialize-tables` | 就地改写 01 目录 JSON |
| 切分建库 | `process-reports` | `databases/` |
| 问答 | `process-questions` | 当前目录 `answers*.json` |

> **自测（合并问题）**
> 1. 五个命令的顺序为什么不可任意颠倒？
> 2. 哪两个命令之间可以插入任意多次失败重试而不污染产物？
> 3. `process-reports` 和 `process-questions` 的 config 是两个不同字典
>    （`preprocess_configs` / `configs`）——为什么必须分开？
