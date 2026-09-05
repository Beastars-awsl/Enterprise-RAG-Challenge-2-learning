# 第 20 章 CLI 与目录契约

> **本章目标**：把 main.py 的 5 个命令、目录契约与产物命名规则串成一张对照表。
> 分治策略：编排层的分治体现在**接口固化**——阶段间以磁盘产物接力，
> 每个阶段可独立重跑，命令顺序不可颠倒。

## 20.1 五个命令

`main.py`（118 行，纯命令解析无业务逻辑）：

| 命令 | 对应 Pipeline 方法 | 产物 |
|---|---|---|
| `download-models` | `download_docling_models` | docling 模型缓存 |
| `parse-pdfs` | `parse_pdf_reports` | 01_parsed_reports |
| `serialize-tables` | `serialize_tables` | 就地改写 01 JSON（追加 serialized 字段） |
| `process-reports` | `process_parsed_reports` | 02/03 中间产物 + databases |
| `process-questions` | `process_questions` | 当前目录 answers*.json |

其中 `parse-pdfs` 支持 `--parallel/--sequential`，
`process-reports --config` 在 ser_tab / no_ser_tab 中二选一，
`process-questions --config` 在 9 个问答配置中二选一。

## 20.2 目录契约与命名规则

`main.py:8` 的 docstring 说明了全部约定：

- 必须在数据目录里执行（如 `cd data/test_set`）
- 阶段间通过磁盘接力，可中断后按序重跑
- 答案文件不覆盖：按 config_suffix 区分，同名追加 `_NN` 编号

`PipelineConfig`（`src/pipeline.py:47`）是"产物放哪"的唯一事实来源，
改目录布局只需动本类。`_ser_tab` 后缀规则让同一份 01 解析结果派生多套
下游产物并排隔离。

> **动手**
> 1. 不运行，仅靠 `--help` 和源码画出命令→方法→产物对照表，与本章表格对照。
> 2. 思考：为什么编排层只做"命令解析 + 委托调用"，不含任何业务逻辑？

> **自测（合并问题）**
> 1. 五个命令的顺序为什么不可任意颠倒？
>    产物链条是 pdf → 01 → 02/03 → databases → answers，每个命令依赖哪些前置？
> 2. "磁盘文件即接口"还有一个好处：换掉中间任何一级的实现，只要产物格式不变，
>    上下游无感知——这如何服务第 19 章的实验矩阵？
