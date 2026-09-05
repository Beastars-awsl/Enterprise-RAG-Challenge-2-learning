# 第 1 章 全景地图：从比赛题目到冠军方案

> **本章目标**：不读一行实现代码，建立整个系统的全景地图。
> 分治策略：把"理解整个系统"拆成 **定界 → 纵切 → 横切 → 交叉验证 → 合并** 五步——
> 先钉死输入输出（问题域），再沿数据流纵切一刀（时间维），再按模块横切一刀（空间维），
> 两个切面互验后合并成你自己的分治树。

## 1.1 定界：系统的输入与输出

只回答三个问题——**进什么、出什么、评分看什么**。

输入（目录契约，见 `main.py` 顶部 docstring）：

| 文件/目录 | 作用 |
|---|---|
| `subset.csv` | 报告元信息：sha1 → 公司名映射 |
| `pdf_reports/*.pdf` | 原始年报 PDF |
| `questions.json` | 待回答问题集 |

问题分五种题型：`name` / `number` / `boolean` / `names` / `comparative`，
其中 `comparative`（比较题）涉及 ≥2 家公司，走独立的 Map-Reduce 流水线（第 17 章）。

输出：`answers{config_suffix}.json`（提交格式），每题含 `final_answer` 与
`relevant_pages` 引用列表。**评分同时看答案正确性与引用可验证性**——这决定了
第 18 章引用处理的严格程度。

> **动手**：打开 `data/test_set/questions.json`，按题型各找出 1 道题，手工分类"单公司 / 比较题"。

## 1.2 纵切：一次问答的数据形态演变

沿时间轴追踪**一份 PDF 的形态变化**，每站只记"变成什么样"，不问"怎么变"：

```text
PDF ──解析──▶ 结构化 JSON（页 + 表 + metainfo）
     ──切分+向量化──▶ FAISS 索引 + chunk/父页双层文本
     ──检索──▶ top-k 上下文文本
     ──LLM──▶ 结构化答案 dict
```

前三站发生在提问之前（**离线建库**），后两站在提问之后（**在线问答**）。
五个站点正好对应 `main.py` 的五个 CLI 命令。

## 1.3 横切：模块地图与文件归属

沿空间轴把 `src/` 的 12 个文件归入四大分支（行数即复杂度信号）：

| 分支 | 文件（行数） | 一句话职责 |
|---|---|---|
| 摄入 | `pdf_parsing.py` (775) | Docling 解析 PDF → JSON，串行/多进程双路径 |
| 摄入 | `parsed_reports_merging.py` (521) | 清洗排版合并解析产物，导出 Markdown |
| 摄入 | `tables_serialization.py` (433) | 用 LLM 把表格改写为自解释信息块 |
| 索引 | `text_splitter.py` (168) | 300 token 切 chunk，同时保留父页 |
| 索引 | `ingestion.py` (185) | embedding → FAISS；BM25 备选 |
| 检索 | `retrieval.py` (403) | 向量召回 + 父文档上卷 + Hybrid 两段式 |
| 检索 | `reranking.py` (222) | LLM 打分与向量分加权融合重排 |
| 生成 | `prompts.py` (665) | 五题型 prompt 模板族 + 输出 schema |
| 生成 | `questions_processing.py` (825) | 问答总控：路由、并发、断点、落盘 |
| 生成 | `api_requests.py` (798) | openai/ibm/gemini 三厂商适配 |
| 生成 | `api_request_parallel_processor.py` (448) | 限速并发请求队列 |
| 编排 | `pipeline.py` (634) | Pipeline 类 + RunConfig 实验矩阵 |
| 编排 | `main.py` (118) | click CLI，纯命令解析无业务逻辑 |

## 1.4 交叉验证：两个切面对齐

把 1.2 的五站与 1.3 的分支对齐：

| 数据流站点 | 对应分支 | 对应命令 |
|---|---|---|
| 站 1 解析 | 摄入 | `parse-pdfs` |
| 站 2 合并/导出 | 摄入 | `serialize-tables`（可选）+ `process-reports` 前半 |
| 站 3 切分建库 | 索引 | `process-reports` 后半 |
| 站 4 检索 | 检索 | `process-questions` 内部 |
| 站 5 生成 | 生成 | `process-questions` 内部 |

编排层（`pipeline.py` / `main.py`）不在线上，而是每站的"扳道工"。
对不齐的地方（如 `reranking.py` 属检索分支但运行于站 4 精排段）就是理解加深处。

## 1.5 冠军配方速览

读 `README.md` 技术清单与 `src/pipeline.py:583` 的 `configs` 字典。
冠军配置 `max_nst_o3m` ≈ **父文档检索 + LLM 重排 + 不做表格序列化（nst）+ o3-mini 作答**。
9 个命名配置（`base → pdr → max → max_nst_o3m → …`）就是这套配方的消融矩阵，
IBM 系列已因比赛 API 关闭而失效。配置细节在第 19 章展开。

## 1.6 合并：产出你的分治树

综合 1.1~1.5，画出带标注的全景图：树的每个节点标注 **对应文件 + 对应 CLI 阶段 +
输入/输出产物路径**。此图是后续所有章节的导航地图。

> **动手**：手绘或用工具画出分治树，与上方"分治地图"对照，差异处写明谁对、为什么。

> **自测（合并问题）**
> 1. 如果让你把系统拆成 4 个可独立开发的模块，边界怎么划？
> 2. 哪两条边界上有共享文件（如 `questions.json`）？共享带来什么约束？
