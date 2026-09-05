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

<details>
<summary><b>参考答案（先自己画，再展开对照）</b></summary>

以下路径均核实自 `main.py` docstring 与 `src/pipeline.py` 的 `__init__`。
工作目录 = `data/test_set`，`(...)` 表示 `ser_tab` 变体后缀。

```text
RAG-Challenge-2 分治树 · 磁盘文件即阶段间接口
│
├─ 【输入契约】main.py docstring 定义（1.1 定界）
│   ├─ subset.csv / subset.json      sha1 → 公司名
│   ├─ pdf_reports/*.pdf             原始年报（比赛只给 PDF，不给文本）
│   └─ questions.json                五种题型：name/number/boolean/names/comparative
│
├─ 【摄入篇】第 3~5 章 ─────────────────────────────────
│   ├─ 站① 解析
│   │    CLI: parse-pdfs              文件: pdf_parsing.py (775行)
│   │    输入: pdf_reports/*.pdf + subset.csv
│   │    输出: debug_data/01_parsed_reports/*.json
│   │         (docling 原始输出 → 01_parsed_reports_debug/，仅排障)
│   ├─ 站①½ 表格序列化（可选，冠军配置 nst = 不做这步）
│   │    CLI: serialize-tables        文件: tables_serialization.py (433行)
│   │    输入/输出: 就地改写 01_parsed_reports/*.json（追加 serialized 字段）
│   └─ 站② 合并 + 导出
│        CLI: process-reports 前半    文件: parsed_reports_merging.py (521行)
│        输入: 01_parsed_reports/     输出: debug_data/02_merged_reports(...)/
│                                          debug_data/03_reports_markdown(...)/*.md
│                                          （03 供人工审查 + full_context 模式）
│
├─ 【索引篇】第 6~8 章 ─────────────────────────────────
│   └─ 站③ 切分 + 建库
│        CLI: process-reports 后半    文件: text_splitter.py (168行)
│                                            + ingestion.py (185行)
│        输入: 02_merged_reports(...)/
│        输出: databases(...)/chunked_reports/   300 token 切块 + 父页
│              databases(...)/vector_dbs/{sha1}.faiss   FAISS 逐报告建库
│              databases(...)/bm25_dbs/{sha1}.pkl       BM25 备选（独立步骤，链路未启用）
│
├─ 【检索篇】第 9~12 章 ────────────────────────────────
│   └─ 站④ 检索（process-questions 内部，无独立 CLI）
│        文件: retrieval.py (403行)   向量召回 + 父文档上卷 + Hybrid 两段式
│             reranking.py (222行)    LLM 精排 ← 属检索分支、却跑在站④后段
│        输入: questions.json + vector_dbs/ + chunked_reports/
│        输出: 每题 top-k 上下文文本（内存中，不落盘）
│
├─ 【生成篇】第 13~18 章 ───────────────────────────────
│   └─ 站⑤ 生成（process-questions 内部）
│        CLI: process-questions --config <预设>
│        文件: questions_processing.py (825行)  路由/并发/断点总控
│             prompts.py (665行)                五题型模板族 + 输出 schema
│             api_requests.py (798行)           openai/ibm/gemini 适配
│             api_request_parallel_processor.py (448行)  限速并发队列
│        输入: 检索上下文 + questions.json
│        输出: answers{config_suffix}.json          提交文件
│              answers{config_suffix}_debug.json    推理过程 + token 统计
│              （同名已存在自动追加 _NN，绝不覆盖）
│
└─ 【编排层】第 19~20 章 ── 不在数据流上，是每一站的"扳道工"
     文件: pipeline.py (634行)   Pipeline 类 + RunConfig 实验矩阵
           main.py (118行)       click CLI，纯命令解析无业务逻辑
```

三条对齐要点（对应 1.4 交叉验证）：

1. **后缀规则**：`_ser_tab` 标识表格序列化变体的整条产物链（02/03/databases），
   `answers` 文件后缀来自 config 名（如 `answers_max_nst_o3m.json`）——两套命名规则不同，
   定义都在 `Pipeline.__init__`。
2. **站间解耦**：`01_parsed_reports` 是共享输入，`ser_tab`/`no_ser_tab` 两套下游产物可
   并排运行；`process-reports` 一个命令实际覆盖 1.2 中的两个站（②+③），所以 1.4 的
   表格里它是"前半/后半"。
3. **唯一不落盘的站**：站④⑤发生在 `process-questions` 进程内部，检索结果不写磁盘，
   只有最终答案落盘——这也是断点续跑只能按"题"粒度而不能按"站"粒度的原因。

</details>

> **自测（合并问题）**
> 1. 如果让你把系统拆成 4 个可独立开发的模块，边界怎么划？
> 2. 哪两条边界上有共享文件（如 `questions.json`）？共享带来什么约束？
