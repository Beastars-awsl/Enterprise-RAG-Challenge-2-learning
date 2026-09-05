# 第 19 章 RunConfig 与实验矩阵

> **本章目标**：理解 RunConfig 如何把一整套实验固化为一个名字，
> 以及 9 个命名配置如何构成消融矩阵。
> 分治策略：把"跑一次实验"的全部自由度收拢为一个配置对象，
> 实验间的差异完全由配置差异表达——可复现实验的基石。

## 19.1 RunConfig 的字段面

`src/pipeline.py:85` 的 `RunConfig`：一次运行的完整自由度——检索器选择
（llm_reranking / PDR 开关）、表格序列化开关、schema 体系、模型选择、
full_context、config_suffix（产物后缀）等。
Pipeline 用它实例化 QuestionsProcessor（第 13 章）。

后缀规则：`answers{config_suffix}.json` 与 `databases_ser_tab/` 等目录后缀，
让**同一份 01 解析结果派生多套下游产物并排运行**——
同一输入、不同配置的产物互不覆盖，实验矩阵得以并排展开。

## 19.2 消融矩阵

`src/pipeline.py:583` 的 `configs` 字典：

| 配置名 | 增量含义 |
|---|---|
| base | 最小：向量检索、无重排、无 PDR、gpt-4o 作答 |
| pdr | + 父文档检索 |
| max | + LLM 重排 + 表格序列化 |
| max_no_ser_tab / max_nst_o3m | max 的去表格序列化变体（nst = no ser tab） |
| max_nst_o3m | **冠军配置**：max 配方 - 表格序列化 + o3-mini 作答 |
| ibm_llama70b / ibm_llama8b | IBM 厂商对照（API 已失效） |
| gemini_thinking | full_context 对照组（第 12 章） |

注意 `preprocess_configs`（ser_tab / no_ser_tab）是**建库侧**的独立字典——
建库与问答的配置解耦：同一份解析产物可以建两套库并排实验。

阅读时对照 `configs` 区逐个定义，把每个配置与第 3~18 章的知识点对上。

> **动手**
> 1. 读 `configs` 区，给每个配置标注"相对 base 新增/移除了哪个组件"。
> 2. 找出唯一差异是"LLM 重排"的配置对，这就是量化重排贡献的消融对。

> **配置名反推练习**
> 从 `max_nst_o3m` 反推：max（满配）+ nst（no ser tab，去掉表格序列化）+
> o3m（o3-mini 作答模型）。配置命名即实验摘要。

> **自测（合并问题）**
> 1. 想量化"LLM 重排的贡献"该比较哪两个配置？为什么消融要求单变量差异？
> 2. 建库配置与问答配置为什么必须解耦？（提示：一次建库、多次问答实验复用）
