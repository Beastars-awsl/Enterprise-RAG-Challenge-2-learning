# 第 9 章 父文档检索（PDR）

> **本章目标**：逐行精读 `VectorRetriever.retrieve_by_company_name`——全项目
> 命名的主角（提交名 "Ilia_Ris SO CoT + Parent Document Retrieval"）。
> 分治策略：检索器内部再分治为 **路由 → 向量化 → top-k → 上卷去重** 四段，
> 每段一个可独立替换的决策点。

## 9.1 检索主入口精读

`src/retrieval.py:225` 的 `retrieve_by_company_name`，四段流程：

1. **公司路由**：遍历 `all_dbs`，精确匹配 `metainfo.company_name`——
   纯 Python 字典式路由，零计算成本（第 7 章分库架构的直接收益）
2. **查询向量化**：与建库同模型（`text-embedding-3-large`）、同样单位化，
   IP 分数即余弦
3. **FAISS top-k**：`vector_db.search(x, k)`；k 不允许超过索引样本数
   （`actual_top_n = min(top_n, len(chunks))`，否则 FAISS 报错）
4. **父文档上卷**：chunk 命中 → 找到所在页 → 返回**整页文本**；
   `seen_pages` 集合按页号去重——FAISS 返回按相似度降序，每页号的首次出现
   即该页最优 chunk，去重后整页进上下文，且**页间不互相挤占名额**

`return_parent_pages` 开关：False 返回 chunk 级结果（消融对照用），
True 即 PDR 模式。开/关对比是第 22 章消融实验的关键一组。

## 9.2 retrieve_all：无检索路径

`src/retrieval` 的 `retrieve_all`（第 303 行）：返回整本报告全部页面，
按页码升序。`distance` 固定 0.5 只是占位——**页面按页码排序而非相似度**，
消费方不应对 distance 做语义假设。这是第 12 章 full_context 模式的底层。

## 9.3 PDR 的收益与代价

| 维度 | chunk 级返回 | 父页级返回（PDR） |
|---|---|---|
| 检索精度 | 高（小块匹配） | 不变（仍小块匹配） |
| 上下文完整性 | 差 | **整页语义完整** |
| token 消耗 | 低 | 高（页 > 块） |
| 典型失败 | 答案被截断在块边界 | 无关段落稀释上下文 |

收益来自 6.1 节的矛盾化解：**索引小块，返回大块**。

> **动手**
> 1. 用 test_set 的向量库，把 `return_parent_pages` 开/关各跑一次同一道题，
>    对比返回上下文的质量与 token 量。
> 2. 验证去重逻辑：构造一个 top_n=6 但两 chunk 同页的场景，确认只返回 5 页。

> **自测（合并父问题）**
> 1. 上卷到"页"而不是"章"或"节"，为什么页是年报的合适粒度？
>    （提示：docling 解析保真度、token 预算、题目的页级引用格式）
> 2. 去重后页数可能 < top_n，对下游 prompt 的上下文格式化有什么影响？
