"""
LLM 重排层：对向量召回结果做语义精排（llm_reranking 配置的核心组件）。

职责定位:
    - LLMReranker: 让 gpt-4o-mini 按「与问题的相关性」给候选块打 0~1 分（结构化输出，
      温度 0），再与向量相似度加权融合成 combined_score 后排序截断 top-n。
      动机: 向量相似度对「同义不同词/指标名变体」不敏感，而年报题目常常换了措辞
      提问 —— LLM 精排补偿这一步，是比赛提分的关键技巧之一。
    - JinaReranker: 第三方 API 重排器（备用实现，主链路未使用）。

数据流位置:
    输入: VectorRetriever 的粗召结果（HybridRetriever.retrieve_by_company_name 调用）；
    输出: 带 relevance_score/combined_score 的排序列表，供 questions_processing
          拼接 RAG 上下文（top_n 截断发生在此模块或调用侧）。

评分语义契约:
    - relevance_score: 0~1（越大越相关，来源 RerankingPrompt 的评分细则）；
    - combined_score = llm_weight*relevance + (1-llm_weight)*distance —— 注意 distance
      来自 FAISS 内积相似度（越大越相似）；若未来向量索引换成 L2 距离（越小越近），
      此融合公式与排序方向必须同步反转，否则两条信号互相抵消。

核心依赖与副作用:
    - 每次重排 = 每批一次 OpenAI 结构化调用（documents_batch_size=1 时逐块调用，
      块间线程并行）；成本随候选量线性增长 —— 粗召窗口不宜过大；
    - 无全局状态；并发安全取决于 OpenAI client（本类每次调用新建 prompt，无实例缓存）。
"""
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
import src.prompts as prompts
from concurrent.futures import ThreadPoolExecutor


class JinaReranker:
    """Jina AI 重排 API 客户端（备用：换 JINA_API_KEY 即可启用，主链路未引用本类）。

    注意: top_n 由服务端截断 —— 与本地 LLMReranker 的「先全量打分再本地截断」
    语义不同，接入时需注意候选数不可控的场景。
    """

    def __init__(self):
        self.url = 'https://api.jina.ai/v1/rerank'
        self.headers = self.get_headers()
        
    def get_headers(self):
        load_dotenv()
        jina_api_key = os.getenv("JINA_API_KEY")    
        headers = {'Content-Type': 'application/json',
                   'Authorization': f'Bearer {jina_api_key}'}
        return headers
    
    def rerank(self, query, documents, top_n = 10):
        data = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "top_n": top_n,
            "documents": documents
        }

        response = requests.post(url=self.url, headers=self.headers, json=data)

        return response.json()

class LLMReranker:
    """本地 LLM 重排器：单块打分与多块成批打分双路径 + 向量分融合。

    路径选择（rerank_documents 内）:
        documents_batch_size == 1 -> 单块 prompt，块间线程并行（打分上下文最小化）；
        > 1                       -> 多块同 prompt 成批打分（省调用数，块间相对比较）。
    两条路径产出的 relevance_score 语义一致，可混用。

    防御: 多块路径下 LLM 可能少返回排名（漏块）—— 缺失块以 0.0 分补齐并打印告警，
    不让漏块静默消失（少一个候选最多是排序吃亏，绝不能丢数据）。
    """

    def __init__(self):
        self.llm = self.set_up_llm()
        self.system_prompt_rerank_single_block = prompts.RerankingPrompt.system_prompt_rerank_single_block
        self.system_prompt_rerank_multiple_blocks = prompts.RerankingPrompt.system_prompt_rerank_multiple_blocks
        self.schema_for_single_block = prompts.RetrievalRankingSingleBlock
        self.schema_for_multiple_blocks = prompts.RetrievalRankingMultipleBlocks

    def set_up_llm(self):
        load_dotenv()
        llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return llm

    def get_rank_for_single_block(self, query, retrieved_document):
        """对单个文本块请求一次相关性打分（结构化输出，返回 reasoning + relevance_score）。

        注意: prompt 里作为分隔符的 /n 是字面量而非换行符（历史笔误，未修以避免扰动
        线上行为）；对打分的影响有限 —— 评分指令全部承载于 system prompt，user prompt
        仅负责把 query 与文本送入模型。
        """
        user_prompt = f'/nHere is the query:/n"{query}"/n/nHere is the retrieved text block:/n"""/n{retrieved_document}/n"""/n'
        
        completion = self.llm.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            messages=[
                {"role": "system", "content": self.system_prompt_rerank_single_block},
                {"role": "user", "content": user_prompt},
            ],
            response_format=self.schema_for_single_block
        )

        response = completion.choices[0].message.parsed
        response_dict = response.model_dump()
        
        return response_dict

    def get_rank_for_multiple_blocks(self, query, retrieved_documents):
        formatted_blocks = "\n\n---\n\n".join([f'Block {i+1}:\n\n"""\n{text}\n"""' for i, text in enumerate(retrieved_documents)])
        user_prompt = (
            f"Here is the query: \"{query}\"\n\n"
            "Here are the retrieved text blocks:\n"
            f"{formatted_blocks}\n\n"
            f"You should provide exactly {len(retrieved_documents)} rankings, in order."
        )

        completion = self.llm.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            messages=[
                {"role": "system", "content": self.system_prompt_rerank_multiple_blocks},
                {"role": "user", "content": user_prompt},
            ],
            response_format=self.schema_for_multiple_blocks
        )

        response = completion.choices[0].message.parsed
        response_dict = response.model_dump()
      
        return response_dict

    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 4, llm_weight: float = 0.7):
        """
        重排入口: 切批 -> 并行打分 -> 融合向量分 -> 降序返回全量（截断由调用方做）。

        融合公式: combined = llm_weight * relevance + (1 - llm_weight) * distance
        distance 是 FAISS 内积相似度（大 = 好）。注意原实现中曾有注释称
        "distance is inverted since lower is better" —— 那是 L2 距离时代的遗留说法，
        与当前 IndexFlatIP 的语义相反，当前公式两个信号方向一致（都是越大越好）。

        Args:
            query: 问题原文（与打分 prompt 一起发给 LLM）
            documents: 向量召回结果（须含 text 与 distance 字段）
            documents_batch_size: 每批块数（=1 走单块打分路径）
            llm_weight: LLM 分权重（0~1）；调高更信语义、调低更信向量

        Returns:
            全量候选（含 relevance_score、combined_score），按 combined_score 降序；
            实际送入上下文的 top_n 截断在调用方（HybridRetriever）执行。
        """
        # Create batches of documents
        doc_batches = [documents[i:i + documents_batch_size] for i in range(0, len(documents), documents_batch_size)]
        vector_weight = 1 - llm_weight
        
        if documents_batch_size == 1:
            def process_single_doc(doc):
                # Get ranking for single document
                ranking = self.get_rank_for_single_block(query, doc['text'])
                
                doc_with_score = doc.copy()
                doc_with_score["relevance_score"] = ranking["relevance_score"]
                # 融合分 = 加权 LLM 相关性 + 加权向量相似度（FAISS 内积，越大越相似；
                # 与上方模块 docstring 说明一致 —— 两个信号方向同为「越大越好」）
                doc_with_score["combined_score"] = round(
                    llm_weight * ranking["relevance_score"] + 
                    vector_weight * doc['distance'],
                    4
                )
                return doc_with_score

            # Process all documents in parallel using single-block method
            with ThreadPoolExecutor() as executor:
                all_results = list(executor.map(process_single_doc, documents))
                
        else:
            def process_batch(batch):
                texts = [doc['text'] for doc in batch]
                rankings = self.get_rank_for_multiple_blocks(query, texts)
                results = []
                block_rankings = rankings.get('block_rankings', [])
                
                if len(block_rankings) < len(batch):
                    print(f"\nWarning: Expected {len(batch)} rankings but got {len(block_rankings)}")
                    for i in range(len(block_rankings), len(batch)):
                        doc = batch[i]
                        print(f"Missing ranking for document on page {doc.get('page', 'unknown')}:")
                        print(f"Text preview: {doc['text'][:100]}...\n")
                    
                    for _ in range(len(batch) - len(block_rankings)):
                        block_rankings.append({
                            "relevance_score": 0.0, 
                            "reasoning": "Default ranking due to missing LLM response"
                        })
                
                for doc, rank in zip(batch, block_rankings):
                    doc_with_score = doc.copy()
                    doc_with_score["relevance_score"] = rank["relevance_score"]
                    doc_with_score["combined_score"] = round(
                        llm_weight * rank["relevance_score"] + 
                        vector_weight * doc['distance'],
                        4
                    )
                    results.append(doc_with_score)
                return results

            # Process batches in parallel using threads
            with ThreadPoolExecutor() as executor:
                batch_results = list(executor.map(process_batch, doc_batches))
            
            # Flatten results
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)
        
        # Sort results by combined score in descending order
        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return all_results
