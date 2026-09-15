import logging
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.services.embeddings_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.bm25_service import BM25Service
from app.services.reranker_service import RerankerService

logger = logging.getLogger("rag_app.retrieval_service")


class RetrievalService:
    """Orchestrates the full retrieval pipeline used by RAGService:

    1. Pre-retrieval: LLM query rewriting + HyDE (hypothetical document embeddings)
    2. Hybrid search: dense (Pinecone) + lexical (BM25) fused via Reciprocal Rank Fusion
    3. Post-retrieval: dedupe + cross-encoder reranking

    Returns chunks in the same shape VectorService.search() already produces,
    so RAGService._build_context/_format_sources need no changes.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_service: VectorService,
        api_key: str,
        bm25_service: Optional[BM25Service] = None,
        reranker_service: Optional[RerankerService] = None,
    ):
        self.embedding_service = embedding_service
        self.vector_service = vector_service
        self.bm25_service = bm25_service or BM25Service()
        self.reranker_service = reranker_service or RerankerService()

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=api_key)

        self._rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Rewrite the user's question into a clear, standalone search query. "
             "Fix typos, expand acronyms if obvious, remove filler words. "
             "Reply with ONLY the rewritten query, no explanation."),
            ("human", "{question}"),
        ])

        self._hyde_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Write a short, plausible passage (2-4 sentences) that would answer "
             "the user's question, as if it came from a reference document. "
             "This is for retrieval purposes only - it does not need to be factually "
             "correct, just topically representative."),
            ("human", "{question}"),
        ])

    async def _rewrite_query(self, question: str) -> str:
        if not settings.ENABLE_QUERY_REWRITE:
            return question

        try:
            chain = self._rewrite_prompt | self.llm
            result = await chain.ainvoke({"question": question})
            rewritten = result.content.strip()
            return rewritten or question
        except Exception as e:
            logger.warning(f"Query rewrite failed, falling back to original question: {e}")
            return question

    async def _generate_hyde_passage(self, question: str) -> str:
        if not settings.ENABLE_HYDE:
            return question

        try:
            chain = self._hyde_prompt | self.llm
            result = await chain.ainvoke({"question": question})
            passage = result.content.strip()
            return passage or question
        except Exception as e:
            logger.warning(f"HyDE generation failed, falling back to original question: {e}")
            return question

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: List[List[Dict[str, Any]]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """Fuse multiple ranked chunk lists by Reciprocal Rank Fusion.

        RRF avoids needing to normalize incompatible score scales (BM25 vs
        cosine similarity) - it only uses each result's rank position.
        """
        fused_scores: Dict[str, float] = {}
        chunk_by_id: Dict[str, Dict[str, Any]] = {}

        for ranked_list in ranked_lists:
            for rank, chunk in enumerate(ranked_list):
                chunk_id = chunk["id"]
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
                chunk_by_id.setdefault(chunk_id, chunk)

        ordered_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)

        fused = []
        for chunk_id in ordered_ids:
            chunk = dict(chunk_by_id[chunk_id])
            chunk["score"] = fused_scores[chunk_id]
            fused.append(chunk)

        return fused

    @staticmethod
    def _dedupe(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []

        for chunk in chunks:
            key = (chunk["metadata"].get("filename"), chunk["metadata"].get("chunk_index"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(chunk)

        return deduped

    async def retrieve(
        self,
        question: str,
        top_k: int = 3,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        candidate_k = top_k * settings.RERANK_CANDIDATE_MULTIPLIER

        # Pre-retrieval
        rewritten_query = await self._rewrite_query(question)
        hyde_passage = await self._generate_hyde_passage(rewritten_query)

        hyde_embedding = await self.embedding_service.generate_single_embedding(hyde_passage)

        # Hybrid search
        dense_results = await self.vector_service.search(
            query_embedding=hyde_embedding,
            top_k=candidate_k,
            namespace=namespace,
        )
        bm25_results = self.bm25_service.search(rewritten_query, top_k=candidate_k)

        fused = self._reciprocal_rank_fusion(
            [dense_results["chunks"], bm25_results],
            k=settings.RRF_K,
        )

        # Post-retrieval
        deduped = self._dedupe(fused)
        final_chunks = self.reranker_service.rerank(question, deduped, top_k=top_k)

        return {
            "chunks": final_chunks,
            "total_found": len(final_chunks),
            "rewritten_query": rewritten_query,
        }
