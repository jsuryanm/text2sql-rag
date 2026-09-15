import logging
from typing import List, Dict, Any, Optional

from app.config import settings

logger = logging.getLogger("rag_app.reranker_service")


class RerankerService:
    """Cross-encoder reranking of retrieved chunks against the query.

    Lazily imports/loads sentence-transformers' CrossEncoder on first use so
    importing this module (and app startup) doesn't pay the model-download
    cost - mirrors the DOCLING_AVAILABLE lazy-import pattern in
    docling_service.py.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.RERANK_MODEL
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Score each chunk's relevance to the query with a cross-encoder and
        return the top_k highest-scoring chunks, sorted descending.

        Args:
            query: The user's question
            chunks: Candidate chunks to rerank (must have a 'text' key)
            top_k: Number of chunks to keep after reranking

        Returns:
            Chunks with an added 'rerank_score' key, truncated to top_k
        """
        if not chunks:
            return []

        model = self._get_model()
        pairs = [(query, chunk.get("text", "")) for chunk in chunks]
        scores = model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]
