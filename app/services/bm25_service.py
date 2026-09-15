import re
import logging
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi

from app.services.cache_service import CacheService

logger = logging.getLogger("rag_app.bm25_service")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Service:
    """Lexical (keyword) retrieval over cached document chunks using BM25Okapi.

    Builds its corpus from CacheService/StorageBackend chunk data (full chunk
    text) rather than Pinecone metadata, which truncates chunk text to 1000
    chars. The corpus is not namespace-scoped - it spans all cached documents,
    since neither CacheService nor StorageBackend track Pinecone namespaces.
    """

    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache_service = cache_service or CacheService()
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: List[Dict[str, Any]] = []

    def invalidate(self) -> None:
        """Force a corpus/index rebuild on the next search() call."""
        self._bm25 = None
        self._corpus = []

    def _load_corpus(self) -> List[Dict[str, Any]]:
        corpus: List[Dict[str, Any]] = []

        for doc_id in self.cache_service.storage.list_documents():
            # file_extension is unused by LocalStorageBackend and only affects
            # S3 key layout; list_documents() doesn't expose it, matching the
            # same limitation CacheService.clear_cache() already has for the
            # "clear everything" path.
            cached = self.cache_service.load_chunks_and_embeddings(doc_id, "")
            if not cached:
                continue

            filename = cached["metadata"].get("filename", doc_id)

            for chunk in cached["chunks"]:
                corpus.append({
                    "id": f"{filename}_{chunk.get('chunk_index', 0)}",
                    "text": chunk.get("text", ""),
                    "metadata": {
                        "filename": filename,
                        "chunk_index": chunk.get("chunk_index", 0),
                        "token_count": chunk.get("token_count", 0),
                        "headings": chunk.get("headings", []),
                        "page_numbers": chunk.get("page_numbers", []),
                    },
                })

        return corpus

    def build_index(self) -> None:
        self._corpus = self._load_corpus()
        tokenized = [_tokenize(entry["text"]) for entry in self._corpus]

        if tokenized:
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

        logger.info(f"Built BM25 index over {len(self._corpus)} chunks")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self._bm25 is None:
            self.build_index()

        if self._bm25 is None or not self._corpus:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in ranked:
            entry = self._corpus[idx]
            results.append({
                "id": entry["id"],
                "score": float(scores[idx]),
                "text": entry["text"],
                "metadata": entry["metadata"],
            })

        return results
