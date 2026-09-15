"""
Tests for RetrievalService (app/services/retrieval_service.py).

Fakes every collaborator (EmbeddingService, VectorService, BM25Service,
RerankerService) so no real OpenAI/Pinecone/model calls happen. Query
rewrite/HyDE (real LLM calls) are disabled via settings for the end-to-end
retrieve() test; the Reciprocal Rank Fusion and dedupe logic are pure
functions tested directly.
"""

import pytest

from app.config import settings
from app.services.retrieval_service import RetrievalService


def make_chunk(chunk_id, filename, chunk_index, score, text="text"):
    return {
        "id": chunk_id,
        "score": score,
        "text": text,
        "metadata": {"filename": filename, "chunk_index": chunk_index},
    }


class FakeEmbeddingService:
    async def generate_single_embedding(self, text):
        return [0.1, 0.2, 0.3]


class FakeVectorService:
    def __init__(self, chunks):
        self._chunks = chunks

    async def search(self, query_embedding, top_k, namespace):
        return {"chunks": self._chunks[:top_k], "total_found": len(self._chunks)}


class FakeBM25Service:
    def __init__(self, chunks):
        self._chunks = chunks

    def search(self, query, top_k):
        return self._chunks[:top_k]


class FakeRerankerService:
    def rerank(self, query, chunks, top_k):
        # Identity rerank for these tests - just truncate.
        return chunks[:top_k]


def make_service(dense_chunks, bm25_chunks, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_QUERY_REWRITE", False)
    monkeypatch.setattr(settings, "ENABLE_HYDE", False)

    return RetrievalService(
        embedding_service=FakeEmbeddingService(),
        vector_service=FakeVectorService(dense_chunks),
        api_key="test-key",
        bm25_service=FakeBM25Service(bm25_chunks),
        reranker_service=FakeRerankerService(),
    )


class TestReciprocalRankFusion:
    def test_chunk_ranked_first_in_both_lists_wins(self):
        list_a = [make_chunk("a", "f", 0, 0.9), make_chunk("b", "f", 1, 0.5)]
        list_b = [make_chunk("a", "f", 0, 10.0), make_chunk("c", "f", 2, 5.0)]

        fused = RetrievalService._reciprocal_rank_fusion([list_a, list_b], k=60)

        assert fused[0]["id"] == "a"

    def test_all_unique_chunks_present_in_fused_result(self):
        list_a = [make_chunk("a", "f", 0, 0.9)]
        list_b = [make_chunk("b", "f", 1, 5.0)]

        fused = RetrievalService._reciprocal_rank_fusion([list_a, list_b], k=60)

        assert {c["id"] for c in fused} == {"a", "b"}


class TestDedupe:
    def test_removes_duplicate_filename_chunk_index_pairs(self):
        chunks = [
            make_chunk("a", "f", 0, 1.0),
            make_chunk("a-dup", "f", 0, 0.5),
            make_chunk("b", "f", 1, 0.8),
        ]

        deduped = RetrievalService._dedupe(chunks)

        assert len(deduped) == 2
        assert [c["id"] for c in deduped] == ["a", "b"]


class TestRetrieve:
    @pytest.mark.asyncio
    async def test_returns_fused_reranked_chunks(self, monkeypatch):
        dense_chunks = [make_chunk("a", "f", 0, 0.9, "dense hit")]
        bm25_chunks = [make_chunk("b", "f", 1, 5.0, "bm25 hit")]
        service = make_service(dense_chunks, bm25_chunks, monkeypatch)

        result = await service.retrieve("question", top_k=2, namespace="default")

        assert result["total_found"] == 2
        assert {c["id"] for c in result["chunks"]} == {"a", "b"}
        assert result["rewritten_query"] == "question"

    @pytest.mark.asyncio
    async def test_dedupes_overlapping_dense_and_bm25_hits(self, monkeypatch):
        overlapping = make_chunk("a", "f", 0, 0.9, "same chunk")
        service = make_service([overlapping], [overlapping], monkeypatch)

        result = await service.retrieve("question", top_k=5, namespace="default")

        assert result["total_found"] == 1
