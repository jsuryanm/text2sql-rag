"""
Tests for RerankerService (app/services/reranker_service.py).

The real CrossEncoder (sentence-transformers) downloads model weights over
the network, so `_get_model()` is monkeypatched to return a fake model with
a deterministic `predict()` - same lazy-attribute-swap trick used for
Docling's DocumentConverter/HybridChunker in test_docling_service.py.
"""

from app.services.reranker_service import RerankerService


class FakeCrossEncoder:
    """Scores a (query, text) pair by how many query words appear in text."""

    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            scores.append(float(len(query_words & text_words)))
        return scores


def make_chunk(text):
    return {"text": text, "metadata": {"filename": "doc.txt", "chunk_index": 0}}


class TestRerank:
    def test_reorders_chunks_by_relevance(self, monkeypatch):
        service = RerankerService()
        monkeypatch.setattr(service, "_get_model", lambda: FakeCrossEncoder())

        chunks = [
            make_chunk("completely unrelated content"),
            make_chunk("python programming tutorial guide"),
        ]

        result = service.rerank("python tutorial", chunks, top_k=2)

        assert result[0]["text"] == "python programming tutorial guide"
        assert result[0]["rerank_score"] > result[1]["rerank_score"]

    def test_truncates_to_top_k(self, monkeypatch):
        service = RerankerService()
        monkeypatch.setattr(service, "_get_model", lambda: FakeCrossEncoder())

        chunks = [make_chunk(f"chunk number {i}") for i in range(5)]

        result = service.rerank("chunk", chunks, top_k=2)

        assert len(result) == 2

    def test_empty_chunks_returns_empty(self, monkeypatch):
        service = RerankerService()
        monkeypatch.setattr(service, "_get_model", lambda: FakeCrossEncoder())

        result = service.rerank("query", [], top_k=3)

        assert result == []

    def test_model_loaded_lazily(self):
        service = RerankerService()
        assert service._model is None
