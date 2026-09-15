"""
Tests for BM25Service (app/services/bm25_service.py).

BM25Service builds its corpus from CacheService.load_chunks_and_embeddings(),
so we inject a fake CacheService whose storage returns known chunk text -
no real filesystem/S3 access, no real BM25 network calls (rank_bm25 is a
pure-Python in-memory library, so the real BM25Okapi class is used directly).
"""

from app.services.bm25_service import BM25Service


class FakeStorage:
    def __init__(self, documents):
        # documents: dict[doc_id -> list of chunk dicts]
        self._documents = documents

    def list_documents(self):
        return list(self._documents.keys())


class FakeCacheService:
    def __init__(self, documents):
        self.storage = FakeStorage(documents)
        self._documents = documents

    def load_chunks_and_embeddings(self, doc_id, file_extension):
        chunks = self._documents.get(doc_id)
        if chunks is None:
            return None
        return {
            "chunks": chunks,
            "embeddings": [[0.0] * 3 for _ in chunks],
            "metadata": {"filename": f"{doc_id}.txt"},
        }


def make_chunk(text, chunk_index=0):
    return {
        "text": text,
        "chunk_index": chunk_index,
        "token_count": len(text.split()),
        "headings": [],
        "page_numbers": [],
    }


class TestBM25Search:
    def test_ranks_lexically_matching_chunk_first(self):
        documents = {
            "doc1": [make_chunk("The invoice number is INV-98234 for this order.")],
            "doc2": [make_chunk("The weather today is sunny and warm.")],
            "doc3": [make_chunk("Unrelated content about gardening tips.")],
        }
        service = BM25Service(cache_service=FakeCacheService(documents))

        results = service.search("INV-98234 invoice", top_k=3)

        assert results[0]["text"] == documents["doc1"][0]["text"]
        assert results[0]["score"] > results[1]["score"]

    def test_returns_chunks_shaped_like_vector_service(self):
        documents = {"doc1": [make_chunk("Some searchable text here")]}
        service = BM25Service(cache_service=FakeCacheService(documents))

        results = service.search("searchable text", top_k=1)

        assert len(results) == 1
        chunk = results[0]
        assert set(["id", "score", "text", "metadata"]).issubset(chunk.keys())
        assert chunk["metadata"]["filename"] == "doc1.txt"
        assert chunk["metadata"]["chunk_index"] == 0

    def test_empty_corpus_returns_empty_results(self):
        service = BM25Service(cache_service=FakeCacheService({}))

        results = service.search("anything", top_k=5)

        assert results == []

    def test_invalidate_forces_rebuild(self):
        documents = {"doc1": [make_chunk("original text")]}
        cache_service = FakeCacheService(documents)
        service = BM25Service(cache_service=cache_service)

        service.search("original", top_k=1)
        assert len(service._corpus) == 1

        documents["doc2"] = [make_chunk("second document text")]
        service.invalidate()
        service.search("second", top_k=2)

        assert len(service._corpus) == 2
