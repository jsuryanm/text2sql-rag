"""
Tests for CacheService (app/services/cache_service.py).

CacheService doesn't talk to disk or S3 directly - it delegates everything
to a `StorageBackend` (LocalStorageBackend or S3StorageBackend). That means
we can test CacheService's *logic* (hashing, validation, error handling)
without touching real storage at all, by handing it a small hand-written
fake backend that just remembers things in a Python dict.

This is a simpler alternative to unittest.mock.MagicMock: instead of a
generic "record every call" object, we write a tiny real class that behaves
the way a storage backend should. It's easier to read and debug.
"""

import pytest

from app.services.cache_service import CacheService
from app.services.storage_backend import StorageBackend


class FakeStorageBackend(StorageBackend):
    """An in-memory stand-in for LocalStorageBackend/S3StorageBackend."""

    def __init__(self):
        self.documents = {}  # doc_id -> {"chunks": ..., "embeddings": ..., "metadata": ...}

    def exists(self, document_id, file_extension):
        doc = self.documents.get(document_id)
        return doc is not None and all(k in doc for k in ("chunks", "embeddings", "metadata"))

    def save_document(self, document_id, file_path, file_extension):
        self.documents.setdefault(document_id, {})["document_path"] = str(file_path)

    def save_chunks(self, document_id, file_extension, chunks):
        self.documents.setdefault(document_id, {})["chunks"] = chunks

    def save_embeddings(self, document_id, file_extension, embeddings):
        self.documents.setdefault(document_id, {})["embeddings"] = embeddings

    def save_metadata(self, document_id, file_extension, metadata):
        self.documents.setdefault(document_id, {})["metadata"] = metadata

    def load_chunks(self, document_id, file_extension):
        return self.documents[document_id]["chunks"]

    def load_embeddings(self, document_id, file_extension):
        return self.documents[document_id]["embeddings"]

    def load_metadata(self, document_id, file_extension):
        return self.documents[document_id]["metadata"]

    def delete(self, document_id, file_extension):
        self.documents.pop(document_id, None)

    def list_documents(self):
        return list(self.documents.keys())

    def get_stats(self):
        return {"backend": "fake", "total_documents": len(self.documents)}


@pytest.fixture
def cache_service():
    """CacheService wired to our fake backend instead of local/S3 storage."""
    return CacheService(storage_backend=FakeStorageBackend())


class TestComputeDocumentId:
    def test_same_content_gives_same_id(self, cache_service, tmp_path):
        file_a = tmp_path / "a.txt"
        file_b = tmp_path / "b.txt"
        file_a.write_text("identical content")
        file_b.write_text("identical content")

        assert cache_service.compute_document_id(file_a) == cache_service.compute_document_id(file_b)

    def test_different_content_gives_different_id(self, cache_service, tmp_path):
        file_a = tmp_path / "a.txt"
        file_b = tmp_path / "b.txt"
        file_a.write_text("content A")
        file_b.write_text("content B")

        assert cache_service.compute_document_id(file_a) != cache_service.compute_document_id(file_b)

    def test_missing_file_raises(self, cache_service, tmp_path):
        with pytest.raises(FileNotFoundError):
            cache_service.compute_document_id(tmp_path / "does_not_exist.txt")


class TestSaveAndLoadChunksAndEmbeddings:
    def test_round_trip(self, cache_service, sample_chunks, sample_embeddings, sample_metadata):
        doc_id = "doc123"
        cache_service.save_chunks_and_embeddings(
            doc_id, "pdf", sample_chunks, sample_embeddings.tolist(), sample_metadata
        )

        result = cache_service.load_chunks_and_embeddings(doc_id, "pdf")

        assert result is not None
        assert result["chunks"] == sample_chunks
        assert result["metadata"] == sample_metadata
        assert len(result["embeddings"]) == len(sample_chunks)

    def test_mismatched_lengths_raise_value_error(self, cache_service, sample_chunks, sample_metadata):
        # 2 chunks but only 1 embedding - should be rejected before saving anything.
        with pytest.raises(ValueError):
            cache_service.save_chunks_and_embeddings(
                "doc123", "pdf", sample_chunks, [[0.1] * 1536], sample_metadata
            )

    def test_load_returns_none_when_nothing_cached(self, cache_service):
        # This is the bug that was fixed: cache_exists() being True used to
        # (incorrectly) short-circuit to None. Now it only returns None when
        # the cache genuinely doesn't exist.
        assert cache_service.load_chunks_and_embeddings("no-such-doc", "pdf") is None


class TestClearCache:
    def test_clear_single_document(self, cache_service, sample_chunks, sample_embeddings, sample_metadata):
        doc_id = "doc123"
        cache_service.save_chunks_and_embeddings(
            doc_id, "pdf", sample_chunks, sample_embeddings.tolist(), sample_metadata
        )

        result = cache_service.clear_cache(doc_id=doc_id, file_extension="pdf")

        assert result["cleared"] is True
        assert cache_service.cache_exists(doc_id, "pdf") is False

    def test_clear_single_document_requires_file_extension(self, cache_service):
        result = cache_service.clear_cache(doc_id="doc123")
        assert result["cleared"] is False

    def test_clear_all_documents(self, cache_service, sample_chunks, sample_embeddings, sample_metadata):
        for i in range(3):
            cache_service.save_chunks_and_embeddings(
                f"doc_{i}", "pdf", sample_chunks, sample_embeddings.tolist(), sample_metadata
            )

        result = cache_service.clear_cache()

        assert result["cleared"] is True
        assert result["documents_cleared"] == 3
        assert cache_service.get_cache_stats()["total_documents"] == 0
