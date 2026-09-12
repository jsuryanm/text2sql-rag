"""
Tests for VectorService (app/services/vector_service.py).

VectorService wraps Pinecone (a hosted vector database) via
langchain-pinecone. Unlike CacheService, it doesn't accept a pluggable
backend through its constructor - `self.pc` (the Pinecone client) and
`self.vector_store` (the langchain-pinecone wrapper) are always built
internally. So instead of injecting a fake at construction time (like
test_cache_service.py does), these tests build a real VectorService and
then swap `service.pc` / `service.vector_store` for hand-written fakes by
plain attribute assignment - the same trick test_embeddings_service.py
uses for `service.client`. The OpenAI embeddings object built inside
connect_to_index() is neutralized with monkeypatch, since we never want a
real network call during a test.
"""

import pytest

from app.config import settings
from app.services import vector_service as module
from app.services.vector_service import VectorService


class FakePineconeClient:
    """Stands in for pinecone.Pinecone - the index-management client."""

    def __init__(self, existing_index_names=None):
        # Track which index names "already exist" before create_index() is called.
        self._existing = list(existing_index_names or [])
        self.create_index_calls = []

    def list_indexes(self):
        # Real Pinecone returns objects that support `index['name']` access.
        return [{"name": name} for name in self._existing]

    def create_index(self, name, dimension, metric, spec):
        self.create_index_calls.append(name)
        self._existing.append(name)

    def describe_index(self, name):
        class FakeIndexDescription:
            host = f"fake-host-for-{name}"

        return FakeIndexDescription()

    def Index(self, host):
        return FakeIndex()


class FakeIndex:
    """Stands in for the low-level Pinecone Index object (`self.index`)."""

    def __init__(self):
        self.upserted_batches = []

    def upsert(self, vectors, namespace):
        self.upserted_batches.append((namespace, vectors))

    def describe_index_stats(self):
        return {
            "total_vector_count": 5,
            "dimension": 1536,
            "namespaces": {"default": {"vector_count": 5}},
        }


class FakeDoc:
    """Stands in for a langchain Document returned from similarity search."""

    def __init__(self, doc_id, text, metadata):
        self.id = doc_id
        self.page_content = text
        self.metadata = metadata


class FakeVectorStore:
    """Stands in for langchain_pinecone.PineconeVectorStore."""

    def __init__(self, index):
        self.index = index
        self.deleted = []

    async def asimilarity_search_by_vector_with_score(self, embedding, k, namespace, filter):
        doc = FakeDoc(
            doc_id="doc_1",
            text="fake chunk text",
            metadata={"filename": "report.pdf", "chunk_index": 0, "token_count": 12, "headings": "[]", "page_numbers": "[]"},
        )
        return [(doc, 0.95)]

    def delete(self, filter, namespace):
        self.deleted.append((filter, namespace))


@pytest.fixture
def vector_service(monkeypatch):
    """A VectorService with no real Pinecone/OpenAI calls."""
    monkeypatch.setattr(module, "OpenAIEmbeddings", lambda model, api_key: object())
    # The real PineconeVectorStore validates that `index` is an actual
    # Pinecone Index object, which our FakeIndex isn't - so connect_to_index()
    # gets a lightweight fake constructor instead, matching FakeVectorStore's
    # shape (an `.index` attribute plus the methods tests actually call).
    monkeypatch.setattr(module, "PineconeVectorStore", lambda index, embedding: FakeVectorStore(index=index))

    service = VectorService(api_key="fake-pinecone-key", openai_api_key="fake-openai-key")
    return service


class TestConstructor:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "PINECONE_API_KEY", None)
        with pytest.raises(ValueError):
            VectorService(api_key=None)

    def test_vector_store_starts_unset(self, vector_service):
        assert vector_service.vector_store is None
        assert vector_service.index is None


class TestConnectToIndex:
    def test_creates_index_when_missing(self, vector_service):
        vector_service.pc = FakePineconeClient(existing_index_names=[])

        vector_service.connect_to_index()

        assert vector_service.pc.create_index_calls == [vector_service.index_name]
        assert vector_service.vector_store is not None

    def test_connects_without_recreating_when_index_already_exists(self, vector_service):
        """
        Regression test for the bug where connect_to_index() only built
        self.vector_store inside the "index missing" branch. When the index
        already existed, the method used to do nothing at all, leaving
        vector_store as None forever - every other method's lazy-connect
        guard (`if not self.vector_store: self.connect_to_index()`) would
        then call this repeatedly and still get None.
        """
        vector_service.pc = FakePineconeClient(existing_index_names=[vector_service.index_name])

        vector_service.connect_to_index()

        assert vector_service.pc.create_index_calls == []  # not recreated
        assert vector_service.vector_store is not None  # but still connected
        assert vector_service.index is not None


class TestAddDocuments:
    def test_upserts_one_batch(self, vector_service, sample_chunks, sample_embeddings):
        vector_service.pc = FakePineconeClient(existing_index_names=[vector_service.index_name])
        vector_service.connect_to_index()

        # sample_chunks fixture entries don't include chunk_index/token_count,
        # so add the fields add_documents() requires.
        chunks = [
            {**chunk, "chunk_index": i, "token_count": 10}
            for i, chunk in enumerate(sample_chunks)
        ]

        vector_service.add_documents(chunks, sample_embeddings.tolist(), filename="report.pdf")

        namespace, batch = vector_service.vector_store.index.upserted_batches[0]
        assert namespace == "default"
        assert len(batch) == len(chunks)
        assert batch[0][0] == "report.pdf_0"  # vector_id format: "{filename}_{chunk_index}"

    def test_mismatched_lengths_raise_value_error(self, vector_service, sample_chunks):
        vector_service.pc = FakePineconeClient(existing_index_names=[vector_service.index_name])
        vector_service.connect_to_index()

        with pytest.raises(ValueError):
            vector_service.add_documents(sample_chunks, [[0.1] * 1536], filename="report.pdf")


class TestSearch:
    async def test_returns_matched_chunks(self, vector_service):
        vector_service.vector_store = FakeVectorStore(index=FakeIndex())

        result = await vector_service.search(query_embedding=[0.1] * 1536, top_k=3)

        assert result["total_found"] == 1
        assert result["chunks"][0]["id"] == "doc_1"
        assert result["chunks"][0]["metadata"]["filename"] == "report.pdf"


class TestGetIndexStats:
    def test_reports_requested_namespace_vector_count(self, vector_service):
        vector_service.vector_store = FakeVectorStore(index=FakeIndex())

        stats = vector_service.get_index_stats(namespace="default")

        assert stats["total_vector_count"] == 5
        # Regression test for the bug where the `namespace` argument was
        # accepted but never actually used to look anything up.
        assert stats["namespace_vector_count"] == 5

    def test_unknown_namespace_reports_zero(self, vector_service):
        vector_service.vector_store = FakeVectorStore(index=FakeIndex())

        stats = vector_service.get_index_stats(namespace="no-such-namespace")

        assert stats["namespace_vector_count"] == 0


class TestDeleteByFilename:
    def test_deletes_with_filename_filter(self, vector_service):
        vector_service.vector_store = FakeVectorStore(index=FakeIndex())

        vector_service.delete_by_filename("report.pdf", namespace="default")

        filter_used, namespace_used = vector_service.vector_store.deleted[0]
        assert filter_used == {"filename": {"$eq": "report.pdf"}}
        assert namespace_used == "default"
