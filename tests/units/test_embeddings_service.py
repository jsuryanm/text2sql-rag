"""
Tests for EmbeddingService (app/services/embeddings_service.py).

Two things make this service tricky to test directly:

1. It calls OpenAI's API (`aembed_documents`) - real calls cost money and
   need network access. We replace that one method with a fake async
   function using `monkeypatch`.
2. Its methods are `async def` - pytest needs `pytest-asyncio` to run
   coroutines as tests. This project has `asyncio_mode = "auto"` set in
   pyproject.toml, so any `async def test_...` function just works, no
   `@pytest.mark.asyncio` decorator required.
"""

import pytest

from app.config import settings
from app.services.embeddings_service import EmbeddingService
from app.services.query_cache_service import QueryCacheService


class FakeEmbeddingsClient:
    """Stands in for langchain_openai.OpenAIEmbeddings."""

    def __init__(self):
        self.calls = []

    async def aembed_documents(self, texts):
        self.calls.append(texts)
        # Return a fake but correctly-shaped embedding per input text.
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture
def disabled_cache():
    """A QueryCacheService with no Redis configured - always a cache miss."""
    return QueryCacheService()


@pytest.fixture
def embedding_service(disabled_cache):
    service = EmbeddingService(api_key="fake-key-for-tests", query_cache_service=disabled_cache)
    service.client = FakeEmbeddingsClient()
    return service


class TestConstructor:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "OPENAI_API_KEY", None)
        with pytest.raises(ValueError):
            EmbeddingService(api_key=None)

    def test_passed_in_query_cache_service_is_kept(self, disabled_cache):
        # Bug fix check: the constructor used to discard the query_cache_service
        # you passed in and silently build a brand new one instead.
        service = EmbeddingService(api_key="fake-key", query_cache_service=disabled_cache)
        assert service.query_cache_service is disabled_cache

    def test_default_query_cache_service_created_when_none_given(self):
        service = EmbeddingService(api_key="fake-key", query_cache_service=None)
        assert isinstance(service.query_cache_service, QueryCacheService)


class TestGenerateEmbeddings:
    async def test_empty_list_returns_empty(self, embedding_service):
        embeddings, usage = await embedding_service.generate_embeddings([])
        assert embeddings == []
        assert usage is None

    async def test_no_cache_path_calls_openai_and_returns_all_embeddings(self, embedding_service):
        # disabled_cache.enabled is False, so this exercises the "no cache
        # available" branch at the bottom of generate_embeddings().
        embeddings, usage = await embedding_service.generate_embeddings(["hello", "world"])

        assert len(embeddings) == 2
        assert usage["model"] == "text-embedding-3-small"

    async def test_cache_hit_path_skips_openai_call(self, embedding_service):
        cache = embedding_service.query_cache_service
        cache.client = _FakeRedis()
        cache.enabled = True

        key = cache.get_embeddings_key("hello")
        cache.set(key, {"embedding": [9.0, 9.0, 9.0]}, ttl=60, cache_type="embedding")

        embeddings, usage = await embedding_service.generate_embeddings(["hello"])

        assert embeddings == [[9.0, 9.0, 9.0]]
        assert embedding_service.client.calls == []  # OpenAI was never called
        assert usage["cache_hits"] == 1
        assert usage["cache_misses"] == 0

    async def test_mixed_cache_hits_and_misses_returns_all_texts(self, embedding_service):
        """
        Regression test for the indentation bug: `return embeddings, usage_info`
        used to live inside the for-loop that fills in cache-miss results, so
        with more than one cache-miss text the function returned after only
        the first one instead of after processing all of them.
        """
        cache = embedding_service.query_cache_service
        cache.client = _FakeRedis()
        cache.enabled = True

        # Pre-cache one of three texts; the other two are cache misses.
        cache.set(cache.get_embeddings_key("cached"), {"embedding": [1.0, 1.0, 1.0]}, ttl=60, cache_type="embedding")

        embeddings, usage = await embedding_service.generate_embeddings(["cached", "miss-one", "miss-two"])

        assert len(embeddings) == 3
        assert all(e is not None for e in embeddings)
        assert usage["cache_hits"] == 1
        assert usage["cache_misses"] == 2


class _FakeRedis:
    """Same minimal fake Redis client used in test_query_cache_service.py."""

    def __init__(self):
        self.store = {}

    def ping(self):
        return True

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value

    def keys(self, pattern):
        prefix = pattern.rstrip("*")
        return [k for k in self.store if k.startswith(prefix)]

    def delete(self, key):
        self.store.pop(key, None)

    def flushdb(self):
        self.store.clear()


class TestGenerateSingleEmbedding:
    async def test_returns_first_embedding(self, embedding_service):
        # Bug fix check: generate_single_embedding used to call
        # generate_embeddings() without `await`, which would return a
        # coroutine object instead of a (embeddings, usage) tuple and crash.
        embedding = await embedding_service.generate_single_embedding("hello")
        assert embedding == [0.1, 0.2, 0.3]


class TestGetEmbeddingDimension:
    def test_returns_1536(self, embedding_service):
        assert embedding_service.get_embedding_dimension() == 1536
