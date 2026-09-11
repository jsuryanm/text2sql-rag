"""
Tests for QueryCacheService (app/services/query_cache_service.py).

This service wraps Upstash Redis. Two things make it easy to test:

1. Without credentials, it deliberately runs in "disabled" pass-through mode
   (every get() is a miss, every set() is a no-op) so the app still works
   without Redis configured. We test that mode directly - no mocking needed.

2. With credentials, it builds a real `upstash_redis.Redis` client. We
   replace that client with a small fake using `monkeypatch`, so we can test
   the hit/miss/delete logic without a real Redis server.
"""

import pytest

from app.services.query_cache_service import QueryCacheService


class TestDisabledMode:
    """No redis_url/redis_token given -> service should be safely disabled."""

    def test_service_is_disabled_without_credentials(self):
        service = QueryCacheService()
        assert service.enabled is False

    def test_get_is_always_a_miss_when_disabled(self):
        service = QueryCacheService()
        assert service.get("some-key") is None

    def test_set_returns_false_when_disabled(self):
        service = QueryCacheService()
        assert service.set("some-key", {"value": 1}, ttl=60) is False

    def test_delete_returns_zero_when_disabled(self):
        # Bug fix check: delete() used to run its "not enabled" check backwards
        # and would try to talk to a Redis client that doesn't exist.
        service = QueryCacheService()
        assert service.delete("some-pattern:*") == 0

    def test_health_check_reports_disabled(self):
        service = QueryCacheService()
        assert service.health_check()["status"] == "disabled"


class FakeRedisClient:
    """A tiny in-memory stand-in for upstash_redis.Redis."""

    def __init__(self):
        self.store = {}

    def ping(self):
        return True

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value

    def keys(self, pattern):
        # Real Redis supports glob patterns; our fake only needs exact
        # prefix matching ("rag:*" -> anything starting with "rag:").
        prefix = pattern.rstrip("*")
        return [k for k in self.store if k.startswith(prefix)]

    def delete(self, key):
        self.store.pop(key, None)

    def flushdb(self):
        self.store.clear()


@pytest.fixture
def enabled_service(monkeypatch):
    """
    A QueryCacheService with a fake Redis client already plugged in.

    We build the service normally (credentials missing, so it starts
    disabled), then flip on `enabled` and swap in our FakeRedisClient - this
    avoids needing the real upstash_redis package or a network connection.
    """
    service = QueryCacheService()
    service.client = FakeRedisClient()
    service.enabled = True
    return service


class TestEnabledMode:
    def test_get_is_a_miss_for_unknown_key(self, enabled_service):
        assert enabled_service.get("missing-key") is None
        assert enabled_service.stats["rag"]["misses"] == 1

    def test_set_then_get_is_a_hit(self, enabled_service):
        enabled_service.set("my-key", {"answer": 42}, ttl=60)

        result = enabled_service.get("my-key")

        assert result == {"answer": 42}
        assert enabled_service.stats["rag"]["hits"] == 1

    def test_delete_removes_matching_keys(self, enabled_service):
        enabled_service.set("rag:1", {"a": 1}, ttl=60)
        enabled_service.set("rag:2", {"a": 2}, ttl=60)
        enabled_service.set("sql_gen:1", {"a": 3}, ttl=60)

        deleted = enabled_service.delete("rag:*")

        assert deleted == 2
        assert enabled_service.get("sql_gen:1", cache_type="sql_gen") == {"a": 3}

    def test_flush_all_clears_everything(self, enabled_service):
        enabled_service.set("rag:1", {"a": 1}, ttl=60)

        assert enabled_service.flush_all() is True
        assert enabled_service.get("rag:1") is None


class TestKeyGenerators:
    def test_same_text_gives_same_embeddings_key(self):
        service = QueryCacheService()
        assert service.get_embeddings_key("hello") == service.get_embeddings_key("hello")

    def test_rag_key_includes_top_k(self):
        service = QueryCacheService()
        key_top3 = service.get_rag_key("what is RAG?", top_k=3)
        key_top5 = service.get_rag_key("what is RAG?", top_k=5)
        assert key_top3 != key_top5

    def test_sql_result_key_normalizes_whitespace_and_case(self):
        service = QueryCacheService()
        key_a = service.get_sql_result_key("SELECT  *   FROM users")
        key_b = service.get_sql_result_key("select * from users")
        assert key_a == key_b


class TestStats:
    def test_get_stats_computes_hit_rate(self, enabled_service):
        enabled_service.set("my-key", {"a": 1}, ttl=60)
        enabled_service.get("my-key")  # hit
        enabled_service.get("missing")  # miss

        stats = enabled_service.get_stats()

        assert stats["cache_types"]["rag"]["hits"] == 1
        assert stats["cache_types"]["rag"]["misses"] == 1
        assert stats["cache_types"]["rag"]["hit_rate"] == "50.0%"

    def test_reset_stats_zeroes_counters(self, enabled_service):
        enabled_service.get("missing")
        enabled_service.reset_stats()

        assert enabled_service.stats["rag"] == {"hits": 0, "misses": 0}
