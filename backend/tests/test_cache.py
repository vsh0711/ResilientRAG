from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.agent.cache import AnswerCache, SafeRedisCache, hash_chunks, hash_text


class TestHashing:
    def test_hash_text_deterministic(self):
        assert hash_text("hello") == hash_text("hello")

    def test_hash_text_differs_for_different_input(self):
        assert hash_text("hello") != hash_text("world")

    def test_hash_chunks_order_sensitive(self):
        assert hash_chunks(["a", "b"]) != hash_chunks(["b", "a"])


class TestSafeRedisCacheDegradesGracefully:
    def test_unreachable_redis_disables_caching_without_raising(self):
        with patch("app.agent.cache.redis.from_url") as mock_from_url:
            mock_client = MagicMock()
            mock_client.ping.side_effect = ConnectionError("no redis here")
            mock_from_url.return_value = mock_client

            cache = SafeRedisCache(redis_url="redis://nonexistent:6379/0")
            assert cache.available is False
            # get/set must no-op, not raise
            assert cache.get_json("some-key") is None
            cache.set_json("some-key", {"x": 1})  # should not raise

    def test_available_when_ping_succeeds(self):
        with patch("app.agent.cache.redis.from_url") as mock_from_url:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_from_url.return_value = mock_client

            cache = SafeRedisCache(redis_url="redis://fake:6379/0")
            assert cache.available is True

    def test_get_set_roundtrip_through_mocked_client(self):
        with patch("app.agent.cache.redis.from_url") as mock_from_url:
            store = {}
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_client.setex.side_effect = lambda k, ttl, v: store.__setitem__(k, v)
            mock_client.get.side_effect = lambda k: store.get(k)
            mock_from_url.return_value = mock_client

            cache = SafeRedisCache(redis_url="redis://fake:6379/0")
            cache.set_json("k1", {"answer": "42"})
            assert cache.get_json("k1") == {"answer": "42"}


class TestAnswerCache:
    def test_different_queries_produce_different_keys(self):
        cache = AnswerCache()
        cache._cache._client = None  # force no-op; we only test key logic here
        key1 = cache._key("doc1", "what is X?", "dense")
        key2 = cache._key("doc1", "what is Y?", "dense")
        assert key1 != key2

    def test_case_and_whitespace_normalized(self):
        cache = AnswerCache()
        key1 = cache._key("doc1", "What is X?", "dense")
        key2 = cache._key("doc1", "  what is x?  ", "dense")
        assert key1 == key2

    def test_different_retrieval_mode_produces_different_key(self):
        cache = AnswerCache()
        key1 = cache._key("doc1", "q", "dense")
        key2 = cache._key("doc1", "q", "hybrid_rerank")
        assert key1 != key2
