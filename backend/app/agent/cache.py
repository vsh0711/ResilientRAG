"""
Redis-backed cache for embeddings and generated answers.

The original repo re-embeds the *entire document* on every single retry
of the self-healing loop (see `retrieve_node` calling `embed_docs(text)`
unconditionally). For a document with a few hundred chunks that's wasted
CPU/GPU time on every retry iteration. This module gives two things:

1. Document chunk text is stored under `docchunks:{hash}` with no TTL so
   any API replica can rebuild the agent state. Embeddings themselves
   live in Qdrant, which is also the fallback if this key is missing.
2. `AnswerCache` — keyed by (document hash, normalized query, retrieval
   mode), so a repeated question against the same document skips the LLM.

Both are no-ops (always miss) if Redis is unreachable or disabled, so the
agent degrades gracefully rather than crashing when Redis isn't running.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

import redis

from app.config import get_settings

logger = logging.getLogger(__name__)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def hash_chunks(chunks: list[str]) -> str:
    joined = "\u0001".join(chunks)
    return hash_text(joined)


class SafeRedisCache:
    """Wraps redis so any connection failure degrades to a cache miss."""

    def __init__(self, redis_url: Optional[str] = None):
        settings = get_settings()
        self._enabled = settings.cache_enabled
        self._ttl = settings.cache_ttl_seconds
        self._client: Optional[redis.Redis] = None
        if self._enabled:
            try:
                self._client = redis.from_url(
                    redis_url or settings.redis_url, socket_connect_timeout=1.0
                )
                self._client.ping()
            except Exception as exc:
                logger.warning("Redis unavailable, caching disabled: %s", exc)
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def get_json(self, key: str) -> Optional[Any]:
        if not self._client:
            return None
        try:
            raw = self._client.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning("Redis GET failed for %s: %s", key, exc)
            return None

    def set_json(self, key: str, value: Any) -> None:
        if not self._client:
            return
        try:
            self._client.set(key, json.dumps(value), ex=self._ttl)
        except Exception as exc:
            logger.warning("Redis SET failed for %s: %s", key, exc)

    def set_persistent(self, key: str, value: Any) -> None:
        """Store a value with no TTL. Used for document chunks, which must
        outlive the one-hour answer cache. Redis can still drop the key
        under a memory eviction policy; Qdrant remains the durable copy."""
        if not self._client:
            return
        try:
            self._client.set(key, json.dumps(value))
        except Exception as exc:
            logger.warning("Redis SET failed for %s: %s", key, exc)

    def delete(self, key: str) -> None:
        if not self._client:
            return
        try:
            self._client.delete(key)
        except Exception as exc:
            logger.warning("Redis DEL failed for %s: %s", key, exc)

    def increment(self, key: str, ttl_seconds: int) -> Optional[int]:
        """Atomically increments a counter, setting its expiry only the
        first time it's created (so a fixed window actually expires on
        schedule rather than having its TTL pushed back on every hit).
        Returns None (never limits) if Redis is unavailable."""
        if not self._client:
            return None
        try:
            pipe = self._client.pipeline()
            pipe.incr(key)
            pipe.expire(key, ttl_seconds, nx=True)
            count, _ = pipe.execute()
            return int(count)
        except Exception as exc:
            logger.warning("Redis INCR failed for %s: %s", key, exc)
            return None


class AnswerCache:
    def __init__(self, cache: Optional[SafeRedisCache] = None):
        self._cache = cache or SafeRedisCache()

    def _key(self, document_hash: str, query: str, retrieval_mode: str) -> str:
        normalized = query.strip().lower()
        return f"answer:{document_hash}:{retrieval_mode}:{hash_text(normalized)}"

    def get(self, document_hash: str, query: str, retrieval_mode: str) -> Optional[dict]:
        return self._cache.get_json(self._key(document_hash, query, retrieval_mode))

    def set(self, document_hash: str, query: str, retrieval_mode: str, payload: dict) -> None:
        self._cache.set_json(self._key(document_hash, query, retrieval_mode), payload)

    @property
    def available(self) -> bool:
        return self._cache.available
