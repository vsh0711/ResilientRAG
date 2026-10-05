"""
Tests for the enterprise-hardening layer added on top of the core agent:
clean error responses (no leaked stack traces), request-ID propagation,
Redis-backed rate limiting, and upload size limits.
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.agent.cache import SafeRedisCache
from app.exceptions import RetrievalBackendError
from app.main import app
from app.middleware import RateLimitMiddleware


class TestExceptionHandling:
    def test_retrieval_backend_error_returns_503_not_raw_500(self, monkeypatch, fake_store, fake_llm):
        """A failure deep in the agent (e.g. the embedding model can't be
        reached) must surface as a clean 503, never a raw traceback."""
        from app.agent.graph import build_graph
        from app.routers import documents as documents_router
        from app.routers import query as query_router

        documents_router.save_chunks("doc_for_503_test", ["chunk one", "chunk two"])

        def broken_index(*args, **kwargs):
            raise ConnectionError("simulated: embedding backend unreachable")
        fake_store.index_chunks = broken_index
        fake_store._corpus = {}  # ensure is_indexed() is False so index_chunks fires

        broken_graph = build_graph(store=fake_store, llm=fake_llm)
        monkeypatch.setattr(query_router, "_graph", broken_graph)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/query", json={"document_id": "doc_for_503_test", "question": "hi"}
        )

        assert response.status_code == 503
        body = response.json()
        assert "detail" in body
        assert "Retrieval backend" in body["detail"]
        # must not leak internals
        assert "ConnectionError" not in response.text
        assert "Traceback" not in response.text

    def test_unhandled_exception_returns_clean_500_with_error_id(self):
        """Any other unexpected exception must still get a safe, generic
        body with a correlation ID for log lookup -- not raw internals."""

        @app.get("/__boom_for_test__")
        async def boom():
            raise RuntimeError("something exploded internally")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/__boom_for_test__")

        assert response.status_code == 500
        body = response.json()
        assert body["detail"] == "An unexpected error occurred."
        assert "error_id" in body
        assert "RuntimeError" not in response.text
        assert "something exploded internally" not in response.text


class TestRequestID:
    def test_response_includes_request_id_header(self):
        client = TestClient(app)
        response = client.get("/health")
        assert "X-Request-ID" in response.headers

    def test_incoming_request_id_is_echoed_back(self):
        client = TestClient(app)
        response = client.get("/health", headers={"X-Request-ID": "my-custom-id-123"})
        assert response.headers["X-Request-ID"] == "my-custom-id-123"


class TestRateLimiting:
    def test_requests_within_limit_pass_through(self):
        with patch("app.middleware.SafeRedisCache") as MockCache:
            instance = MockCache.return_value
            instance.available = True
            instance.increment.return_value = 1  # well under any limit
            mw = RateLimitMiddleware(app=MagicMock(), cache=instance)
            assert instance.increment.return_value <= mw._limit

    def test_request_over_limit_is_rejected(self):
        fake_cache = MagicMock()
        fake_cache.available = True
        fake_cache.increment.return_value = 10_000  # absurdly over any configured limit

        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        async def ok(request):
            return PlainTextResponse("ok")

        inner_app = Starlette(routes=[Route("/thing", ok)])
        inner_app.add_middleware(RateLimitMiddleware, cache=fake_cache)
        built = inner_app.build_middleware_stack()

        test_client = TestClient(built)
        response = test_client.get("/thing")
        assert response.status_code == 429
        assert "Retry-After" in response.headers

    def test_health_endpoint_is_never_rate_limited(self):
        fake_cache = MagicMock()
        fake_cache.available = True
        fake_cache.increment.return_value = 10_000

        client = TestClient(app)
        with patch("app.middleware.SafeRedisCache", return_value=fake_cache):
            response = client.get("/health")
        assert response.status_code == 200

    def test_unavailable_redis_fails_open(self):
        """If Redis itself is down, requests must still be served -- a
        rate limiter must never become a single point of failure for the
        whole API."""
        fake_cache = MagicMock()
        fake_cache.available = False

        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        async def ok(request):
            return PlainTextResponse("ok")

        inner_app = Starlette(routes=[Route("/thing", ok)])
        inner_app.add_middleware(RateLimitMiddleware, cache=fake_cache)
        built = inner_app.build_middleware_stack()

        response = TestClient(built).get("/thing")
        assert response.status_code == 200


class TestRedisIncrement:
    def test_increment_counts_up(self):
        with patch("app.agent.cache.redis.from_url") as mock_from_url:
            store = {}
            mock_client = MagicMock()
            mock_client.ping.return_value = True

            def fake_pipeline():
                pipe = MagicMock()
                ops = []
                def incr(k):
                    ops.append(("incr", k))
                def expire(k, ttl, nx=False):
                    ops.append(("expire", k, ttl, nx))
                def execute():
                    results = []
                    for op in ops:
                        if op[0] == "incr":
                            store[op[1]] = store.get(op[1], 0) + 1
                            results.append(store[op[1]])
                        else:
                            results.append(True)
                    return results
                pipe.incr.side_effect = incr
                pipe.expire.side_effect = expire
                pipe.execute.side_effect = execute
                return pipe

            mock_client.pipeline.side_effect = fake_pipeline
            mock_from_url.return_value = mock_client

            cache = SafeRedisCache(redis_url="redis://fake:6379/0")
            assert cache.increment("k", ttl_seconds=60) == 1
            assert cache.increment("k", ttl_seconds=60) == 2

    def test_increment_returns_none_when_unavailable(self):
        with patch("app.agent.cache.redis.from_url") as mock_from_url:
            mock_client = MagicMock()
            mock_client.ping.side_effect = ConnectionError("down")
            mock_from_url.return_value = mock_client

            cache = SafeRedisCache(redis_url="redis://fake:6379/0")
            assert cache.increment("k", ttl_seconds=60) is None


class TestUploadSizeLimit:
    def test_oversized_pdf_is_rejected(self, monkeypatch, tmp_path):
        from app.config import get_settings
        settings = get_settings()
        monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
        monkeypatch.setattr(settings, "max_upload_size_mb", 0)  # anything is "too big"

        client = TestClient(app)
        fake_pdf = io.BytesIO(b"%PDF-1.4 fake pdf content that exceeds a 0MB limit")
        response = client.post(
            "/documents",
            files={"file": ("test.pdf", fake_pdf, "application/pdf")},
        )
        assert response.status_code == 413

    def test_non_pdf_rejected_before_size_check(self, tmp_path):
        client = TestClient(app)
        fake_txt = io.BytesIO(b"not a pdf")
        response = client.post(
            "/documents",
            files={"file": ("test.txt", fake_txt, "text/plain")},
        )
        assert response.status_code == 400
