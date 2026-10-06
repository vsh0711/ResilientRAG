"""Without Redis the limiter must still limit, or a Redis-less demo has no protection."""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from app.config import get_settings
from app.middleware import LocalCounter, RateLimitMiddleware


def test_local_counter_counts_and_expires(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("app.middleware.time.monotonic", lambda: clock[0])
    c = LocalCounter()
    assert [c.increment("k", 60) for _ in range(3)] == [1, 2, 3]
    clock[0] += 61
    assert c.increment("k", 60) == 1


def test_with_redis_down_the_limit_still_applies(monkeypatch):
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 3)
    cache = MagicMock()
    cache.available = False

    async def ok(request):
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/thing", ok)])
    app.add_middleware(RateLimitMiddleware, cache=cache)
    client = TestClient(app.build_middleware_stack())
    codes = [client.get("/thing", headers={"X-Session-Id": "demo-session-1"}).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    cache.increment.assert_not_called()


def test_local_counter_memory_is_bounded():
    c = LocalCounter()
    for i in range(12_000):
        c.increment(f"k{i}", 0)  # all expire immediately
    c.increment("fresh", 60)
    assert len(c._counts) < 12_000
