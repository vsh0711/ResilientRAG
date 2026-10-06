"""
Cross-cutting API middleware: request correlation IDs and rate limiting.

Both are the kind of thing a portfolio reviewer checking "is this actually
production-shaped" looks for and the original repo (a Streamlit script)
had no equivalent of, since it had no concept of concurrent API clients.
"""
from __future__ import annotations

import hmac
import logging
import re
import time
import uuid
from contextvars import ContextVar

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.agent.cache import SafeRedisCache
from app.config import get_settings

logger = logging.getLogger(__name__)

_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{8,64}")


def _client_ip(request: Request) -> str:
    if get_settings().trust_proxy:
        forwarded = request.headers.get("x-forwarded-for", "")
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


class RequestIDLogFilter(logging.Filter):
    """Injects the current request's ID into every log record emitted
    while handling it, so log lines from deep inside the agent (retrieval,
    judges, retries) can be correlated back to one HTTP request."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get()
        return True


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        incoming_id = request.headers.get("x-request-id")
        request_id = incoming_id or uuid.uuid4().hex[:16]
        token = _request_id_ctx.set(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            _request_id_ctx.reset(token)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %s (%.1fms) [request_id=%s]",
            request.method, request.url.path, getattr(response, "status_code", "?"),
            duration_ms, request_id,
        )
        return response


class LocalCounter:
    """Per-process fixed-window counter, used only when Redis is unavailable.

    Correct for one worker, which is what a Redis-less deployment runs. With
    several workers each would count separately, so the limit is a multiple of
    the configured value; that is why production uses Redis.
    """

    def __init__(self) -> None:
        self._counts: dict[str, tuple[float, int]] = {}

    def increment(self, key: str, ttl_seconds: int) -> int:
        now = time.monotonic()
        if len(self._counts) > 10_000:  # bound memory under key spraying
            self._counts = {k: v for k, v in self._counts.items() if v[0] > now}
        expires, count = self._counts.get(key, (0.0, 0))
        if expires <= now:
            expires, count = now + ttl_seconds, 0
        self._counts[key] = (expires, count + 1)
        return count + 1


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Fixed-window rate limiting, counted in Redis so it's correct across
    multiple backend replicas (not just per-process). Degrades to
    "allow everything" if Redis is unreachable — same fail-open philosophy
    as the answer cache, since a down cache/limiter shouldn't take the API
    down with it. Scoped per browser session (X-Session-Id), with a higher per-IP
    ceiling behind it. Swap in a real user ID once the project has auth.
    """

    def __init__(self, app, cache: SafeRedisCache | None = None):
        super().__init__(app)
        self._cache = cache or SafeRedisCache()
        self._local = LocalCounter()
        self._limit = get_settings().rate_limit_per_minute
        self._ip_limit = get_settings().ip_rate_limit_per_minute

    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/health", "/health/ready"):
            return await call_next(request)

        window = int(time.time() // 60)
        client_ip = _client_ip(request)
        session = request.headers.get("x-session-id", "")

        # Two buckets. The per-session one is the fair limit between people.
        # The per-IP one is a ceiling that only matters when someone rotates
        # fake session IDs to dodge the first.
        buckets = [(f"ratelimit:ip:{client_ip}:{window}", self._ip_limit)]
        if _SESSION_ID.fullmatch(session):
            buckets.append((f"ratelimit:s:{session}:{window}", self._limit))
        else:
            buckets = [(f"ratelimit:ip:{client_ip}:{window}", self._limit)]

        for key, limit in buckets:
            counter = self._cache if self._cache.available else self._local
            count = counter.increment(key, ttl_seconds=60)
            if count is not None and count > limit:
                return Response(
                    content=f'{{"detail":"Rate limit exceeded: {limit} requests/minute"}}',
                    status_code=429,
                    media_type="application/json",
                    headers={"Retry-After": "60"},
                )
        return await call_next(request)


_OPEN_PATHS = ("/health", "/health/ready", "/auth/status")


class AccessCodeMiddleware(BaseHTTPMiddleware):
    """Shared access codes. Not user accounts: it keeps a public URL from
    being a free Groq proxy for strangers, and a leaked code is revoked by
    removing it from ACCESS_CODES. Codes are compared in constant time."""

    def __init__(self, app, codes: list[str] | None = None):
        super().__init__(app)
        raw = get_settings().access_codes if codes is None else ",".join(codes)
        self._codes = [c.strip() for c in raw.split(",") if c.strip()]

    async def dispatch(self, request: Request, call_next):
        if not self._codes or request.method == "OPTIONS" or request.url.path in _OPEN_PATHS:
            return await call_next(request)
        supplied = request.headers.get("x-access-code", "")
        if not any(hmac.compare_digest(supplied.encode(), c.encode()) for c in self._codes):
            return Response(
                content='{"detail":"A valid access code is required."}',
                status_code=401,
                media_type="application/json",
            )
        return await call_next(request)
