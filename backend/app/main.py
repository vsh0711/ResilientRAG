from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent.cache import SafeRedisCache
from app.config import get_settings
from app.exceptions import PayloadTooLargeError, RetrievalBackendError, UnreadableDocumentError, retrieval_error_detail
from app.middleware import AccessCodeMiddleware, RateLimitMiddleware, RequestIDLogFilter, RequestIDMiddleware
from app.routers import documents, query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [request_id=%(request_id)s] %(name)s: %(message)s",
)
for handler in logging.getLogger().handlers:
    handler.addFilter(RequestIDLogFilter())

logger = logging.getLogger(__name__)
settings = get_settings()


def _warmup_models() -> None:
    from app.agent.vectorstore import _dense_model, _reranker_model, _sparse_model

    _dense_model()
    _sparse_model()
    _reranker_model()
    logger.info("Embedding models are loaded")


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    # asyncio's default pool is min(32, cpus + 4), about 10 here. A dozen
    # questions blocked on LLM calls would use all of it and starve every other
    # to_thread caller, including the readiness probe (measured: 68 s p95).
    cfg = get_settings()
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=cfg.max_concurrent_queries + cfg.max_concurrent_uploads + 16)
    )
    if get_settings().warmup_models:
        import asyncio

        try:
            await asyncio.to_thread(_warmup_models)
        except Exception:
            logger.exception("Embedding model warmup failed; the first upload will retry it")
    yield


app = FastAPI(
    title="ResilientRAG API",
    description=(
        "Self-healing Retrieval-Augmented Generation agent. Detects retrieval "
        "and faithfulness failures via split LLM judges and automatically "
        "escalates retrieval strategy (dense -> reranked -> hybrid -> hybrid+"
        "reranked), rewrites queries, and retries until quality improves or "
        "the retry budget is exhausted."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AccessCodeMiddleware)  # innermost: rate limiting still counts rejected calls
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestIDMiddleware)

app.include_router(documents.router)
app.include_router(query.router)


# --- Exception handlers -----------------------------------------------
# Map internal/operational failures to clean, specific JSON responses
# instead of leaking a raw stack trace to the client. The real traceback
# is still logged server-side, tagged with the request ID, for debugging.

@app.exception_handler(RetrievalBackendError)
async def retrieval_backend_error_handler(request: Request, exc: RetrievalBackendError):
    logger.error("Retrieval backend error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=503,
        content={"detail": retrieval_error_detail(exc)},
    )


@app.exception_handler(UnreadableDocumentError)
async def unreadable_document_handler(request: Request, exc: UnreadableDocumentError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(PayloadTooLargeError)
async def payload_too_large_handler(request: Request, exc: PayloadTooLargeError):
    return JSONResponse(status_code=413, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    error_id = uuid.uuid4().hex[:8]
    logger.error("Unhandled exception [error_id=%s]: %s", error_id, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred.",
            "error_id": error_id,
        },
    )


# --- Health -------------------------------------------------------------

@app.get("/auth/status")
async def auth_status() -> dict:
    """Lets the UI know whether to show the access-code screen."""
    return {"required": bool([c for c in get_settings().access_codes.split(",") if c.strip()])}


@app.get("/health")
async def health() -> dict:
    """Liveness probe — always returns 200 if the process is up. Does not
    check dependencies, so it can't be dragged down by Qdrant/Redis being
    slow (that's what /health/ready is for)."""
    return {"status": "ok"}


from concurrent.futures import ThreadPoolExecutor as _TPE

# Probes get their own threads so a busy app can still say whether it is healthy.
_PROBE_POOL = _TPE(max_workers=2, thread_name_prefix="probe")

_model_check: tuple[float, list[str]] = (0.0, [])


async def _missing_llm_models() -> list[str]:
    """Configured Groq models the key cannot see. Cached for five minutes.

    Providers retire models. A retired name makes every answer and every
    judge call fail, and nothing else in the stack notices, so readiness does.
    """
    import asyncio
    import time

    global _model_check
    if not settings.groq_api_key:
        return []
    checked_at, missing = _model_check
    if checked_at and time.monotonic() - checked_at < 300:
        return missing

    def _list() -> list[str]:
        from groq import Groq

        have = {m.id for m in Groq(api_key=settings.groq_api_key, timeout=5).models.list().data}
        wanted = {settings.groq_primary_model, settings.groq_fallback_model, settings.groq_judge_model}
        return sorted(wanted - have)

    try:
        missing = await asyncio.get_running_loop().run_in_executor(_PROBE_POOL, _list)
    except Exception:
        return []  # Groq unreachable is a different problem; do not claim models are missing
    _model_check = (time.monotonic(), missing)
    return missing


@app.get("/health/ready")
async def readiness() -> JSONResponse:
    """Readiness probe — reports dependency status without ever raising.
    A 200 with cache_available=false still means the API itself is up
    (caching fails open), so this intentionally never returns 503 on
    Redis alone; only report degraded for things that would make /query
    actually fail.

    The Redis and Qdrant clients are synchronous. Calling them on the event
    loop froze every request behind a slow dependency (measured: 38 s), so
    they run in worker threads."""
    import asyncio

    def _probe() -> tuple[bool, bool, str]:
        redis_ok = SafeRedisCache().available
        try:
            from app.agent.vectorstore import VectorStore

            VectorStore().is_indexed("__readiness_probe__")
            return redis_ok, True, "ok"
        except Exception as exc:  # pragma: no cover - depends on live infra
            return redis_ok, False, str(exc)

    redis_ok, qdrant_ok, qdrant_detail = await asyncio.get_running_loop().run_in_executor(_PROBE_POOL, _probe)
    llm_missing = await _missing_llm_models()
    llm_ok = not llm_missing
    body = {
        "status": "ok" if (qdrant_ok and llm_ok) else "degraded",
        "dependencies": {
            "redis_cache": {"available": redis_ok},
            "qdrant": {"available": qdrant_ok, "detail": qdrant_detail},
            "groq_api_key_configured": bool(settings.groq_api_key),
            "llm_models": {"available": llm_ok, "missing": llm_missing},
        },
    }
    return JSONResponse(status_code=200 if (qdrant_ok and llm_ok) else 503, content=body)
