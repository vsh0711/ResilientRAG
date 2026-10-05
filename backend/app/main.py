from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent.cache import SafeRedisCache
from app.config import get_settings
from app.exceptions import PayloadTooLargeError, RetrievalBackendError
from app.middleware import RateLimitMiddleware, RequestIDLogFilter, RequestIDMiddleware
from app.routers import documents, query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [request_id=%(request_id)s] %(name)s: %(message)s",
)
for handler in logging.getLogger().handlers:
    handler.addFilter(RequestIDLogFilter())

logger = logging.getLogger(__name__)
settings = get_settings()

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
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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
        content={"detail": "Retrieval backend is currently unavailable. Please try again shortly."},
    )


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

@app.get("/health")
async def health() -> dict:
    """Liveness probe — always returns 200 if the process is up. Does not
    check dependencies, so it can't be dragged down by Qdrant/Redis being
    slow (that's what /health/ready is for)."""
    return {"status": "ok"}


@app.get("/health/ready")
async def readiness() -> JSONResponse:
    """Readiness probe — reports dependency status without ever raising.
    A 200 with cache_available=false still means the API itself is up
    (caching fails open), so this intentionally never returns 503 on
    Redis alone; only report degraded for things that would make /query
    actually fail."""
    redis_cache = SafeRedisCache()
    qdrant_ok = True
    qdrant_detail = "ok"
    try:
        from app.agent.vectorstore import VectorStore
        VectorStore().is_indexed("__readiness_probe__")
    except Exception as exc:  # pragma: no cover - depends on live infra
        qdrant_ok = False
        qdrant_detail = str(exc)

    body = {
        "status": "ok" if qdrant_ok else "degraded",
        "dependencies": {
            "redis_cache": {"available": redis_cache.available},
            "qdrant": {"available": qdrant_ok, "detail": qdrant_detail},
            "groq_api_key_configured": bool(settings.groq_api_key),
        },
    }
    return JSONResponse(status_code=200 if qdrant_ok else 503, content=body)
