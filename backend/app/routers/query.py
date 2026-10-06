from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from app.agent.graph import build_graph, initial_state
from app.agent.llm import Deadline, deadline_var
from app.config import get_settings
from app.exceptions import RetrievalBackendError
from app.routers.documents import load_chunks
from app.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])

# Graph is stateless across requests (all state is passed in per-invocation),
# so one compiled graph instance can safely be reused for every request.
_graph = None
_slots: asyncio.Semaphore | None = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _semaphore() -> asyncio.Semaphore:
    global _slots
    if _slots is None:
        _slots = asyncio.Semaphore(get_settings().max_concurrent_queries)
    return _slots


@router.post("", response_model=QueryResponse)
async def run_query(request: QueryRequest) -> QueryResponse | JSONResponse:
    """Run the healing loop off the event loop.

    `graph.invoke` is synchronous and holds the worker for the whole
    Groq round-trip. Awaiting it directly would stall every other
    request, including /health. A semaphore caps how many of those
    threads run at once so a burst of users queues instead of stampeding
    Groq and the embedding models.
    """
    chunks = load_chunks(request.document_id)
    graph = get_graph()
    settings = get_settings()
    state = initial_state(chunks=chunks, query=request.question, max_retries=request.max_retries)

    sem = _semaphore()
    try:
        await asyncio.wait_for(sem.acquire(), timeout=settings.query_queue_timeout_seconds)
    except TimeoutError:
        return JSONResponse(
            status_code=429,
            content={"detail": "The server is busy answering other questions. Try again in a moment."},
            headers={"Retry-After": "15"},
        )

    deadline = Deadline(settings.query_timeout_seconds)
    deadline_var.set(deadline)  # copied into the worker thread by to_thread
    try:
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(graph.invoke, state),
                timeout=settings.query_timeout_seconds,
            )
        except (TimeoutError, asyncio.CancelledError) as exc:
            deadline.cancel()  # stop the abandoned thread from spending quota
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise RetrievalBackendError("query deadline exceeded") from exc
    finally:
        sem.release()

    return _to_response(result)


def _to_response(result: dict) -> QueryResponse:
    return QueryResponse(
        answer=result["answer"],
        final_score=result["score"],
        relevance_score=result["relevance_score"],
        faithfulness_score=result["faithfulness_score"],
        failure_reason=result["failure_reason"],
        retry_count=result["retry_count"],
        retrieval_mode=result["retrieval_mode"],
        healing_trace=result["healing_trace"],
        latency_ms=result.get("latency_ms", {}),
        token_usage=result.get("token_usage", {}),
        cache_hits=result.get("cache_hits", {}),
        sources=result.get("retrieved_docs", []),
    )


def _sse(event: dict) -> bytes:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()


def _node_event(node: str, delta: dict, state: dict) -> dict | None:
    settings = get_settings()
    attempt = state.get("retry_count", 0) + 1
    if node == "retrieve":
        return {
            "type": "retrieve", "attempt": attempt, "mode": state["retrieval_mode"],
            "budget": state["retrieval_budget"], "query": state["query"],
            "docs": [d[:180] for d in delta.get("retrieved_docs", [])],
        }
    if node == "generate":
        return {"type": "generate", "attempt": attempt, "answer": delta["answer"],
                "cached": bool(state.get("cache_hits", {}).get("generate"))}
    if node == "score":
        return {
            "type": "score", "attempt": attempt,
            "relevance": delta["relevance_score"], "faithfulness": delta["faithfulness_score"],
            "combined": delta["score"], "failure_reason": delta["failure_reason"],
            "passed": delta["score"] >= settings.score_pass_threshold,
            "threshold": settings.score_pass_threshold,
        }
    if node == "retry":
        trace = delta.get("healing_trace") or []
        return {"type": "heal", "step": trace[-1]} if trace else None
    return None


@router.post("/stream", response_model=None)
async def run_query_stream(request: QueryRequest) -> StreamingResponse | JSONResponse:
    """Same loop as POST /query, narrated node by node as server-sent events."""
    chunks = load_chunks(request.document_id)
    graph = get_graph()
    settings = get_settings()
    state = initial_state(chunks=chunks, query=request.question, max_retries=request.max_retries)

    sem = _semaphore()
    try:
        await asyncio.wait_for(sem.acquire(), timeout=settings.query_queue_timeout_seconds)
    except TimeoutError:
        return JSONResponse(
            status_code=429,
            content={"detail": "The server is busy answering other questions. Try again in a moment."},
            headers={"Retry-After": "15"},
        )

    deadline = Deadline(settings.query_timeout_seconds)
    deadline_var.set(deadline)

    async def events():
        merged: dict = dict(state)
        started = time.perf_counter()
        try:
            yield _sse({"type": "start", "question": request.question,
                        "max_retries": request.max_retries})
            async for update in iterate_in_threadpool(graph.stream(state, stream_mode="updates")):
                for node, delta in update.items():
                    merged.update(delta)
                    event = _node_event(node, delta, merged)
                    if event:
                        yield _sse(event)
                if time.perf_counter() - started > settings.query_timeout_seconds:
                    yield _sse({"type": "error", "detail": "This question took too long and was stopped."})
                    return
            yield _sse({"type": "result", "result": _to_response(merged).model_dump()})
        except RetrievalBackendError as exc:
            from app.exceptions import retrieval_error_detail

            yield _sse({"type": "error", "detail": retrieval_error_detail(exc)})
        except Exception:
            logger.error("Query stream failed", exc_info=True)
            yield _sse({"type": "error", "detail": "Something went wrong while answering."})
        finally:
            deadline.cancel()  # client left or run finished: stop any remaining work
            sem.release()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
