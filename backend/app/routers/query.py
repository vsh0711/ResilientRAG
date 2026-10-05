from __future__ import annotations

from fastapi import APIRouter

from app.agent.graph import build_graph, initial_state
from app.routers.documents import load_chunks
from app.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])

# Graph is stateless across requests (all state is passed in per-invocation),
# so one compiled graph instance can safely be reused for every request.
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


@router.post("", response_model=QueryResponse)
async def run_query(request: QueryRequest) -> QueryResponse:
    chunks = load_chunks(request.document_id)
    graph = get_graph()

    state = initial_state(chunks=chunks, query=request.question, max_retries=request.max_retries)
    result = graph.invoke(state)

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
    )
