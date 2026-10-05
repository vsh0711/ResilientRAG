"""
LangGraph control flow assembly.

Same shape as the original repo (retrieve -> generate -> score -> retry?
-> increment -> retrieve), but nodes are now injected (dependency
injection via factory functions in `nodes.py`) rather than importing
module-level globals, so tests and the eval harness can substitute a fake
VectorStore / fake LLM client without any monkeypatching or network calls.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agent.cache import AnswerCache
from app.agent.llm import ResilientLLMClient
from app.agent.nodes import (
    make_generate_node,
    make_retrieve_node,
    make_retry_node,
    make_score_node,
    retry_count_node,
    should_retry,
)
from app.agent.state import RAGState
from app.agent.vectorstore import VectorStore


def build_graph(
    store: VectorStore | None = None,
    llm: ResilientLLMClient | None = None,
    cache: AnswerCache | None = None,
):
    store = store or VectorStore()
    llm = llm or ResilientLLMClient()
    cache = cache or AnswerCache()

    builder = StateGraph(RAGState)

    builder.add_node("retrieve", make_retrieve_node(store))
    builder.add_node("generate", make_generate_node(llm, cache))
    builder.add_node("score", make_score_node(llm))
    builder.add_node("retry", make_retry_node(llm))
    builder.add_node("increment_retry", retry_count_node)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "score")

    builder.add_conditional_edges(
        "score",
        should_retry,
        {"retry": "retry", "end": END},
    )

    builder.add_edge("retry", "increment_retry")
    builder.add_edge("increment_retry", "retrieve")

    return builder.compile()


def initial_state(
    *,
    chunks: list[str],
    query: str,
    max_retries: int | None = None,
) -> RAGState:
    from app.config import get_settings

    settings = get_settings()
    return {
        "chunks": chunks,
        "query": query,
        "original_query": query,
        "retrieval_mode": "dense",
        "retrieval_budget": settings.default_retrieval_budget,
        "retrieved_docs": [],
        "answer": "",
        "relevance_score": 0.0,
        "faithfulness_score": 0.0,
        "score": 0.0,
        "failure_reason": "",
        "retry_count": 0,
        "max_retries": max_retries if max_retries is not None else settings.max_retries_default,
        "healing_trace": [],
        "latency_ms": {},
        "token_usage": {},
        "cache_hits": {},
    }
