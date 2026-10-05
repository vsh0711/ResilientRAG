"""
Retrieval strategy dispatcher.

Maps a `RetrievalMode` onto the right VectorStore calls, so the graph
nodes don't need to know how each mode is implemented. This is the seam
that turns "self-healing" from a budget/rerank toggle into a genuine
strategy switch across dense / hybrid / reranked variants.
"""
from __future__ import annotations

from app.agent.state import RetrievalMode
from app.agent.vectorstore import VectorStore


def retrieve(
    store: VectorStore,
    document_hash: str,
    query: str,
    k: int,
    mode: RetrievalMode | str,
) -> list[str]:
    mode = RetrievalMode(mode)

    if mode == RetrievalMode.DENSE:
        return store.search_dense(document_hash, query, k)

    if mode == RetrievalMode.DENSE_RERANK:
        # Over-fetch before reranking so the reranker has real signal to
        # work with, then trim back to k.
        candidates = store.search_dense(document_hash, query, k=max(k * 2, k))
        return store.rerank(query, candidates)[:k]

    if mode == RetrievalMode.HYBRID:
        return store.search_hybrid(document_hash, query, k)

    if mode == RetrievalMode.HYBRID_RERANK:
        candidates = store.search_hybrid(document_hash, query, k=max(k * 2, k))
        return store.rerank(query, candidates)[:k]

    raise ValueError(f"Unknown retrieval mode: {mode}")
