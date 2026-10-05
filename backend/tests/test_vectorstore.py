"""
Tests for VectorStore plumbing (Qdrant collection creation, named dense +
sparse vectors, RRF fusion) using Qdrant's local ":memory:" mode, which
requires no network access, and deterministic fake embedding models in
place of the real HuggingFace-hosted ones.

NOTE: the real dense/sparse/reranker models (fastembed, pulled from
HuggingFace Hub) cannot be downloaded inside this build sandbox — its
outbound network is restricted to PyPI/npm/GitHub, so HuggingFace Hub is
unreachable here. These tests verify the *plumbing* (collection schema,
named-vector wiring, RRF math) is correct using fake deterministic
embedders; they do not and cannot verify real semantic retrieval quality
in this environment. Real semantic retrieval metrics are produced by
`eval/run_retrieval_eval.py`, which the project owner runs locally where
HuggingFace Hub is reachable (see eval/README.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
from qdrant_client import QdrantClient

from app.agent.vectorstore import VectorStore, reciprocal_rank_fusion


# --- deterministic fake embedding models (no network) ---------------------

VOCAB = ["cat", "dog", "car", "python", "rag", "retrieval", "healing"]


def _bow_vector(text: str) -> list[float]:
    words = text.lower().split()
    vec = [float(words.count(term)) for term in VOCAB]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeDenseModel:
    def embed(self, texts):
        return [_bow_vector(t) for t in texts]

    def query_embed(self, query):
        return [_bow_vector(query)]


@dataclass
class _SparseVec:
    indices: list
    values: list

    def as_object(self):
        return {"indices": self.indices, "values": self.values}


class FakeSparseModel:
    def embed(self, texts):
        return [self._vec(t) for t in texts]

    def query_embed(self, query):
        return [self._vec(query)]

    def _vec(self, text):
        words = set(text.lower().split())
        indices = [VOCAB.index(w) for w in words if w in VOCAB]
        values = [1.0] * len(indices)
        return _SparseVec(indices=indices, values=values)


class FakeReranker:
    def rerank(self, query, docs):
        # Score by bag-of-words cosine similarity, same signal as dense,
        # just to exercise the rerank code path deterministically.
        qv = _bow_vector(query)
        for doc in docs:
            dv = _bow_vector(doc)
            yield sum(a * b for a, b in zip(qv, dv))


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr("app.agent.vectorstore._dense_model", lambda: FakeDenseModel())
    monkeypatch.setattr("app.agent.vectorstore._sparse_model", lambda: FakeSparseModel())
    monkeypatch.setattr("app.agent.vectorstore._reranker_model", lambda: FakeReranker())
    return VectorStore(client=QdrantClient(":memory:"))


CHUNKS = [
    "cat cat dog",            # 0: about cats and dogs
    "python rag retrieval",   # 1: about python rag
    "car car car",            # 2: about cars
    "healing retrieval rag",  # 3: about healing + rag
]


class TestReciprocalRankFusion:
    def test_doc_ranked_first_in_both_lists_wins(self):
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]])
        assert fused[0] == "a"

    def test_doc_only_in_one_list_still_included(self):
        fused = reciprocal_rank_fusion([["a", "b"], ["c"]])
        assert set(fused) == {"a", "b", "c"}

    def test_empty_lists_produce_empty_result(self):
        assert reciprocal_rank_fusion([[], []]) == []


class TestVectorStoreIndexing:
    def test_index_then_is_indexed(self, store):
        doc_hash = "doc1"
        assert store.is_indexed(doc_hash) is False
        store.index_chunks(doc_hash, CHUNKS)
        assert store.is_indexed(doc_hash) is True

    def test_reindexing_same_hash_is_a_noop(self, store):
        doc_hash = "doc1"
        store.index_chunks(doc_hash, CHUNKS)
        # Should not raise even though the collection already exists.
        store.index_chunks(doc_hash, CHUNKS)
        assert store.is_indexed(doc_hash) is True


class TestVectorStoreSearch:
    def test_dense_search_returns_k_results(self, store):
        doc_hash = "doc1"
        store.index_chunks(doc_hash, CHUNKS)
        results = store.search_dense(doc_hash, "cat dog", k=2)
        assert len(results) == 2
        assert CHUNKS[0] in results  # the cat/dog chunk should surface

    def test_hybrid_search_fuses_dense_and_sparse(self, store):
        doc_hash = "doc1"
        store.index_chunks(doc_hash, CHUNKS)
        results = store.search_hybrid(doc_hash, "rag retrieval", k=2)
        assert len(results) <= 2
        # Both rag-related chunks (1 and 3) should be favored over the
        # unrelated car chunk.
        assert CHUNKS[2] not in results

    def test_rerank_reorders_by_relevance(self, store):
        docs = [CHUNKS[2], CHUNKS[0]]  # car chunk first, cat chunk second
        reranked = store.rerank("cat dog", docs)
        assert reranked[0] == CHUNKS[0]  # cat chunk should move to top
