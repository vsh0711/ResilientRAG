from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.agent.llm import LLMResult


class FakeVectorStore:
    """In-memory fake standing in for VectorStore. Supports dense/hybrid
    search over a tiny deterministic corpus without pulling in fastembed
    or qdrant-client, so node/graph tests run fast and offline."""

    def __init__(self, corpus: dict[str, list[str]] | None = None):
        # corpus[document_hash] = list of chunk texts
        self._corpus = corpus or {}
        self.index_calls = 0
        self.search_calls: list[tuple[str, str]] = []  # (mode, query)

    def is_indexed(self, document_hash: str) -> bool:
        return document_hash in self._corpus

    def index_chunks(self, document_hash: str, chunks: list[str]) -> None:
        self.index_calls += 1
        self._corpus[document_hash] = chunks

    def _rank_by_overlap(self, document_hash: str, query: str, k: int) -> list[str]:
        chunks = self._corpus.get(document_hash, [])
        query_terms = set(query.lower().split())

        def overlap(chunk: str) -> int:
            return len(query_terms & set(chunk.lower().split()))

        ranked = sorted(chunks, key=overlap, reverse=True)
        return ranked[:k]

    def search_dense(self, document_hash: str, query: str, k: int) -> list[str]:
        self.search_calls.append(("dense", query))
        return self._rank_by_overlap(document_hash, query, k)

    def search_hybrid(self, document_hash: str, query: str, k: int) -> list[str]:
        self.search_calls.append(("hybrid", query))
        return self._rank_by_overlap(document_hash, query, k)

    def rerank(self, query: str, docs: list[str]) -> list[str]:
        return docs  # identity rerank for determinism in tests


class FakeLLMClient:
    """Stands in for ResilientLLMClient. `judge_sequence` lets a test
    script a sequence of judge responses to simulate healing over
    multiple retries (e.g. fail then pass)."""

    def __init__(self, judge_sequence: list[dict] | None = None, answer: str = "A generated answer."):
        # Each entry in judge_sequence represents ONE score_node call and
        # must contain both relevance and faithfulness fields, e.g.:
        # {"relevant_docs": True, "sufficient_context": False, "relevance_score": 0.6,
        #  "is_faithful": True, "faithfulness_score": 0.9}
        default_pass = {
            "relevant_docs": True, "sufficient_context": True, "relevance_score": 1.0,
            "is_faithful": True, "faithfulness_score": 1.0,
        }
        self._judge_sequence = list(judge_sequence or [default_pass])
        self._raw_call_count = 0
        self.answer = answer
        self.chat_text_calls: list[str] = []
        self.chat_json_calls: list[str] = []

    def chat_text(self, *, system_prompt: str, user_prompt: str, model=None, temperature=0.2):
        self.chat_text_calls.append(system_prompt)
        if "rewrite" in system_prompt.lower() or "rewrites user questions" in system_prompt.lower():
            content = f"rewritten: {user_prompt[:20]}"
        else:
            content = self.answer
        return LLMResult(
            content=content, model_used="fake-model", attempts=1,
            used_fallback=False, latency_ms=1.0, prompt_tokens=10, completion_tokens=5,
            total_tokens=15,
        )

    def chat_json(self, *, system_prompt: str, user_prompt: str, model=None, temperature=0.0):
        self.chat_json_calls.append(system_prompt[:30])
        pair_index = min(self._raw_call_count // 2, len(self._judge_sequence) - 1)
        within_pair = self._raw_call_count % 2
        entry = self._judge_sequence[pair_index]
        self._raw_call_count += 1

        if within_pair == 0:
            # judge_relevance call
            payload = {
                "relevant_docs": entry["relevant_docs"],
                "sufficient_context": entry["sufficient_context"],
                "relevance_score": entry["relevance_score"],
                "reasoning": "fake",
            }
        else:
            # judge_faithfulness call
            payload = {
                "is_faithful": entry["is_faithful"],
                "faithfulness_score": entry["faithfulness_score"],
                "unsupported_claims": entry.get("unsupported_claims", []),
                "reasoning": "fake",
            }
        result = LLMResult(
            content="{}", model_used="fake-model", attempts=1, used_fallback=False,
            latency_ms=1.0, prompt_tokens=10, completion_tokens=5, total_tokens=15,
        )
        return payload, result


@pytest.fixture
def fake_store():
    return FakeVectorStore()


@pytest.fixture
def fake_llm():
    return FakeLLMClient()
