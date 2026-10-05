"""
Integration tests for the full LangGraph flow, using the fake store/LLM
from conftest so no network calls (Groq, Qdrant, Redis) are made.
"""
from __future__ import annotations

from app.agent.cache import AnswerCache
from app.agent.graph import build_graph, initial_state
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
from langgraph.graph import END, StateGraph


def _build_fake_graph(fake_store, fake_llm):
    """Builds the graph wired to fakes directly (build_graph() itself
    always constructs real VectorStore/ResilientLLMClient, so the graph
    shape is reassembled here against the injected fakes)."""
    cache = AnswerCache()
    cache._cache._client = None  # force no-op cache for determinism

    builder = StateGraph(RAGState)
    builder.add_node("retrieve", make_retrieve_node(fake_store))
    builder.add_node("generate", make_generate_node(fake_llm, cache))
    builder.add_node("score", make_score_node(fake_llm))
    builder.add_node("retry", make_retry_node(fake_llm))
    builder.add_node("increment_retry", retry_count_node)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "score")
    builder.add_conditional_edges("score", should_retry, {"retry": "retry", "end": END})
    builder.add_edge("retry", "increment_retry")
    builder.add_edge("increment_retry", "retrieve")
    return builder.compile()


class TestFullGraphFlow:
    def test_passes_immediately_when_judge_approves_first_try(self, fake_store, fake_llm):
        fake_llm._judge_sequence = [{
            "relevant_docs": True, "sufficient_context": True, "relevance_score": 0.95,
            "is_faithful": True, "faithfulness_score": 0.95,
        }]
        graph = _build_fake_graph(fake_store, fake_llm)
        state = initial_state(chunks=["cats are great", "dogs are great"], query="cats")
        result = graph.invoke(state)

        assert result["retry_count"] == 0
        assert result["failure_reason"] == "none"
        assert len(result["healing_trace"]) == 0
        assert result["score"] >= 0.8

    def test_heals_after_one_failed_attempt(self, fake_store, fake_llm):
        """First attempt fails on missing_context, second attempt passes —
        verifies the retry -> increment -> retrieve loop actually
        terminates with an improved score, which is the core self-healing
        claim of this project."""
        fake_llm._judge_sequence = [
            {"relevant_docs": True, "sufficient_context": False, "relevance_score": 0.5,
             "is_faithful": True, "faithfulness_score": 0.6},
            {"relevant_docs": True, "sufficient_context": True, "relevance_score": 0.9,
             "is_faithful": True, "faithfulness_score": 0.9},
        ]
        graph = _build_fake_graph(fake_store, fake_llm)
        state = initial_state(chunks=["cats are great", "dogs are great"], query="cats", max_retries=3)
        result = graph.invoke(state)

        assert result["retry_count"] == 1
        assert len(result["healing_trace"]) == 1
        assert result["healing_trace"][0]["failure_reason"] == "missing_context"
        assert result["score"] == 0.9
        assert result["score"] > 0.6  # demonstrably improved after healing

    def test_stops_at_max_retries_even_if_still_failing(self, fake_store, fake_llm):
        fake_llm._judge_sequence = [
            {"relevant_docs": False, "sufficient_context": False, "relevance_score": 0.1,
             "is_faithful": False, "faithfulness_score": 0.1},
        ]
        graph = _build_fake_graph(fake_store, fake_llm)
        state = initial_state(chunks=["unrelated text"], query="cats", max_retries=2)
        result = graph.invoke(state)

        assert result["retry_count"] == 2
        assert len(result["healing_trace"]) == 2
        assert result["score"] < 0.8

    def test_document_embedded_only_once_across_all_retries(self, fake_store, fake_llm):
        """Regression test for the original repo's bug: re-embedding the
        whole document on every retry."""
        fake_llm._judge_sequence = [
            {"relevant_docs": True, "sufficient_context": False, "relevance_score": 0.4,
             "is_faithful": True, "faithfulness_score": 0.4},
            {"relevant_docs": True, "sufficient_context": False, "relevance_score": 0.4,
             "is_faithful": True, "faithfulness_score": 0.4},
            {"relevant_docs": True, "sufficient_context": True, "relevance_score": 0.9,
             "is_faithful": True, "faithfulness_score": 0.9},
        ]
        graph = _build_fake_graph(fake_store, fake_llm)
        state = initial_state(chunks=["doc a", "doc b"], query="q", max_retries=5)
        graph.invoke(state)

        assert fake_store.index_calls == 1


class TestJudgeOutage:
    def test_a_judge_that_cannot_run_does_not_trigger_healing(self, fake_store, fake_llm):
        """Found in the real benchmark: a rate-limited judge returned 0.00 for
        correct answers and the loop escalated retrieval for nothing."""
        from app.agent.llm import LLMCallError

        def judge_down(**_kwargs):
            raise LLMCallError(message="rate limited", attempts=4)

        fake_llm.chat_json = judge_down
        graph = _build_fake_graph(fake_store, fake_llm)
        result = graph.invoke(initial_state(chunks=["cats are great"], query="cats"))

        assert result["failure_reason"] == "judge_unavailable"
        assert result["retry_count"] == 0
        assert result["healing_trace"] == []
        assert result["score"] == 0.0  # never claims the answer was validated
        assert result["answer"] == fake_llm.answer  # the answer is still returned
        assert len(fake_store.search_calls) == 1
