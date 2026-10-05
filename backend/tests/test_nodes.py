from __future__ import annotations

from app.agent.cache import AnswerCache, SafeRedisCache
from app.agent.nodes import (
    make_generate_node,
    make_retrieve_node,
    make_retry_node,
    make_score_node,
    retry_count_node,
    should_retry,
)
from app.agent.state import FailureReason, RetrievalMode


def base_state(**overrides):
    state = {
        "chunks": ["doc about cats", "doc about dogs", "irrelevant doc about cars"],
        "query": "tell me about cats",
        "original_query": "tell me about cats",
        "retrieval_mode": "dense",
        "retrieval_budget": 2,
        "retrieved_docs": [],
        "answer": "",
        "relevance_score": 0.0,
        "faithfulness_score": 0.0,
        "score": 0.0,
        "failure_reason": "",
        "retry_count": 0,
        "max_retries": 3,
        "healing_trace": [],
        "latency_ms": {},
        "token_usage": {},
        "cache_hits": {},
    }
    state.update(overrides)
    return state


class TestRetrieveNode:
    def test_indexes_once_and_reuses_on_retry(self, fake_store):
        node = make_retrieve_node(fake_store)
        state = base_state()

        result1 = node(state)
        assert fake_store.index_calls == 1
        assert "document_id" in result1

        state2 = {**state, "document_id": result1["document_id"]}
        node(state2)
        assert fake_store.index_calls == 1, "should not re-embed an already-indexed document"

    def test_retrieval_returns_relevant_docs_for_overlapping_query(self, fake_store):
        node = make_retrieve_node(fake_store)
        state = base_state(query="cats")
        result = node(state)
        assert any("cats" in d for d in result["retrieved_docs"])

    def test_latency_is_recorded(self, fake_store):
        node = make_retrieve_node(fake_store)
        result = node(base_state())
        assert "retrieve" in result["latency_ms"]
        assert result["latency_ms"]["retrieve"] >= 0


class TestGenerateNode:
    def test_calls_llm_and_returns_answer(self, fake_store, fake_llm):
        # disable cache entirely for this test by using a cache whose
        # underlying client is forced unavailable
        cache = AnswerCache()
        cache._cache._client = None  # force cache miss/no-op
        node = make_generate_node(fake_llm, cache=cache)

        state = base_state(document_id="abc123", retrieved_docs=["doc about cats"])
        result = node(state)
        assert result["answer"] == fake_llm.answer
        assert "generate" in result["latency_ms"]

    def test_uses_strict_prompt_when_unfaithful(self, fake_store, fake_llm):
        cache = AnswerCache()
        cache._cache._client = None
        node = make_generate_node(fake_llm, cache=cache)

        state = base_state(
            document_id="abc123",
            retrieved_docs=["doc"],
            failure_reason=FailureReason.UNFAITHFUL.value,
        )
        node(state)
        last_prompt = fake_llm.chat_text_calls[-1].lower()
        assert "faithful" in last_prompt or "support" in last_prompt or "claim" in last_prompt


class TestScoreNode:
    def test_relevant_and_sufficient_and_faithful_yields_none(self, fake_llm):
        fake_llm._judge_sequence = [{
            "relevant_docs": True, "sufficient_context": True, "relevance_score": 0.95,
            "is_faithful": True, "faithfulness_score": 0.95,
        }]
        node = make_score_node(fake_llm)
        result = node(base_state(answer="cats are mammals"))
        assert result["failure_reason"] == FailureReason.NONE.value
        assert result["score"] == 0.95

    def test_irrelevant_docs_detected(self, fake_llm):
        fake_llm._judge_sequence = [{
            "relevant_docs": False, "sufficient_context": False, "relevance_score": 0.1,
            "is_faithful": True, "faithfulness_score": 0.5,
        }]
        node = make_score_node(fake_llm)
        result = node(base_state())
        assert result["failure_reason"] == FailureReason.IRRELEVANT_DOCS.value

    def test_missing_context_detected(self, fake_llm):
        fake_llm._judge_sequence = [{
            "relevant_docs": True, "sufficient_context": False, "relevance_score": 0.6,
            "is_faithful": True, "faithfulness_score": 0.9,
        }]
        node = make_score_node(fake_llm)
        result = node(base_state())
        assert result["failure_reason"] == FailureReason.MISSING_CONTEXT.value

    def test_unfaithful_detected_independently_of_relevance(self, fake_llm):
        """A key behavior the split judges enable: docs can be perfectly
        relevant/sufficient but the answer still hallucinates."""
        fake_llm._judge_sequence = [{
            "relevant_docs": True, "sufficient_context": True, "relevance_score": 1.0,
            "is_faithful": False, "faithfulness_score": 0.2,
        }]
        node = make_score_node(fake_llm)
        result = node(base_state())
        assert result["failure_reason"] == FailureReason.UNFAITHFUL.value
        assert result["relevance_score"] == 1.0
        assert result["faithfulness_score"] == 0.2


class TestRetryNode:
    def test_missing_context_escalates_mode_and_budget(self, fake_llm):
        node = make_retry_node(fake_llm)
        state = base_state(
            retrieval_mode=RetrievalMode.DENSE.value,
            retrieval_budget=3,
            failure_reason=FailureReason.MISSING_CONTEXT.value,
            retry_count=0,
        )
        result = node(state)
        assert result["retrieval_mode"] == RetrievalMode.DENSE_RERANK.value
        assert result["retrieval_budget"] == 6  # +3 increment
        assert len(result["healing_trace"]) == 1

    def test_irrelevant_docs_escalates_mode(self, fake_llm):
        node = make_retry_node(fake_llm)
        state = base_state(
            retrieval_mode=RetrievalMode.DENSE_RERANK.value,
            retrieval_budget=3,
            failure_reason=FailureReason.IRRELEVANT_DOCS.value,
            retry_count=0,
        )
        result = node(state)
        assert result["retrieval_mode"] == RetrievalMode.HYBRID.value
        assert result["retrieval_budget"] == 5  # +2 increment

    def test_escalation_ladder_caps_at_hybrid_rerank(self, fake_llm):
        node = make_retry_node(fake_llm)
        state = base_state(
            retrieval_mode=RetrievalMode.HYBRID_RERANK.value,
            retrieval_budget=3,
            failure_reason=FailureReason.IRRELEVANT_DOCS.value,
            retry_count=0,
        )
        result = node(state)
        assert result["retrieval_mode"] == RetrievalMode.HYBRID_RERANK.value  # stays capped

    def test_unfaithful_does_not_change_retrieval_mode(self, fake_llm):
        node = make_retry_node(fake_llm)
        state = base_state(
            retrieval_mode=RetrievalMode.DENSE.value,
            retrieval_budget=3,
            failure_reason=FailureReason.UNFAITHFUL.value,
            retry_count=0,
        )
        result = node(state)
        assert result["retrieval_mode"] == RetrievalMode.DENSE.value
        assert result["retrieval_budget"] == 4

    def test_query_rewrite_triggers_at_configured_retry_count(self, fake_llm):
        node = make_retry_node(fake_llm)
        state = base_state(
            retrieval_mode=RetrievalMode.DENSE.value,
            failure_reason=FailureReason.MISSING_CONTEXT.value,
            retry_count=1,  # >= retry_count_trigger_query_rewrite (default 1)
        )
        result = node(state)
        assert "query" in result
        assert result["healing_trace"][-1]["query_rewritten"] is True

    def test_no_query_rewrite_on_first_retry(self, fake_llm):
        node = make_retry_node(fake_llm)
        state = base_state(
            retrieval_mode=RetrievalMode.DENSE.value,
            failure_reason=FailureReason.MISSING_CONTEXT.value,
            retry_count=0,
        )
        result = node(state)
        assert "query" not in result
        assert result["healing_trace"][-1]["query_rewritten"] is False


class TestShouldRetry:
    def test_retries_when_score_below_threshold_and_budget_remains(self):
        state = base_state(score=0.5, retry_count=1, max_retries=3)
        assert should_retry(state) == "retry"

    def test_ends_when_score_above_threshold(self):
        state = base_state(score=0.95, retry_count=1, max_retries=3)
        assert should_retry(state) == "end"

    def test_ends_when_retry_budget_exhausted_even_if_score_low(self):
        state = base_state(score=0.1, retry_count=3, max_retries=3)
        assert should_retry(state) == "end"


class TestRetryCountNode:
    def test_increments(self):
        result = retry_count_node(base_state(retry_count=1))
        assert result["retry_count"] == 2
