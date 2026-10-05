from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agent.judges import judge_faithfulness, judge_relevance
from app.agent.state import FaithfulnessJudgment, RelevanceJudgment


class FakeLLMReturning:
    """Minimal fake that returns a fixed (possibly malformed) payload for
    chat_json, to test schema validation in isolation from node logic."""

    def __init__(self, payload: dict):
        self._payload = payload

    def chat_json(self, *, system_prompt, user_prompt, model=None, temperature=0.0):
        from app.agent.llm import LLMResult
        return self._payload, LLMResult(
            content="{}", model_used="fake", attempts=1, used_fallback=False, latency_ms=1.0,
        )


class TestRelevanceJudgment:
    def test_valid_payload_parses(self):
        llm = FakeLLMReturning({
            "relevant_docs": True, "sufficient_context": False,
            "relevance_score": 0.7, "reasoning": "ok",
        })
        judgment, _ = judge_relevance(llm, "q", ["doc"])
        assert isinstance(judgment, RelevanceJudgment)
        assert judgment.relevant_docs is True
        assert judgment.sufficient_context is False
        assert judgment.relevance_score == 0.7

    def test_missing_field_raises_validation_error(self):
        llm = FakeLLMReturning({"relevant_docs": True})  # missing required fields
        with pytest.raises(ValidationError):
            judge_relevance(llm, "q", ["doc"])

    def test_out_of_range_score_raises(self):
        llm = FakeLLMReturning({
            "relevant_docs": True, "sufficient_context": True, "relevance_score": 1.5,
        })
        with pytest.raises(ValidationError):
            judge_relevance(llm, "q", ["doc"])


class TestFaithfulnessJudgment:
    def test_valid_payload_parses(self):
        llm = FakeLLMReturning({
            "is_faithful": False, "faithfulness_score": 0.3,
            "unsupported_claims": ["claim X not in docs"], "reasoning": "hallucinated",
        })
        judgment, _ = judge_faithfulness(llm, "q", ["doc"], "answer")
        assert isinstance(judgment, FaithfulnessJudgment)
        assert judgment.is_faithful is False
        assert judgment.unsupported_claims == ["claim X not in docs"]

    def test_defaults_apply_for_optional_fields(self):
        llm = FakeLLMReturning({"is_faithful": True, "faithfulness_score": 1.0})
        judgment, _ = judge_faithfulness(llm, "q", ["doc"], "answer")
        assert judgment.unsupported_claims == []
        assert judgment.reasoning == ""
