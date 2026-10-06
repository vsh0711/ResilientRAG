"""Abandoned work must stop spending provider quota."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.agent.llm import Deadline, LLMCallError, ResilientLLMClient, deadline_var
from app.agent.nodes import should_retry


def test_a_cancelled_deadline_stops_the_call_before_it_reaches_the_provider():
    client = MagicMock()
    llm = ResilientLLMClient(client=client)
    d = Deadline(60)
    d.cancel()
    token = deadline_var.set(d)
    try:
        with pytest.raises(LLMCallError):
            llm.chat_text(system_prompt="s", user_prompt="u")
    finally:
        deadline_var.reset(token)
    client.chat.completions.create.assert_not_called()


def test_an_expired_deadline_ends_the_healing_loop():
    d = Deadline(0)
    token = deadline_var.set(d)
    try:
        state = {"score": 0.1, "retry_count": 0, "max_retries": 3, "failure_reason": "irrelevant_docs"}
        assert should_retry(state) == "end"
    finally:
        deadline_var.reset(token)


def test_without_a_deadline_behaviour_is_unchanged():
    state = {"score": 0.1, "retry_count": 0, "max_retries": 3, "failure_reason": "irrelevant_docs"}
    assert should_retry(state) == "retry"


def test_retry_backoff_never_sleeps_past_the_deadline(monkeypatch):
    from groq import RateLimitError

    slept: list[float] = []
    monkeypatch.setattr("app.agent.llm.time.sleep", lambda s: slept.append(s))
    client = MagicMock()
    err = RateLimitError("limited", response=MagicMock(headers={"retry-after": "30"}), body=None)
    client.chat.completions.create.side_effect = err
    llm = ResilientLLMClient(client=client)
    token = deadline_var.set(Deadline(0.5))
    try:
        with pytest.raises(LLMCallError):
            llm.chat_text(system_prompt="s", user_prompt="u")
    finally:
        deadline_var.reset(token)
    assert all(s <= 0.5 for s in slept)


def test_a_spent_daily_quota_switches_model_without_sleeping(monkeypatch):
    from groq import RateLimitError

    slept: list[float] = []
    monkeypatch.setattr("app.agent.llm.time.sleep", lambda s: slept.append(s))
    daily = RateLimitError(
        "Rate limit reached for model x on tokens per day (TPD): Limit 200000, Used 199999",
        response=MagicMock(headers={"retry-after": "250"}), body=None,
    )
    ok = MagicMock()
    ok.choices = [MagicMock(message=MagicMock(content="fine"))]
    ok.usage = None
    calls: list[str] = []

    def create(**kw):
        calls.append(kw["model"])
        if len(calls) == 1:
            raise daily
        return ok

    client = MagicMock()
    client.chat.completions.create.side_effect = create
    llm = ResilientLLMClient(client=client)
    result = llm.chat_text(system_prompt="s", user_prompt="u", model="primary-model")
    assert result.content == "fine"
    assert result.used_fallback
    assert len(calls) == 2 and calls[0] == "primary-model"
    assert slept == []  # never waited on a budget that will not refill
