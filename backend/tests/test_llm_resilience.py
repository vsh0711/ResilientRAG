"""
Tests for ResilientLLMClient's retry/backoff/fallback behavior — the
direct answer to "what happens when the LLM API fails", which the
original repo did not handle at all (a bare, unguarded
`client.chat.completions.create(...)` call).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
from groq import APITimeoutError, RateLimitError

from app.agent.llm import LLMCallError, ResilientLLMClient
from app.config import get_settings


def make_rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request, json={"error": "rate limited"})
    return RateLimitError("rate limited", response=response, body={"error": "rate limited"})


def make_timeout_error() -> APITimeoutError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return APITimeoutError(request=request)


def make_success_response(content: str = "ok", prompt_tokens=10, completion_tokens=5):
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    """Don't actually sleep through exponential backoff during tests."""
    monkeypatch.setattr("time.sleep", lambda _: None)


class TestRetryOnTransientError:
    def test_succeeds_after_transient_rate_limit_then_recovery(self):
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.side_effect = [
            make_rate_limit_error(),
            make_success_response("recovered answer"),
        ]
        client = ResilientLLMClient(client=mock_groq)

        result = client.chat_text(system_prompt="sys", user_prompt="hi")

        assert result.content == "recovered answer"
        assert result.attempts == 2
        assert result.used_fallback is False
        assert mock_groq.chat.completions.create.call_count == 2

    def test_retries_on_timeout_error(self):
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.side_effect = [
            make_timeout_error(),
            make_success_response("ok after timeout"),
        ]
        client = ResilientLLMClient(client=mock_groq)

        result = client.chat_text(system_prompt="sys", user_prompt="hi")
        assert result.content == "ok after timeout"
        assert result.attempts == 2


class TestFallbackModel:
    def test_falls_back_to_secondary_model_after_exhausting_primary_retries(self):
        settings = get_settings()
        mock_groq = MagicMock()
        # Exhaust all primary-model retries, then succeed on the fallback model's single attempt.
        mock_groq.chat.completions.create.side_effect = (
            [make_rate_limit_error()] * settings.llm_max_retries
            + [make_success_response("fallback saved the day")]
        )
        client = ResilientLLMClient(client=mock_groq)

        result = client.chat_text(system_prompt="sys", user_prompt="hi")

        assert result.content == "fallback saved the day"
        assert result.used_fallback is True
        assert result.model_used == settings.groq_fallback_model

        used_models = [call.kwargs["model"] for call in mock_groq.chat.completions.create.call_args_list]
        assert used_models[-1] == settings.groq_fallback_model
        assert all(m == settings.groq_primary_model for m in used_models[:-1])


class TestTotalFailure:
    def test_raises_llm_call_error_when_all_models_exhausted(self):
        settings = get_settings()
        mock_groq = MagicMock()
        total_attempts = settings.llm_max_retries + 1  # primary retries + 1 fallback attempt
        mock_groq.chat.completions.create.side_effect = [
            make_rate_limit_error() for _ in range(total_attempts)
        ]
        client = ResilientLLMClient(client=mock_groq)

        with pytest.raises(LLMCallError) as exc_info:
            client.chat_text(system_prompt="sys", user_prompt="hi")

        assert exc_info.value.attempts == total_attempts

    def test_non_retryable_error_fails_fast_without_exhausting_all_retries(self):
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.side_effect = ValueError("unexpected/non-retryable")
        client = ResilientLLMClient(client=mock_groq)

        with pytest.raises(LLMCallError):
            client.chat_text(system_prompt="sys", user_prompt="hi")
        # one attempt on primary (breaks immediately), then one attempt on fallback
        assert mock_groq.chat.completions.create.call_count == 2


class TestJsonMode:
    def test_chat_json_parses_valid_json(self):
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value = make_success_response('{"score": 0.9}')
        client = ResilientLLMClient(client=mock_groq)

        parsed, result = client.chat_json(system_prompt="sys", user_prompt="hi")
        assert parsed == {"score": 0.9}

        kwargs = mock_groq.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_chat_json_raises_on_malformed_json(self):
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value = make_success_response("not json at all")
        client = ResilientLLMClient(client=mock_groq)

        with pytest.raises(LLMCallError):
            client.chat_json(system_prompt="sys", user_prompt="hi")
