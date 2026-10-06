"""
Resilient Groq LLM client.

The original repo called `client.chat.completions.create(...)` directly
inside a node with no error handling at all — any transient API error,
rate limit, or timeout would crash the whole Streamlit run. This wrapper
adds:

  1. Exponential backoff retry on transient errors (rate limit, timeout,
     5xx) using the primary model.
  2. Automatic fallback to a smaller/cheaper model if the primary model
     keeps failing (or is itself rate-limited), so the agent degrades
     gracefully instead of hard-failing.
  3. Per-call latency + token usage capture, fed into the eval harness so
     the "healing" story has real cost/latency numbers attached to it.
"""
from __future__ import annotations

import json
import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional

from groq import APIStatusError, APITimeoutError, Groq, RateLimitError

from app.config import get_settings

logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (RateLimitError, APITimeoutError, APIStatusError)


@dataclass
class LLMResult:
    content: str
    model_used: str
    attempts: int
    used_fallback: bool
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMCallError(Exception):
    message: str
    attempts: int
    last_exception: Optional[Exception] = field(default=None)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class Deadline:
    """A budget shared between the request handler and the worker thread.

    Timing out a thread via asyncio.wait_for abandons it without stopping it:
    the LLM calls keep retrying and keep spending the provider's token quota
    long after nobody is listening. The handler sets this; the client checks
    it before every attempt and every sleep, so abandoned work stops.
    """

    def __init__(self, seconds: float):
        self.expires = time.monotonic() + seconds
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    @property
    def over(self) -> bool:
        return self.cancelled or time.monotonic() >= self.expires


deadline_var: ContextVar[Optional[Deadline]] = ContextVar("llm_deadline", default=None)


def _past_deadline() -> bool:
    d = deadline_var.get()
    return bool(d and d.over)


def _daily_quota_exhausted(exc: Exception) -> bool:
    """Groq reports per-day limits as '(TPD)' / '(RPD)' in a 429. Those do not recover in seconds."""
    text = str(exc)
    return isinstance(exc, RateLimitError) and ("(TPD)" in text or "(RPD)" in text)


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    """Seconds the provider asked us to wait, if it said."""
    try:
        value = exc.response.headers.get("retry-after")  # type: ignore[attr-defined]
        return float(value) if value is not None else None
    except Exception:
        return None


class ResilientLLMClient:
    """Thin wrapper around the Groq SDK with retry/backoff/fallback."""

    def __init__(self, client: Optional[Groq] = None):
        settings = get_settings()
        self._settings = settings
        self._client = client or Groq(api_key=settings.groq_api_key)

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> tuple[dict[str, Any], LLMResult]:
        """
        Call the chat completion API requesting a JSON object response and
        parse it. Raises LLMCallError if every retry/fallback attempt
        fails, or if the model returns text that isn't valid JSON.
        """
        result = self._call_with_resilience(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model or self._settings.groq_primary_model,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        try:
            parsed = json.loads(result.content)
        except json.JSONDecodeError as exc:
            raise LLMCallError(
                message=f"Model returned non-JSON content: {exc}",
                attempts=result.attempts,
                last_exception=exc,
            ) from exc
        return parsed, result

    def chat_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> LLMResult:
        return self._call_with_resilience(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model or self._settings.groq_primary_model,
            temperature=temperature,
        )

    # -- internals -----------------------------------------------------

    def _call_with_resilience(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        response_format: Optional[dict] = None,
    ) -> LLMResult:
        start = time.perf_counter()
        attempts = 0
        used_fallback = False
        last_exc: Optional[Exception] = None

        models_to_try = [model]
        if model != self._settings.groq_fallback_model:
            models_to_try.append(self._settings.groq_fallback_model)

        for model_idx, candidate_model in enumerate(models_to_try):
            max_attempts = (
                self._settings.llm_max_retries if model_idx == 0 else 1
            )
            for attempt in range(1, max_attempts + 1):
                if _past_deadline():
                    raise LLMCallError(
                        message="Stopped: the request deadline passed or the client left",
                        attempts=attempts,
                        last_exception=last_exc,
                    )
                attempts += 1
                try:
                    kwargs: dict[str, Any] = dict(
                        model=candidate_model,
                        temperature=temperature,
                        timeout=self._settings.llm_timeout_seconds,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    if response_format:
                        kwargs["response_format"] = response_format

                    response = self._client.chat.completions.create(**kwargs)
                    latency_ms = (time.perf_counter() - start) * 1000
                    usage = getattr(response, "usage", None)
                    return LLMResult(
                        content=response.choices[0].message.content,
                        model_used=candidate_model,
                        attempts=attempts,
                        used_fallback=used_fallback,
                        latency_ms=latency_ms,
                        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                        total_tokens=getattr(usage, "total_tokens", 0) or 0,
                    )
                except RETRYABLE_EXCEPTIONS as exc:
                    last_exc = exc
                    if _daily_quota_exhausted(exc):
                        # Waiting a minute cannot refill a daily budget. Go
                        # straight to the next model instead of sleeping on it.
                        logger.warning("Daily token quota exhausted for %s; switching model", candidate_model)
                        break
                    logger.warning(
                        "LLM call failed (model=%s, attempt=%d/%d): %s",
                        candidate_model, attempt, max_attempts, exc,
                    )
                    if attempt < max_attempts:
                        backoff = self._settings.llm_backoff_base_seconds * (2 ** (attempt - 1))
                        # A 429 from Groq says exactly how long until the token
                        # bucket refills. Waiting that long beats guessing.
                        retry_after = _retry_after_seconds(exc)
                        if retry_after is not None:
                            backoff = max(backoff, min(retry_after, 30.0))
                        d = deadline_var.get()
                        if d:  # never sleep past the point nobody is waiting
                            backoff = min(backoff, max(d.expires - time.monotonic(), 0.0))
                        time.sleep(backoff)
                except Exception as exc:  # non-retryable
                    last_exc = exc
                    logger.error("Non-retryable LLM error: %s", exc)
                    break

            if model_idx == 0 and len(models_to_try) > 1:
                used_fallback = True
                logger.warning(
                    "Falling back from %s to %s after exhausting retries",
                    candidate_model, models_to_try[1],
                )

        raise LLMCallError(
            message=f"All LLM attempts failed after {attempts} tries across {len(models_to_try)} model(s)",
            attempts=attempts,
            last_exception=last_exc,
        )
