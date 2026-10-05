"""
LangGraph nodes — pure, headless, framework-agnostic.

The original repo's nodes imported `streamlit` directly and called
`st.caption(...)` inside business logic, which meant the agent could not
run outside a Streamlit session (not headlessly, not in tests, not from
an API). These nodes do not import any UI framework; they return plain
dicts to merge into state, and record structured latency/token/cache
observability data that the FastAPI layer (or a test, or the eval
harness) can read back out of `state["healing_trace"]`,
`state["latency_ms"]`, and `state["token_usage"]`.
"""
from __future__ import annotations

import time
from typing import Any

from app.agent.cache import AnswerCache, hash_chunks
from app.agent.judges import LLMCallError, judge_faithfulness, judge_relevance
from app.agent.llm import ResilientLLMClient
from app.exceptions import RetrievalBackendError
from app.agent.query_rewrite import rewrite_query
from app.agent.retrieval import retrieve
from app.agent.state import FailureReason, HealingStep, RAGState, RetrievalMode
from app.agent.vectorstore import VectorStore
from app.config import get_settings

GENERATE_SYSTEM_PROMPT = (
    "Use ONLY the following documents to answer the user's question. "
    "If the answer cannot be found in the documents, respond exactly with "
    "'I didn't find any relevant documents.' Do not use outside knowledge.\n\n"
    "Documents:\n{docs}"
)

STRICT_GENERATE_SYSTEM_PROMPT = (
    "Use ONLY the following documents to answer the user's question. "
    "Every claim in your answer MUST be directly supported by the documents below. "
    "Do not infer, extrapolate, or use outside knowledge. If the answer cannot be "
    "fully supported by the documents, say what you can confirm and explicitly note "
    "what is uncertain.\n\nDocuments:\n{docs}"
)

# Escalation ladder the self-healing loop climbs through on repeated failure.
RETRIEVAL_ESCALATION: list[RetrievalMode] = [
    RetrievalMode.DENSE,
    RetrievalMode.DENSE_RERANK,
    RetrievalMode.HYBRID,
    RetrievalMode.HYBRID_RERANK,
]


def _next_mode(current: str) -> RetrievalMode:
    current_mode = RetrievalMode(current)
    idx = RETRIEVAL_ESCALATION.index(current_mode)
    return RETRIEVAL_ESCALATION[min(idx + 1, len(RETRIEVAL_ESCALATION) - 1)]


def _record_latency(state: RAGState, node: str, ms: float) -> dict:
    latencies = dict(state.get("latency_ms", {}))
    latencies[node] = latencies.get(node, 0.0) + ms
    return latencies


def _record_tokens(state: RAGState, node: str, prompt: int, completion: int) -> dict:
    usage = dict(state.get("token_usage", {}))
    prev = usage.get(node, {"prompt_tokens": 0, "completion_tokens": 0})
    usage[node] = {
        "prompt_tokens": prev["prompt_tokens"] + prompt,
        "completion_tokens": prev["completion_tokens"] + completion,
    }
    return usage


def make_retrieve_node(store: VectorStore):
    """Factory so tests/eval can inject a fake VectorStore without
    monkeypatching module-level state."""

    def retrieve_node(state: RAGState) -> dict[str, Any]:
        t0 = time.perf_counter()
        document_hash = hash_chunks(state["chunks"])

        try:
            if not store.is_indexed(document_hash):
                store.index_chunks(document_hash, state["chunks"])

            results = retrieve(
                store,
                document_hash=document_hash,
                query=state["query"],
                k=state["retrieval_budget"],
                mode=state["retrieval_mode"],
            )
        except Exception as exc:
            # Anything here (Qdrant unreachable, embedding model can't be
            # loaded/downloaded, etc.) is an infrastructure failure, not a
            # bad request. Surface it as a typed, specific error instead
            # of letting an arbitrary low-level exception propagate up
            # through LangGraph with its raw stack trace.
            raise RetrievalBackendError(
                f"Retrieval backend failed during '{state['retrieval_mode']}' search: {exc}"
            ) from exc

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "retrieved_docs": results,
            "document_id": document_hash,
            "latency_ms": _record_latency(state, "retrieve", latency_ms),
        }

    return retrieve_node


def make_generate_node(llm: ResilientLLMClient, cache: AnswerCache | None = None):
    cache = cache or AnswerCache()

    def generate_node(state: RAGState) -> dict[str, Any]:
        t0 = time.perf_counter()
        document_hash = state.get("document_id", "")
        retrieval_mode = state["retrieval_mode"]

        cached = cache.get(document_hash, state["query"], retrieval_mode) if document_hash else None
        if cached:
            latency_ms = (time.perf_counter() - t0) * 1000
            cache_hits = dict(state.get("cache_hits", {}))
            cache_hits["generate"] = True
            return {
                "answer": cached["answer"],
                "latency_ms": _record_latency(state, "generate", latency_ms),
                "cache_hits": cache_hits,
            }

        use_strict = state.get("failure_reason") == FailureReason.UNFAITHFUL.value
        system_prompt = (STRICT_GENERATE_SYSTEM_PROMPT if use_strict else GENERATE_SYSTEM_PROMPT).format(
            docs=state["retrieved_docs"]
        )

        try:
            result = llm.chat_text(system_prompt=system_prompt, user_prompt=state["query"])
            answer = result.content
            prompt_tokens, completion_tokens = result.prompt_tokens, result.completion_tokens
        except LLMCallError:
            answer = (
                "I'm temporarily unable to generate an answer due to an upstream "
                "model error. Please try again in a moment."
            )
            prompt_tokens = completion_tokens = 0

        if document_hash:
            cache.set(document_hash, state["query"], retrieval_mode, {"answer": answer})

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "answer": answer,
            "latency_ms": _record_latency(state, "generate", latency_ms),
            "token_usage": _record_tokens(state, "generate", prompt_tokens, completion_tokens),
        }

    return generate_node


def make_score_node(llm: ResilientLLMClient):
    def score_node(state: RAGState) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            relevance, rel_result = judge_relevance(llm, state["query"], state["retrieved_docs"])
            faithfulness, faith_result = judge_faithfulness(
                llm, state["query"], state["retrieved_docs"], state["answer"]
            )
        except LLMCallError:
            # Judge itself failed — fail safe by assuming the worst, so the
            # loop either retries (if budget remains) or ends without
            # pretending the answer was validated.
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "relevance_score": 0.0,
                "faithfulness_score": 0.0,
                "score": 0.0,
                "failure_reason": FailureReason.IRRELEVANT_DOCS.value,
                "latency_ms": _record_latency(state, "score", latency_ms),
            }

        if not relevance.relevant_docs:
            failure_reason = FailureReason.IRRELEVANT_DOCS
        elif not relevance.sufficient_context:
            failure_reason = FailureReason.MISSING_CONTEXT
        elif not faithfulness.is_faithful:
            failure_reason = FailureReason.UNFAITHFUL
        else:
            failure_reason = FailureReason.NONE

        combined = 0.5 * relevance.relevance_score + 0.5 * faithfulness.faithfulness_score
        latency_ms = (time.perf_counter() - t0) * 1000
        tokens = _record_tokens(
            state, "score",
            rel_result.prompt_tokens + faith_result.prompt_tokens,
            rel_result.completion_tokens + faith_result.completion_tokens,
        )
        return {
            "relevance_score": relevance.relevance_score,
            "faithfulness_score": faithfulness.faithfulness_score,
            "score": combined,
            "failure_reason": failure_reason.value,
            "latency_ms": _record_latency(state, "score", latency_ms),
            "token_usage": tokens,
        }

    return score_node


def should_retry(state: RAGState) -> str:
    settings = get_settings()
    if state["score"] < settings.score_pass_threshold and state["retry_count"] < state["max_retries"]:
        return "retry"
    return "end"


def make_retry_node(llm: ResilientLLMClient):
    def retry_node(state: RAGState) -> dict[str, Any]:
        settings = get_settings()
        failure = FailureReason(state["failure_reason"])
        trace: list[dict] = list(state.get("healing_trace", []))
        current_mode = state["retrieval_mode"]
        current_budget = state["retrieval_budget"]
        retry_count = state["retry_count"]

        new_budget = current_budget
        new_mode = current_mode
        query_rewritten = False
        rewritten_query = None
        action = "No healing needed"

        if failure == FailureReason.MISSING_CONTEXT:
            new_budget = current_budget + settings.budget_increment_missing_context
            new_mode = _next_mode(current_mode).value
            action = (
                f"Missing context -> increased retrieval budget by "
                f"{settings.budget_increment_missing_context}, escalated retrieval "
                f"mode {current_mode} -> {new_mode}"
            )
            if retry_count >= settings.retry_count_trigger_query_rewrite:
                query_rewritten = True
                rewritten_query = rewrite_query(
                    llm,
                    original_query=state.get("original_query", state["query"]),
                    failure_reason=failure.value,
                    previous_attempts=[
                        s.get("rewritten_query") for s in trace if s.get("rewritten_query")
                    ] + [state.get("original_query", state["query"])],
                )
                action += f"; rewrote query -> '{rewritten_query}'"

        elif failure == FailureReason.IRRELEVANT_DOCS:
            new_budget = current_budget + settings.budget_increment_irrelevant_docs
            new_mode = _next_mode(current_mode).value
            action = (
                f"Irrelevant docs -> escalated retrieval mode {current_mode} -> {new_mode}, "
                f"increased budget by {settings.budget_increment_irrelevant_docs}"
            )
            if retry_count >= settings.retry_count_trigger_query_rewrite:
                query_rewritten = True
                rewritten_query = rewrite_query(
                    llm,
                    original_query=state.get("original_query", state["query"]),
                    failure_reason=failure.value,
                    previous_attempts=[
                        s.get("rewritten_query") for s in trace if s.get("rewritten_query")
                    ] + [state.get("original_query", state["query"])],
                )
                action += f"; rewrote query -> '{rewritten_query}'"

        elif failure == FailureReason.UNFAITHFUL:
            new_budget = current_budget + 1
            action = (
                "Unfaithful answer -> widened context by 1 doc and will regenerate "
                "with stricter grounding instructions (retrieval mode unchanged: "
                "this is a generation-side failure, not a retrieval-side one)"
            )

        step = HealingStep(
            retry_number=retry_count + 1,
            failure_reason=failure,
            action_taken=action,
            previous_retrieval_mode=RetrievalMode(current_mode),
            new_retrieval_mode=RetrievalMode(new_mode),
            previous_budget=current_budget,
            new_budget=new_budget,
            query_rewritten=query_rewritten,
            rewritten_query=rewritten_query,
        )
        trace.append(step.model_dump(mode="json"))

        result: dict[str, Any] = {
            "retrieval_budget": new_budget,
            "retrieval_mode": new_mode,
            "healing_trace": trace,
        }
        if rewritten_query:
            result["query"] = rewritten_query
        return result

    return retry_node


def retry_count_node(state: RAGState) -> dict[str, Any]:
    return {"retry_count": state["retry_count"] + 1}
