"""
Query rewriting node logic.

New capability vs. the original repo: when retries keep failing on
`missing_context` (the documents are on-topic but don't cover what's
needed), simply fetching more of the *same* documents often doesn't help
— the query itself may be too narrow, too ambiguous, or phrased
differently from how the document discusses the topic. This rewrites the
query using the LLM, informed by what was already tried, before the next
retrieval attempt.
"""
from __future__ import annotations

from app.agent.llm import ResilientLLMClient

QUERY_REWRITE_SYSTEM_PROMPT = """You rewrite user questions to improve retrieval \
recall against a document corpus, without changing their meaning or intent. You are \
given the original question and the reason the previous retrieval attempt failed. \
Produce ONE rewritten question that is more likely to match the phrasing and \
terminology used in the source document. Respond with ONLY the rewritten question, \
no preamble, no quotes."""


def rewrite_query(
    llm: ResilientLLMClient, original_query: str, failure_reason: str, previous_attempts: list[str]
) -> str:
    history = "\n".join(f"- {q}" for q in previous_attempts) or "(none yet)"
    user_prompt = (
        f"Original question: {original_query}\n"
        f"Previous retrieval failure reason: {failure_reason}\n"
        f"Previously tried query phrasings:\n{history}\n\n"
        "Rewrite the question to improve retrieval."
    )
    result = llm.chat_text(
        system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3,
    )
    rewritten = result.content.strip().strip('"')
    return rewritten or original_query
