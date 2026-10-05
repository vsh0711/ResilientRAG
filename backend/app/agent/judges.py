"""
LLM-as-a-judge, split into two independent judges.

The original repo's single judge conflated "are the docs relevant/
sufficient" and "is the answer good" into one blended `score`, and parsed
the response with a bare `json.loads()` — no schema enforcement, so a
malformed response crashes the node.

Here:
  * `judge_relevance` scores retrieval quality only (did we find the
    right documents).
  * `judge_faithfulness` scores generation quality only (is the answer
    actually grounded in what was retrieved, independent of whether the
    retrieval was good) — this is the piece needed to tell a retrieval
    failure apart from a hallucination.
  * Both respond via Groq's JSON-mode (`response_format={"type":
    "json_object"}`) and are validated through a Pydantic model, so a
    malformed response raises a typed error instead of corrupting state.
"""
from __future__ import annotations

from app.agent.llm import LLMCallError, LLMResult, ResilientLLMClient
from app.agent.state import FaithfulnessJudgment, RelevanceJudgment
from app.config import get_settings

RELEVANCE_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of Retrieval-Augmented \
Generation (RAG) systems. You evaluate ONLY whether the retrieved documents are \
relevant to and sufficient for answering the user's question. You do NOT evaluate \
the generated answer's wording or correctness — a separate judge does that.

Respond with a JSON object matching exactly this schema:
{
  "relevant_docs": true | false,
  "sufficient_context": true | false,
  "relevance_score": <float 0.0-1.0>,
  "reasoning": "<one sentence>"
}

Guidelines:
- relevant_docs = false if the documents do not address the topic of the question at all.
- sufficient_context = false if the documents are on-topic but missing details needed \
to fully answer the question.
- relevance_score should reflect overall retrieval quality (topic match + completeness).
"""

FAITHFULNESS_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of hallucination in \
LLM-generated answers. Given retrieved documents and a generated answer, determine \
whether every factual claim in the answer is supported by the documents. Do NOT \
evaluate whether the right documents were retrieved — a separate judge does that. \
Judge faithfulness even if the documents are an imperfect match for the question.

Respond with a JSON object matching exactly this schema:
{
  "is_faithful": true | false,
  "faithfulness_score": <float 0.0-1.0>,
  "unsupported_claims": ["<claim not found in documents>", ...],
  "reasoning": "<one sentence>"
}
"""


def judge_relevance(
    llm: ResilientLLMClient, query: str, retrieved_docs: list[str]
) -> tuple[RelevanceJudgment, LLMResult]:
    user_prompt = (
        f"User question:\n{query}\n\n"
        f"Retrieved documents:\n{retrieved_docs}\n\n"
        "Evaluate relevance and sufficiency per the schema."
    )
    parsed, llm_result = llm.chat_json(
        system_prompt=RELEVANCE_JUDGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=get_settings().groq_judge_model,
    )
    return RelevanceJudgment.model_validate(parsed), llm_result


def judge_faithfulness(
    llm: ResilientLLMClient, query: str, retrieved_docs: list[str], answer: str
) -> tuple[FaithfulnessJudgment, LLMResult]:
    user_prompt = (
        f"User question:\n{query}\n\n"
        f"Retrieved documents:\n{retrieved_docs}\n\n"
        f"Generated answer:\n{answer}\n\n"
        "Evaluate faithfulness per the schema."
    )
    parsed, llm_result = llm.chat_json(
        system_prompt=FAITHFULNESS_JUDGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=get_settings().groq_judge_model,
    )
    return FaithfulnessJudgment.model_validate(parsed), llm_result


__all__ = [
    "judge_relevance",
    "judge_faithfulness",
    "LLMCallError",
]
