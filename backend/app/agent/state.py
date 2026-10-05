"""
Agent state definition.

The original repo used a bare TypedDict with no validation — a judge
returning a malformed payload, or a node writing the wrong type into
`score`, would silently corrupt downstream decisions. Here the state is
still a TypedDict (LangGraph requires dict-like state), but every value
that crosses a trust boundary (LLM output, user input) is validated through
a Pydantic model *before* it's allowed into the state.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional, TypedDict

from pydantic import BaseModel, Field


class RetrievalMode(str, Enum):
    DENSE = "dense"
    DENSE_RERANK = "dense_rerank"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class FailureReason(str, Enum):
    IRRELEVANT_DOCS = "irrelevant_docs"
    MISSING_CONTEXT = "missing_context"
    UNFAITHFUL = "unfaithful_answer"
    JUDGE_UNAVAILABLE = "judge_unavailable"
    NONE = "none"


class RelevanceJudgment(BaseModel):
    """Structured output of the retrieval-relevance judge."""
    relevant_docs: bool
    sufficient_context: bool
    relevance_score: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class FaithfulnessJudgment(BaseModel):
    """
    Structured output of the faithfulness/hallucination judge.

    Split out from relevance on purpose: a system can retrieve perfectly
    relevant documents and still hallucinate an answer that isn't grounded
    in them. Blending both into one score (as the original repo does)
    makes the failure mode undiagnosable — you can't tell whether to fix
    retrieval or fix generation.
    """
    is_faithful: bool
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    unsupported_claims: List[str] = Field(default_factory=list)
    reasoning: str = ""


class HealingStep(BaseModel):
    retry_number: int
    failure_reason: FailureReason
    action_taken: str
    previous_retrieval_mode: RetrievalMode
    new_retrieval_mode: RetrievalMode
    previous_budget: int
    new_budget: int
    query_rewritten: bool = False
    rewritten_query: Optional[str] = None


class RAGState(TypedDict, total=False):
    # Inputs
    document_id: str
    chunks: List[str]
    query: str
    original_query: str

    # Retrieval
    retrieval_mode: str          # RetrievalMode value
    retrieval_budget: int
    retrieved_docs: List[str]

    # Generation
    answer: str

    # Judging (split)
    relevance_score: float
    faithfulness_score: float
    score: float                 # combined, for backwards-compatible gating
    failure_reason: str          # FailureReason value

    # Control
    retry_count: int
    max_retries: int
    healing_trace: List[dict]    # serialized HealingStep entries

    # Observability
    latency_ms: dict             # per-node latency, for the eval harness
    token_usage: dict            # per-call token counts, for cost reporting
    cache_hits: dict              # which steps were served from cache
