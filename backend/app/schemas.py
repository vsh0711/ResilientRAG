from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class UploadResponse(BaseModel):
    document_id: str
    num_chunks: int
    num_pages_estimate: int


class QueryRequest(BaseModel):
    document_id: str
    question: str
    max_retries: int = 3


class HealingStepOut(BaseModel):
    retry_number: int
    failure_reason: str
    action_taken: str
    previous_retrieval_mode: str
    new_retrieval_mode: str
    previous_budget: int
    new_budget: int
    query_rewritten: bool
    rewritten_query: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    final_score: float
    relevance_score: float
    faithfulness_score: float
    failure_reason: str
    retry_count: int
    retrieval_mode: str
    healing_trace: list[HealingStepOut]
    latency_ms: dict
    token_usage: dict
    cache_hits: dict
