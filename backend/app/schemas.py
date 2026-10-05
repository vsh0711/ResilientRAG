from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChunkingStrategyOut(BaseModel):
    id: str
    name: str
    how: str
    best_for: str
    selected: bool


class ChunkingResponse(BaseModel):
    strategy_id: str
    strategy_label: str
    chunk_size: int
    chunk_overlap: int
    overlap_ratio: float
    approx_tokens: int
    separators_display: str
    why: str
    strategies: list[ChunkingStrategyOut]


class UploadResponse(BaseModel):
    document_id: str
    num_chunks: int
    num_pages_estimate: int
    indexed: bool = True
    strategy_id: str = "recursive_character"
    strategy_label: str = "Recursive character"
    chunk_size: int = 500
    chunk_overlap: int = 62
    overlap_ratio: float = 0.125
    avg_chunk_chars: int = 0
    rationale: str = ""


class QueryRequest(BaseModel):
    document_id: str
    question: str = Field(min_length=1, max_length=2000)
    max_retries: int = Field(default=3, ge=0, le=3)


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
    sources: list[str] = []
