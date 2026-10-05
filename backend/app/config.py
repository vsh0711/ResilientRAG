"""
Centralized, env-driven configuration.

Everything that previously lived as hardcoded strings/magic numbers in the
original repo (model names, collection names, thresholds) is pulled out
here so the behavior of the self-healing loop can be tuned and tested
without touching business logic.
"""
from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM (Groq) ---
    groq_api_key: str = ""
    groq_primary_model: str = "llama-3.3-70b-versatile"
    groq_fallback_model: str = "llama-3.1-8b-instant"
    groq_judge_model: str = "llama-3.1-8b-instant"
    llm_max_retries: int = 3
    llm_backoff_base_seconds: float = 1.0
    llm_timeout_seconds: float = 30.0

    # --- Embeddings (local, free, no API key required) ---
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_model_name: str = "Qdrant/bm25"
    reranker_model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # --- Vector store ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "resilientrag_chunks"
    qdrant_use_memory: bool = False  # True for tests/local dev w/o Docker

    # --- Cache ---
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    cache_enabled: bool = True

    # --- Self-healing policy ---
    score_pass_threshold: float = 0.8
    default_retrieval_budget: int = 3
    max_retries_default: int = 3
    budget_increment_missing_context: int = 3
    budget_increment_irrelevant_docs: int = 2
    retry_count_trigger_query_rewrite: int = 1  # rewrite query starting this retry #
    retry_count_trigger_hybrid: int = 2         # escalate to hybrid retrieval starting this retry #

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 100

    # --- API ---
    api_cors_origins: list[str] = ["http://localhost:3000"]
    upload_dir: str = "/tmp/resilientrag_uploads"


@lru_cache
def get_settings() -> Settings:
    return Settings()
