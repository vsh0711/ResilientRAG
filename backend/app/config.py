"""
Centralized, env-driven configuration.

Everything that previously lived as hardcoded strings/magic numbers in the
original repo (model names, collection names, thresholds) is pulled out
here so the behavior of the self-healing loop can be tuned and tested
without touching business logic.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# backend/app/config.py -> repository root, so `uvicorn` started from
# backend/ still reads the .env the user edits at the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(_REPO_ROOT / ".env"), ".env"),
        extra="ignore",
        env_file_encoding="utf-8",
    )

    # --- LLM (Groq) ---
    groq_api_key: str = ""
    groq_primary_model: str = "openai/gpt-oss-120b"
    groq_fallback_model: str = "openai/gpt-oss-20b"
    groq_judge_model: str = "qwen/qwen3.8-27b"
    llm_max_retries: int = 5
    llm_backoff_base_seconds: float = 2.0
    llm_timeout_seconds: float = 30.0

    # --- Embeddings (local, free, no API key required) ---
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_model_name: str = "Qdrant/bm25"
    reranker_model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # --- Vector store ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""  # for Qdrant Cloud
    qdrant_collection: str = "resilientrag_chunks"
    # True keeps vectors in this process: no Qdrant server, lost on restart, and only
    # correct with a single worker. Fine for a free demo, wrong for production.
    qdrant_use_memory: bool = False

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

    # --- Chunking ---
    # Measured, not assumed: on the factoid benchmark in eval/ (real PDFs,
    # top-3 retrieval) 500 characters hit 98-100% and the old 1600 hit 80-89%.
    # See eval/results/bench_chunking_sweep.md. Overlap stays at ~12.5%.
    chunk_size: int = 500
    chunk_overlap: int = 62

    # --- API ---
    # NoDecode: accept "a,b" or a bare URL from the environment, not only JSON.
    api_cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000"]
    upload_dir: str = "/tmp/resilientrag_uploads"
    max_upload_size_mb: int = 20
    # Per browser session (X-Session-Id). Offices put many people behind one IP,
    # so the IP ceiling is much higher and only exists to stop session-id spraying.
    rate_limit_per_minute: int = 60
    ip_rate_limit_per_minute: int = 1200
    # Honor X-Forwarded-For only behind a proxy you control.
    trust_proxy: bool = False
    # How long one /query may run, and how long it may wait for a free slot.
    query_timeout_seconds: float = 90.0
    query_queue_timeout_seconds: float = 15.0
    max_concurrent_queries: int = 12
    # Parsing + chunking + embedding is CPU and memory heavy. Unbounded, a burst of
    # uploads thrashes the box (observed: 6 cores pegged, 2.9 GB, every call 503).
    max_concurrent_uploads: int = 2
    # Delete a document when its last browser tab leaves (reload or close), or after
    # it has been idle this long. In-process bookkeeping: single worker only.
    document_expiry_enabled: bool = False
    document_ttl_minutes: int = 120
    upload_queue_timeout_seconds: float = 20.0
    # Qdrant RPCs fail in a few seconds instead of hanging the request.
    qdrant_timeout_seconds: float = 5.0
    # Comma-separated access codes. Empty means the API is open (local dev).
    # When set, every route except /health* and /auth/status needs X-Access-Code.
    access_codes: str = ""
    # Download embedding models at process start (on in Docker, off in tests).
    warmup_models: bool = False

    @field_validator("api_cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: object) -> object:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("["):
                return json.loads(text)
            return [part.strip() for part in text.split(",") if part.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
