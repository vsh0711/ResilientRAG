from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import documents, query

logging.basicConfig(level=logging.INFO)

settings = get_settings()

app = FastAPI(
    title="ResilientRAG API",
    description=(
        "Self-healing Retrieval-Augmented Generation agent. Detects retrieval "
        "and faithfulness failures via split LLM judges and automatically "
        "escalates retrieval strategy (dense -> reranked -> hybrid -> hybrid+"
        "reranked), rewrites queries, and retries until quality improves or "
        "the retry budget is exhausted."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(query.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
