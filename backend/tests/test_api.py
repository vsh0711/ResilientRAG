"""
API-layer tests using FastAPI's TestClient. The /query endpoint is tested
against the real graph wiring but with document chunks pre-seeded on disk
and the graph's LLM/store swapped for fakes via dependency override of
`get_graph`, so no network calls happen.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.routers import documents as documents_router
from app.routers import query as query_router


def test_health_check():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_unknown_document_returns_404():
    client = TestClient(app)
    response = client.post("/query", json={"document_id": "nonexistent", "question": "hi"})
    assert response.status_code == 404


def test_query_runs_graph_end_to_end(monkeypatch, tmp_path, fake_store, fake_llm):
    from app.agent.graph import build_graph
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))

    documents_router.save_chunks("testdoc", ["cat chunk", "dog chunk"])

    fake_llm._judge_sequence = [{
        "relevant_docs": True, "sufficient_context": True, "relevance_score": 0.9,
        "is_faithful": True, "faithfulness_score": 0.9,
    }]
    test_graph = build_graph(store=fake_store, llm=fake_llm)
    monkeypatch.setattr(query_router, "_graph", test_graph)

    client = TestClient(app)
    response = client.post("/query", json={"document_id": "testdoc", "question": "tell me about cats"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == fake_llm.answer
    assert body["retry_count"] == 0
    assert body["failure_reason"] == "none"
