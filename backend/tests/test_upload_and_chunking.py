"""Upload safety, chunk persistence, and the measured chunking policy."""
from __future__ import annotations

import io
import time

from fastapi.testclient import TestClient

from app.agent.chunking import describe_chunking
from app.agent.document_loader import LoadedDocument, split_text
from app.main import app
from app.routers import documents as documents_router


def test_overlap_sits_in_the_ten_to_fifteen_percent_band():
    policy = describe_chunking()
    assert 0.10 <= policy["overlap_ratio"] <= 0.15
    # 500 characters won the size sweep in eval/results/bench_chunking_sweep.md
    assert 100 <= policy["approx_tokens"] <= 500


def test_recursive_split_keeps_paragraphs_apart_where_a_fixed_window_does_not():
    """Same width. The fixed window swallows the paragraph break; recursive does not."""
    paragraph_a = "Retrieval precision depends on chunk boundaries. " * 20
    paragraph_b = "Faithfulness means the answer stays inside the source. " * 20
    text = paragraph_a + "\n\n" + paragraph_b

    recursive = split_text(text)
    policy = describe_chunking()
    assert len(recursive) >= 2
    assert recursive[0].startswith("Retrieval")
    assert any(chunk.startswith("Faithfulness") for chunk in recursive)
    # Overlap may borrow a sentence across the break. It must not swallow the next paragraph.
    spilled = "Faithfulness means the answer stays inside the source. " * 8
    assert spilled not in recursive[0]
    assert len(recursive[0]) <= policy["chunk_size"]

    fixed_window = text[:1600]
    assert "Retrieval precision" in fixed_window
    assert "Faithfulness means" in fixed_window


def test_chunking_endpoint_marks_recursive_selected():
    client = TestClient(app)
    response = client.get("/documents/chunking")
    assert response.status_code == 200
    body = response.json()
    selected = [row for row in body["strategies"] if row["selected"]]
    assert [row["id"] for row in selected] == ["recursive_character"]
    assert body["chunk_size"] == 500
    assert body["chunk_overlap"] == 62


def test_upload_indexes_before_it_returns(monkeypatch, tmp_path):
    indexed: list[tuple[str, list[str]]] = []

    def fake_load(_path: str) -> LoadedDocument:
        return LoadedDocument(
            chunks=["alpha passage", "beta passage"],
            document_hash="abc123def4567890",
            source_path=_path,
            num_pages_estimate=2,
        )

    def fake_index(document_hash: str, chunks: list[str]) -> None:
        indexed.append((document_hash, chunks))

    monkeypatch.setattr(documents_router, "load_document", fake_load)
    monkeypatch.setattr(documents_router, "index_document", fake_index)
    monkeypatch.setattr(documents_router.get_settings(), "upload_dir", str(tmp_path))

    client = TestClient(app)
    response = client.post(
        "/documents",
        files={"file": ("notes.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert indexed == [("abc123def4567890", ["alpha passage", "beta passage"])]
    assert body["indexed"] is True
    assert body["document_id"] == "abc123def4567890"
    assert body["num_chunks"] == 2
    assert body["avg_chunk_chars"] == round((len("alpha passage") + len("beta passage")) / 2)
    assert body["strategy_id"] == "recursive_character"
    saved = documents_router.load_chunks("abc123def4567890")
    assert saved == ["alpha passage", "beta passage"]


def test_path_like_filename_never_escapes_the_upload_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(documents_router.get_settings(), "upload_dir", str(tmp_path))
    client = TestClient(app)
    response = client.post(
        "/documents",
        files={"file": ("../../etc/passwd.pdf", io.BytesIO(b"%PDF-not-a-real-pdf"), "application/pdf")},
    )
    assert response.status_code == 400
    assert "passwd" not in response.text
    assert list(tmp_path.iterdir()) == []


def test_non_pdf_bytes_are_rejected_quickly(monkeypatch, tmp_path):
    monkeypatch.setattr(documents_router.get_settings(), "upload_dir", str(tmp_path))
    client = TestClient(app)
    started = time.perf_counter()
    response = client.post(
        "/documents",
        files={"file": ("notes.pdf", io.BytesIO(b"not a pdf at all"), "application/pdf")},
    )
    elapsed = time.perf_counter() - started
    assert response.status_code == 400
    assert elapsed < 2.0
    assert "not a PDF" in response.json()["detail"]


def test_missing_document_is_503_when_qdrant_cannot_be_reached(monkeypatch):
    def boom(_document_hash: str):
        from app.exceptions import RetrievalBackendError

        raise RetrievalBackendError("qdrant unreachable while loading document: connection refused")

    monkeypatch.setattr(documents_router, "_chunks_from_qdrant", boom)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/query", json={"document_id": "doesnotexist", "question": "hi"})
    assert response.status_code == 503
    assert "Qdrant" in response.json()["detail"]
    assert "connection refused" not in response.text
    assert "RetrievalBackendError" not in response.text
