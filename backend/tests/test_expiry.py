"""Documents leave when their last browser tab does; same-file re-uploads are instant."""
from __future__ import annotations

import io
import json
import os

import pytest
from fastapi.testclient import TestClient

from app.agent.registry import DocumentRegistry
from app.main import app
from app.routers import documents as docs

TAB_A = "tab-aaaaaaaa"
TAB_B = "tab-bbbbbbbb"


# ------------------------------------------------------------------ registry

def test_last_owner_leaving_marks_the_document_for_deletion():
    r = DocumentRegistry()
    r.touch("d1", TAB_A)
    assert r.release("d1", TAB_A) is True
    assert r.tracked() == 0


def test_a_shared_document_survives_one_owner_leaving():
    r = DocumentRegistry()
    r.touch("d1", TAB_A)
    r.touch("d1", TAB_B)
    assert r.release("d1", TAB_A) is False
    assert r.release("d1", TAB_B) is True


def test_releasing_something_never_registered_deletes_nothing():
    assert DocumentRegistry().release("ghost", TAB_A) is False


def test_idle_owners_expire_and_active_ones_do_not():
    clock = [0.0]
    r = DocumentRegistry(clock=lambda: clock[0])
    r.touch("old", TAB_A)
    clock[0] = 100
    r.touch("fresh", TAB_B)
    clock[0] = 150
    assert r.expired(ttl_seconds=100) == ["old"]
    assert r.tracked() == 1


def test_touching_again_keeps_a_document_alive():
    clock = [0.0]
    r = DocumentRegistry(clock=lambda: clock[0])
    r.touch("d", TAB_A)
    clock[0] = 90
    r.touch("d", TAB_A)
    clock[0] = 150
    assert r.expired(ttl_seconds=100) == []


# ------------------------------------------------------------- HTTP behaviour

@pytest.fixture
def expiry_on(monkeypatch, tmp_path):
    monkeypatch.setattr(docs.get_settings(), "document_expiry_enabled", True)
    monkeypatch.setattr(docs.get_settings(), "upload_dir", str(tmp_path))
    monkeypatch.setattr(docs, "registry", DocumentRegistry())
    monkeypatch.setattr(docs, "_LOCAL_UPLOADS", {})
    deleted: list[str] = []
    monkeypatch.setattr("app.agent.vectorstore.VectorStore.delete", lambda self, h: deleted.append(h))
    return deleted


def _seed(document_id="abcdef0123456789"):
    docs.save_chunks(document_id, ["some text"])
    return document_id


def test_release_by_the_only_tab_deletes_everywhere(expiry_on, tmp_path):
    doc = _seed()
    docs.registry.touch(doc, TAB_A)
    client = TestClient(app)
    r = client.post(f"/documents/{doc}/release", headers={"X-Tab-Id": TAB_A})
    assert r.json() == {"released": True, "deleted": True}
    assert expiry_on == [doc]
    assert not os.path.exists(docs._chunks_path(doc))


def test_release_by_one_of_two_tabs_keeps_the_document(expiry_on):
    doc = _seed()
    docs.registry.touch(doc, TAB_A)
    docs.registry.touch(doc, TAB_B)
    client = TestClient(app)
    assert client.post(f"/documents/{doc}/release", headers={"X-Tab-Id": TAB_A}).json()["deleted"] is False
    assert expiry_on == []
    assert os.path.exists(docs._chunks_path(doc))


def test_expiry_is_off_by_default_and_release_does_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(docs.get_settings(), "document_expiry_enabled", False)
    monkeypatch.setattr(docs.get_settings(), "upload_dir", str(tmp_path))
    doc = _seed()
    r = TestClient(app).post(f"/documents/{doc}/release", headers={"X-Tab-Id": TAB_A})
    assert r.json() == {"released": False, "deleted": False}
    assert os.path.exists(docs._chunks_path(doc))


def test_release_without_a_valid_tab_id_deletes_nothing(expiry_on):
    doc = _seed()
    docs.registry.touch(doc, TAB_A)
    client = TestClient(app)
    assert client.post(f"/documents/{doc}/release").json()["deleted"] is False
    assert client.post(f"/documents/{doc}/release", headers={"X-Tab-Id": "x"}).json()["deleted"] is False
    assert expiry_on == []


def test_release_rejects_path_like_ids(expiry_on):
    assert TestClient(app).post("/documents/..%2Fetc/release").status_code in (404, 422)


def test_the_sweep_deletes_documents_nobody_has_touched(expiry_on, monkeypatch):
    import asyncio

    doc = _seed()
    clock = [0.0]
    docs.registry = DocumentRegistry(clock=lambda: clock[0])
    monkeypatch.setattr(docs, "registry", docs.registry)
    docs.registry.touch(doc, TAB_A)
    clock[0] = docs.get_settings().document_ttl_minutes * 60 + 1
    assert asyncio.run(docs.sweep_expired()) == 1
    assert expiry_on == [doc]


def test_deleting_forgets_the_same_file_shortcut(expiry_on):
    doc = _seed()
    docs._LOCAL_UPLOADS["sha1"] = {"events": [], "upload": {"document_id": doc}}
    docs.delete_document(doc)
    assert docs._LOCAL_UPLOADS == {}


# ------------------------------------------------------- same file, no Redis

def _sse(text):
    return [json.loads(f[5:].strip()) for f in text.split("\n\n") if f.startswith("data:")]


def test_same_file_twice_is_instant_even_without_redis(monkeypatch, tmp_path):
    from tests.test_adaptive_chunking import structured_pages

    calls = {"n": 0}

    def extract(_p):
        calls["n"] += 1
        return structured_pages()

    monkeypatch.setattr(docs, "extract_pages", extract)
    monkeypatch.setattr(docs, "index_document", lambda h, c: None)
    monkeypatch.setattr(docs, "_embed_fn", lambda: None)
    monkeypatch.setattr(docs, "_narrate", lambda f: None)
    monkeypatch.setattr(docs, "_LOCAL_UPLOADS", {})
    monkeypatch.setattr(docs.get_settings(), "upload_dir", str(tmp_path))
    client = TestClient(app)

    def up():
        return client.post("/documents/stream",
                           files={"file": ("a.pdf", io.BytesIO(b"%PDF-1.4 same bytes"), "application/pdf")})

    first = _sse(up().text)
    second = _sse(up().text)
    assert calls["n"] == 1
    assert "reused" in [e["type"] for e in second]
    assert "reused" not in [e["type"] for e in first]
    assert second[-1]["upload"]["document_id"] == first[-1]["upload"]["document_id"]
