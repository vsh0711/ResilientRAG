"""The chunking decision: measured from the file, explained as it happens."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.agent import chunking as ck
from app.main import app
from app.routers import documents as documents_router
from app.routers import query as query_router


def wrap(text: str, width: int = 70) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width))


PROSE = (
    "Retrieval quality depends on how a document is cut before it is embedded, "
    "because a chunk is the unit that gets matched against a question. "
    "A chunk that stops mid-sentence loses the clause that carried the answer. "
)


def prose_pages(n_pages: int = 4) -> list[str]:
    return [wrap(PROSE * 6) + "\n" for _ in range(n_pages)]


def structured_pages() -> list[str]:
    pages = []
    for i in range(1, 7):
        pages.append(
            f"{i}. Section Number {i}\n{wrap(PROSE * 3)}\n\nDetails For Part {i}\n{wrap(PROSE * 2)}\n"
        )
    return pages


def code_pages() -> list[str]:
    block = "def handler(x):\n    return x + 1\n\nclass Thing:\n    def run(self):\n        return 2\n\n"
    return [block * 12]


def one_hot_embed(texts: list[str]) -> np.ndarray:
    """Two topics, two directions. Anything about 'cooking' points one way."""
    out = np.zeros((len(texts), 2))
    for i, t in enumerate(texts):
        out[i, 0 if "soup" in t or "simmer" in t else 1] = 1.0
    return out


# ---------------------------------------------------------------- text handling

def test_headings_are_told_apart_from_prose():
    assert ck.is_heading("3.2 Error Codes")
    assert ck.is_heading("# Setup")
    assert ck.is_heading("INSTALLATION GUIDE")
    assert ck.is_heading("Chapter 4: Zephyr X200")
    assert not ck.is_heading("The pump seals shut when pressure drops below the limit.")
    assert not ck.is_heading("- a bullet point about something")
    assert not ck.is_heading("see below,")


def test_page_margins_do_not_split_a_paragraph():
    pages = ["The pump stops when the sensor reads zero and the valve is\nclosed for more than ten seconds in a row.\n\nNew paragraph here."]
    clean = ck.normalize_text(pages)
    assert "valve is closed for more" in clean
    assert "\n\nNew paragraph here." in clean


def test_hyphenated_line_breaks_are_healed():
    assert "retrieval" in ck.normalize_text(["retrie-\nval is hard"])


def test_code_lines_stay_together():
    clean = ck.normalize_text(code_pages())
    assert "def handler(x):\n    return x + 1" in clean


# --------------------------------------------------------------------- splitters

def test_structure_split_never_crosses_a_heading():
    text = ck.normalize_text(structured_pages())
    chunks = ck.split_structure(text, 1600, 200)
    assert chunks
    for chunk in chunks:
        headings = [line for line in chunk.split("\n") if ck.is_heading(line) and line[0].isdigit()]
        assert len(headings) <= 1, chunk[:200]


def test_structure_split_keeps_the_heading_on_each_part_of_a_long_section():
    long_section = "1. Big Section\n\n" + wrap(PROSE * 30)
    chunks = ck.split_structure(long_section, 600, 80)
    assert len(chunks) > 1
    assert all(c.startswith("Big Section") or c.startswith("1. Big Section") for c in chunks)


def test_fixed_split_is_blind_to_boundaries():
    chunks = ck.split_fixed("a" * 1000, 300, 50)
    assert [len(c) for c in chunks[:3]] == [300, 300, 300]


def test_semantic_split_breaks_where_the_topic_changes():
    cooking = " ".join(f"Stir the soup and simmer it for {i} minutes." for i in range(1, 15))
    cars = " ".join(f"The engine produces {i} horsepower at high revs." for i in range(1, 15))
    chunks = ck.split_semantic(cooking + " " + cars, 5000, 100, one_hot_embed, min_chars=100)
    assert len(chunks) >= 2
    assert not any("soup" in c and "engine" in c for c in chunks)


def test_semantic_falls_back_on_very_short_text():
    assert ck.split_semantic("One. Two.", 1600, 200, one_hot_embed)


def test_size_override_scales_overlap_with_it():
    chunks = ck.apply_strategy("recursive_character", wrap(PROSE * 40, 90), chunk_size=500)
    assert all(len(c) <= 500 for c in chunks)


# ---------------------------------------------------------------------- decision

def decide(pages, shift=None):
    clean = ck.normalize_text(pages)
    return ck.score_strategies(ck.profile_document(pages, clean), shift)[0]


def test_headed_document_picks_structure():
    assert decide(structured_pages()).id == "structure"


def test_code_picks_code_aware():
    assert decide(code_pages()).id == "code"


def test_plain_prose_picks_the_recursive_default():
    assert decide(prose_pages()).id == ck.STRATEGY_ID


def test_headingless_text_that_changes_topic_picks_semantic():
    assert decide(prose_pages(), shift=(0.19, 0.6, 48)).id == "semantic"


def test_uniform_text_with_no_shifts_stays_recursive():
    assert decide(prose_pages(), shift=(0.0, 0.55, 48)).id == ck.STRATEGY_ID


def test_unbuilt_strategies_are_scored_but_never_chosen():
    clean = ck.normalize_text(prose_pages())
    rows = ck.score_strategies(ck.profile_document(prose_pages(), clean), None)
    unbuilt = [r for r in rows if not r.built]
    assert {r.id for r in unbuilt} == {"sentence_window", "parent_child"}
    assert all(r.verdict == "not built" for r in unbuilt)
    assert rows[0].built


def test_exactly_one_strategy_is_chosen():
    clean = ck.normalize_text(structured_pages())
    rows = ck.score_strategies(ck.profile_document(structured_pages(), clean), None)
    assert [r.verdict for r in rows].count("chosen") == 1


# -------------------------------------------------------------------- the stream

def test_plan_emits_reasoning_in_order_and_returns_chunks():
    holder: dict = {}
    events = list(ck.plan_chunking(structured_pages(), holder=holder))
    kinds = [e["type"] for e in events]
    assert kinds.index("profile") < kinds.index("scores") < kinds.index("decision") < kinds.index("chunked")
    plan = holder["plan"]
    assert plan.strategy_id == "structure"
    assert plan.chunks
    chunked = next(e for e in events if e["type"] == "chunked")
    assert chunked["num_chunks"] == len(plan.chunks)
    assert len(chunked["sizes"]) == len(plan.chunks)


def test_every_unbuilt_strategy_is_explained_in_the_commentary():
    text = " ".join(e.get("text", "") for e in ck.plan_chunking(prose_pages(), holder={}))
    assert "Sentence window: not built" in text
    assert "Parent-child chunking: not built" in text


def test_forced_strategy_skips_the_vote_and_is_honoured():
    holder: dict = {}
    events = list(ck.plan_chunking(prose_pages(), force="fixed", holder=holder))
    assert holder["plan"].strategy_id == "fixed"
    assert any("forced" in e.get("text", "") for e in events)


def test_unbuilt_strategy_cannot_be_forced():
    with pytest.raises(ValueError):
        list(ck.plan_chunking(prose_pages(), force="parent_child", holder={}))


def test_blank_pages_produce_no_plan():
    holder: dict = {}
    list(ck.plan_chunking(["   \n  ", ""], holder=holder))
    assert holder["plan"] is None


def test_narration_replaces_the_template_only_when_it_returns_text():
    holder: dict = {}
    list(ck.plan_chunking(prose_pages(), narrate=lambda facts: "I picked this because reasons.", holder=holder))
    assert holder["plan"].rationale == "I picked this because reasons."
    holder = {}
    list(ck.plan_chunking(prose_pages(), narrate=lambda facts: None, holder=holder))
    assert holder["plan"].rationale.startswith(holder["plan"].strategy_label)


# ------------------------------------------------------------------ HTTP: upload

def sse_events(text: str) -> list[dict]:
    return [json.loads(f[5:].strip()) for f in text.split("\n\n") if f.startswith("data:")]


def test_upload_stream_narrates_then_indexes(monkeypatch, tmp_path):
    indexed = []
    monkeypatch.setattr(documents_router, "extract_pages", lambda _p: structured_pages())
    monkeypatch.setattr(documents_router, "index_document", lambda h, c: indexed.append((h, c)))
    monkeypatch.setattr(documents_router, "_embed_fn", lambda: None)
    monkeypatch.setattr(documents_router, "_narrate", lambda facts: None)
    monkeypatch.setattr(documents_router.get_settings(), "upload_dir", str(tmp_path))

    client = TestClient(app)
    response = client.post(
        "/documents/stream",
        files={"file": ("manual.pdf", io.BytesIO(b"%PDF-1.4 x"), "application/pdf")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response.text)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "stage" and kinds[-1] == "done"
    assert kinds.index("decision") < kinds.index("done")
    done = events[-1]["upload"]
    assert done["strategy_id"] == "structure"
    assert indexed and indexed[0][0] == done["document_id"]
    assert list(tmp_path.glob("*.pdf")) == []  # temp upload cleaned up


def test_upload_stream_reports_a_pdf_with_no_text(monkeypatch, tmp_path):
    monkeypatch.setattr(documents_router, "extract_pages", lambda _p: ["", "  "])
    monkeypatch.setattr(documents_router, "_embed_fn", lambda: None)
    monkeypatch.setattr(documents_router.get_settings(), "upload_dir", str(tmp_path))
    client = TestClient(app)
    response = client.post(
        "/documents/stream",
        files={"file": ("scan.pdf", io.BytesIO(b"%PDF-1.4 x"), "application/pdf")},
    )
    events = sse_events(response.text)
    assert events[-1]["type"] == "error"
    assert "OCR" in events[-1]["detail"]


def test_upload_stream_still_rejects_non_pdfs_before_streaming(monkeypatch, tmp_path):
    monkeypatch.setattr(documents_router.get_settings(), "upload_dir", str(tmp_path))
    client = TestClient(app)
    response = client.post(
        "/documents/stream",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400


def test_upload_stream_survives_an_indexing_outage(monkeypatch, tmp_path):
    def boom(h, c):
        raise RuntimeError("connection refused 10.0.0.5:6333")

    monkeypatch.setattr(documents_router, "extract_pages", lambda _p: prose_pages())
    monkeypatch.setattr(documents_router, "index_document", boom)
    monkeypatch.setattr(documents_router, "_embed_fn", lambda: None)
    monkeypatch.setattr(documents_router, "_narrate", lambda facts: None)
    monkeypatch.setattr(documents_router.get_settings(), "upload_dir", str(tmp_path))
    client = TestClient(app)
    response = client.post(
        "/documents/stream",
        files={"file": ("a.pdf", io.BytesIO(b"%PDF-1.4 x"), "application/pdf")},
    )
    last = sse_events(response.text)[-1]
    assert last["type"] == "error"
    assert "10.0.0.5" not in response.text


# ------------------------------------------------------------------ HTTP: query

class FakeGraph:
    """Yields what LangGraph's stream_mode='updates' yields."""

    def __init__(self, fail_first: bool):
        self.fail_first = fail_first

    def stream(self, state, stream_mode="updates"):
        yield {"retrieve": {"retrieved_docs": ["alpha passage"]}}
        yield {"generate": {"answer": "draft one"}}
        if self.fail_first:
            yield {"score": {"relevance_score": 0.2, "faithfulness_score": 0.9, "score": 0.55,
                             "failure_reason": "irrelevant_docs"}}
            step = {"retry_number": 1, "failure_reason": "irrelevant_docs", "action_taken": "escalate",
                    "previous_retrieval_mode": "dense", "new_retrieval_mode": "dense_rerank",
                    "previous_budget": 3, "new_budget": 5, "query_rewritten": False, "rewritten_query": None}
            yield {"retry": {"retrieval_budget": 5, "retrieval_mode": "dense_rerank", "healing_trace": [step]}}
            yield {"increment_retry": {"retry_count": 1}}
            yield {"retrieve": {"retrieved_docs": ["beta passage"]}}
            yield {"generate": {"answer": "draft two"}}
        yield {"score": {"relevance_score": 1.0, "faithfulness_score": 1.0, "score": 1.0,
                         "failure_reason": "none"}}


def test_query_stream_reports_each_node_then_the_result(monkeypatch, tmp_path):
    monkeypatch.setattr(documents_router.get_settings(), "upload_dir", str(tmp_path))
    documents_router.save_chunks("streamdoc", ["alpha passage", "beta passage"])
    monkeypatch.setattr(query_router, "_graph", FakeGraph(fail_first=True))
    client = TestClient(app)
    response = client.post("/query/stream", json={"document_id": "streamdoc", "question": "what is alpha"})
    assert response.status_code == 200
    events = sse_events(response.text)
    kinds = [e["type"] for e in events]
    assert kinds == ["start", "retrieve", "generate", "score", "heal", "retrieve", "generate", "score", "result"]
    assert events[3]["passed"] is False
    assert events[7]["passed"] is True
    assert events[4]["step"]["new_retrieval_mode"] == "dense_rerank"
    final = events[-1]["result"]
    assert final["answer"] == "draft two"
    assert final["sources"] == ["beta passage"]


def test_query_stream_that_passes_first_time_has_no_heal_step(monkeypatch, tmp_path):
    monkeypatch.setattr(documents_router.get_settings(), "upload_dir", str(tmp_path))
    documents_router.save_chunks("streamdoc2", ["alpha passage"])
    monkeypatch.setattr(query_router, "_graph", FakeGraph(fail_first=False))
    client = TestClient(app)
    events = sse_events(client.post("/query/stream", json={"document_id": "streamdoc2", "question": "q"}).text)
    assert "heal" not in [e["type"] for e in events]
    assert events[-1]["type"] == "result"


def test_query_stream_releases_its_slot_even_when_the_graph_explodes(monkeypatch, tmp_path):
    class Boom:
        def stream(self, state, stream_mode="updates"):
            raise RuntimeError("secret internal detail")
            yield  # pragma: no cover

    monkeypatch.setattr(documents_router.get_settings(), "upload_dir", str(tmp_path))
    documents_router.save_chunks("streamdoc3", ["alpha"])
    monkeypatch.setattr(query_router, "_graph", Boom())
    monkeypatch.setattr(query_router, "_slots", None)
    client = TestClient(app)
    response = client.post("/query/stream", json={"document_id": "streamdoc3", "question": "q"})
    events = sse_events(response.text)
    assert events[-1]["type"] == "error"
    assert "secret" not in response.text
    # every slot is free again
    sem = query_router._semaphore()
    assert sem._value == query_router.get_settings().max_concurrent_queries


# ------------------------------------------------------- load protection

class MemoryCache:
    def __init__(self):
        self.data = {}

    def set_persistent(self, key, value):
        self.data[key] = value

    def get_json(self, key):
        return self.data.get(key)


def _stream_upload(client, name="a.pdf"):
    return client.post(
        "/documents/stream",
        files={"file": (name, io.BytesIO(b"%PDF-1.4 identical bytes"), "application/pdf")},
    )


def test_the_same_file_is_not_analysed_twice(monkeypatch, tmp_path):
    calls = {"extract": 0}

    def counting_extract(_p):
        calls["extract"] += 1
        return structured_pages()

    monkeypatch.setattr(documents_router, "extract_pages", counting_extract)
    monkeypatch.setattr(documents_router, "index_document", lambda h, c: None)
    monkeypatch.setattr(documents_router, "_embed_fn", lambda: None)
    monkeypatch.setattr(documents_router, "_narrate", lambda facts: None)
    monkeypatch.setattr(documents_router, "_redis", MemoryCache())
    monkeypatch.setattr(documents_router, "load_chunks", lambda h: ["x"])
    monkeypatch.setattr(documents_router.get_settings(), "upload_dir", str(tmp_path))
    client = TestClient(app)

    first = sse_events(_stream_upload(client).text)
    second = sse_events(_stream_upload(client).text)
    assert calls["extract"] == 1
    assert second[-1]["type"] == "done"
    assert second[-1]["upload"]["document_id"] == first[-1]["upload"]["document_id"]
    assert any("exact file before" in e.get("text", "") for e in second)
    assert any(e["type"] == "decision" for e in second)  # the reasoning is replayed


def test_a_full_house_gets_a_fast_429_not_a_long_wait(monkeypatch, tmp_path):
    import asyncio

    monkeypatch.setattr(documents_router, "_redis", MemoryCache())
    monkeypatch.setattr(documents_router, "_upload_slots", asyncio.Semaphore(0))
    monkeypatch.setattr(documents_router.get_settings(), "upload_queue_timeout_seconds", 0.05)
    monkeypatch.setattr(documents_router.get_settings(), "upload_dir", str(tmp_path))
    client = TestClient(app)
    response = _stream_upload(client)
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "15"
    assert list(tmp_path.glob("*.pdf")) == []  # temp file cleaned up


def test_slots_are_released_after_an_upload_fails(monkeypatch, tmp_path):
    import asyncio

    sem = asyncio.Semaphore(1)
    monkeypatch.setattr(documents_router, "_upload_slots", sem)
    monkeypatch.setattr(documents_router, "_redis", MemoryCache())
    monkeypatch.setattr(documents_router, "extract_pages", lambda _p: ["", " "])
    monkeypatch.setattr(documents_router, "_embed_fn", lambda: None)
    monkeypatch.setattr(documents_router.get_settings(), "upload_dir", str(tmp_path))
    client = TestClient(app)
    _stream_upload(client)
    assert sem._value == 1
