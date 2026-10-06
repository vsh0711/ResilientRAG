"""Demo mode: vectors live in the process, so they must survive between requests."""
from __future__ import annotations

import app.agent.vectorstore as vs
from app.config import get_settings


class FakeEmbedding(list):
    def as_object(self):
        return {"indices": [0], "values": [1.0]}


class FakeDense:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def query_embed(self, q):
        return [[1.0, 0.0]]


class FakeSparse:
    def embed(self, texts):
        return [FakeEmbedding() for _ in texts]

    def query_embed(self, q):
        return [FakeEmbedding()]


def test_a_second_store_instance_sees_what_the_first_indexed(monkeypatch):
    monkeypatch.setattr(get_settings(), "qdrant_use_memory", True)
    monkeypatch.setattr(vs, "_MEMORY_CLIENT", None)
    monkeypatch.setattr(vs, "_dense_model", lambda: FakeDense())
    monkeypatch.setattr(vs, "_sparse_model", lambda: FakeSparse())

    vs.VectorStore().index_chunks("memdoc", ["alpha text", "beta text"])
    other = vs.VectorStore()  # as a later request would create
    assert other.is_indexed("memdoc")
    assert other.fetch_chunks("memdoc") == ["alpha text", "beta text"]


def test_api_key_is_passed_to_a_remote_qdrant(monkeypatch):
    seen = {}

    class Spy:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(get_settings(), "qdrant_use_memory", False)
    monkeypatch.setattr(get_settings(), "qdrant_api_key", "secret-key")
    monkeypatch.setattr(vs, "QdrantClient", Spy)
    vs.VectorStore()
    assert seen["api_key"] == "secret-key"
