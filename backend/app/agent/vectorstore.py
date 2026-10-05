"""
Persistent, hybrid-capable vector store on top of Qdrant.

Fixes vs. the original repo:

1. **Persistence.** The original used `QdrantClient(":memory:")` and
   re-created + re-embedded the collection inside `retrieve_node` on
   *every single call*, including every retry. Here, collections are
   keyed by a content hash of the chunk list and only (re)indexed once;
   subsequent calls (including retries) reuse the existing collection.

2. **Hybrid retrieval wired in for real.** The original shipped a
   `retrieve_docs_hybrid.py` with dense + BM25 + ColBERT embedding code
   that was never called from the graph. Here, dense and BM25(sparse)
   retrieval are fused with Reciprocal Rank Fusion (RRF) and exposed as
   a genuine `HYBRID` retrieval mode the self-healing loop can switch
   into — not dead code. (ColBERT late-interaction re-ranking is noted
   as a documented future extension rather than implemented: the model
   is large/slow to pull and adds a third retrieval path's worth of
   complexity for marginal gain over dense+BM25+cross-encoder rerank at
   this project's scale — see README trade-offs.)
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

from app.config import get_settings

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"


@lru_cache
def _dense_model() -> TextEmbedding:
    return TextEmbedding(model_name=get_settings().dense_model_name)


@lru_cache
def _sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=get_settings().sparse_model_name)


@lru_cache
def _reranker_model() -> TextCrossEncoder:
    return TextCrossEncoder(model_name=get_settings().reranker_model_name)


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> list[str]:
    """
    Fuse multiple ranked lists of document texts into one ranking using
    RRF: score(d) = sum(1 / (k + rank_in_list)) across lists the doc
    appears in. Standard, parameter-light hybrid fusion technique.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return [doc for doc, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


class VectorStore:
    def __init__(self, client: Optional[QdrantClient] = None):
        settings = get_settings()
        self._settings = settings
        if client is not None:
            self._client = client
        elif settings.qdrant_use_memory:
            self._client = QdrantClient(":memory:")
        else:
            self._client = QdrantClient(url=settings.qdrant_url)

    def collection_name(self, document_hash: str) -> str:
        return f"{self._settings.qdrant_collection}_{document_hash}"

    def is_indexed(self, document_hash: str) -> bool:
        return self._client.collection_exists(self.collection_name(document_hash))

    def index_chunks(self, document_hash: str, chunks: list[str]) -> None:
        """Embed and upsert chunks, but only if not already indexed. This
        is the fix for the "re-embed the whole doc on every retry" bug."""
        collection = self.collection_name(document_hash)
        if self._client.collection_exists(collection):
            logger.info("Collection %s already indexed, skipping re-embed", collection)
            return

        dense_embeddings = list(_dense_model().embed(chunks))
        sparse_embeddings = list(_sparse_model().embed(chunks))

        self._client.create_collection(
            collection_name=collection,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=len(dense_embeddings[0]),
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
        )

        points = [
            models.PointStruct(
                id=idx,
                payload={"text": chunk},
                vector={
                    DENSE_VECTOR_NAME: dense_vec,
                    SPARSE_VECTOR_NAME: sparse_vec.as_object(),
                },
            )
            for idx, (chunk, dense_vec, sparse_vec) in enumerate(
                zip(chunks, dense_embeddings, sparse_embeddings)
            )
        ]
        self._client.upsert(collection_name=collection, points=points)
        logger.info("Indexed %d chunks into %s", len(points), collection)

    def search_dense(self, document_hash: str, query: str, k: int) -> list[str]:
        collection = self.collection_name(document_hash)
        query_vec = list(_dense_model().query_embed(query))[0]
        hits = self._client.query_points(
            collection_name=collection,
            using=DENSE_VECTOR_NAME,
            query=query_vec,
            with_payload=True,
            limit=k,
        )
        return [h.payload["text"] for h in hits.points]

    def search_hybrid(self, document_hash: str, query: str, k: int) -> list[str]:
        """Dense + BM25 sparse retrieval fused with RRF."""
        collection = self.collection_name(document_hash)
        dense_query_vec = list(_dense_model().query_embed(query))[0]
        sparse_query_vec = list(_sparse_model().query_embed(query))[0]

        dense_hits = self._client.query_points(
            collection_name=collection,
            using=DENSE_VECTOR_NAME,
            query=dense_query_vec,
            with_payload=True,
            limit=k,
        )
        sparse_hits = self._client.query_points(
            collection_name=collection,
            using=SPARSE_VECTOR_NAME,
            query=models.SparseVector(**sparse_query_vec.as_object()),
            with_payload=True,
            limit=k,
        )

        dense_texts = [h.payload["text"] for h in dense_hits.points]
        sparse_texts = [h.payload["text"] for h in sparse_hits.points]
        fused = reciprocal_rank_fusion([dense_texts, sparse_texts])
        return fused[:k]

    def rerank(self, query: str, docs: list[str]) -> list[str]:
        if not docs:
            return docs
        scores = list(_reranker_model().rerank(query, docs))
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked]
