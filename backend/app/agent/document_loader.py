"""
PDF loading + chunking.

Same approach as the original repo (LangChain's PyPDFLoader +
RecursiveCharacterTextSplitter), pulled out into config-driven chunk
size/overlap instead of hardcoded values, and returning a content hash
alongside the chunks so callers can key caching/persistence off it.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.agent.cache import hash_chunks
from app.config import get_settings


@dataclass
class LoadedDocument:
    chunks: list[str]
    document_hash: str
    source_path: str
    num_pages_estimate: int


def load_document(pdf_path: str) -> LoadedDocument:
    """Load a PDF and split it into chunks for retrieval."""
    settings = get_settings()

    loader = PyPDFLoader(pdf_path, mode="single")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    split_docs = splitter.split_documents(docs)
    chunks = [d.page_content for d in split_docs]

    return LoadedDocument(
        chunks=chunks,
        document_hash=hash_chunks(chunks),
        source_path=pdf_path,
        num_pages_estimate=len(docs),
    )


def load_text_chunks(chunks: list[str]) -> LoadedDocument:
    """Build a LoadedDocument directly from pre-split text (used by tests
    and the eval harness, which work against plain-text fixtures instead
    of real PDFs)."""
    return LoadedDocument(
        chunks=chunks,
        document_hash=hash_chunks(chunks),
        source_path="<in-memory>",
        num_pages_estimate=1,
    )
