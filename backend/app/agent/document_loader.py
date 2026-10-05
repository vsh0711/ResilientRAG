"""
PDF loading + chunking.

Pages are extracted with pypdf, then `plan_chunking` measures the text and
picks the splitter (see chunking.py). The content hash of the resulting
chunks keys caching and persistence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.agent.cache import hash_chunks
from app.agent.chunking import plan_chunking, split_recursive
from app.config import get_settings
from app.exceptions import UnreadableDocumentError


@dataclass
class LoadedDocument:
    chunks: list[str]
    document_hash: str
    source_path: str
    num_pages_estimate: int
    plan: Any = field(default=None)  # ChunkPlan when the strategy was chosen from the file


def split_text(text: str) -> list[str]:
    """Split plain text with the default recursive policy."""
    settings = get_settings()
    return split_recursive(text, settings.chunk_size, settings.chunk_overlap)


def extract_pages(pdf_path: str) -> list[str]:
    """Text of every page, in order."""
    try:
        reader = PdfReader(pdf_path)
        if reader.is_encrypted:
            raise UnreadableDocumentError("This PDF is password-protected. Remove the password and retry.")
        return [(page.extract_text() or "") for page in reader.pages]
    except UnreadableDocumentError:
        raise
    except (PdfReadError, ValueError, OSError, KeyError, TypeError) as exc:
        raise UnreadableDocumentError(
            "Could not read this PDF. The file may be damaged or encrypted."
        ) from exc


def chunk_pages(pages: list[str], strategy: str | None = None, embed: bool = True) -> LoadedDocument:
    """Run the full decision (without the live commentary) and return chunks."""
    embed_fn = None
    if embed:
        from app.agent.vectorstore import embed_texts

        embed_fn = embed_texts
    holder: dict = {}
    for _ in plan_chunking(pages, embed_fn=embed_fn, force=strategy, holder=holder):
        pass
    plan = holder.get("plan")
    if plan is None:
        raise UnreadableDocumentError(
            "This PDF has no extractable text. Scanned pages need OCR, which this app does not run."
        )
    return LoadedDocument(
        chunks=plan.chunks,
        document_hash=hash_chunks(plan.chunks),
        source_path="<pages>",
        num_pages_estimate=len(pages),
        plan=plan,
    )


def load_document(pdf_path: str, strategy: str | None = None) -> LoadedDocument:
    """Load a PDF and split it with the strategy chosen from its content."""
    loaded = chunk_pages(extract_pages(pdf_path), strategy)
    loaded.source_path = pdf_path
    return loaded


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
