from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from app.agent.cache import SafeRedisCache, hash_chunks
from app.agent.chunking import describe_chunking, plan_chunking
from app.agent.document_loader import extract_pages, load_document
from app.config import get_settings
from app.exceptions import (
    PayloadTooLargeError,
    RetrievalBackendError,
    UnreadableDocumentError,
    retrieval_error_detail,
)
from app.schemas import ChunkingResponse, UploadResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

_DOC_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
_redis: SafeRedisCache | None = None
_upload_slots: asyncio.Semaphore | None = None


def _slots() -> asyncio.Semaphore:
    global _upload_slots
    if _upload_slots is None:
        _upload_slots = asyncio.Semaphore(get_settings().max_concurrent_uploads)
    return _upload_slots


async def _acquire_upload_slot() -> None:
    """Wait briefly for a slot, then say so. A fast 429 beats a 70 s 503."""
    try:
        await asyncio.wait_for(_slots().acquire(), timeout=get_settings().upload_queue_timeout_seconds)
    except TimeoutError:
        raise HTTPException(
            status_code=429,
            detail="Many files are being read right now. Try again in a few seconds.",
            headers={"Retry-After": "15"},
        )


def _file_key(sha: str) -> str:
    return f"upload:{sha}"


def _known_upload(sha: str) -> dict | None:
    """The result of reading this exact file before, if that document still exists."""
    record = _chunk_cache().get_json(_file_key(sha))
    if not isinstance(record, dict) or "upload" not in record:
        return None
    try:
        load_chunks(record["upload"]["document_id"])
    except Exception:
        return None
    return record


def _remember_upload(sha: str, events: list[dict], body: UploadResponse) -> None:
    try:
        _chunk_cache().set_persistent(_file_key(sha), {"events": events, "upload": body.model_dump()})
    except Exception:
        logger.warning("Could not remember upload %s", sha, exc_info=True)


def _chunk_cache() -> SafeRedisCache:
    global _redis
    if _redis is None:
        _redis = SafeRedisCache()
    return _redis


def _chunks_path(document_hash: str) -> str:
    settings = get_settings()
    return os.path.join(settings.upload_dir, f"{document_hash}.json")


def _require_document_id(document_hash: str) -> str:
    if not _DOC_ID.fullmatch(document_hash):
        raise HTTPException(status_code=404, detail=f"Unknown document_id: {document_hash}")
    return document_hash


def save_chunks(document_hash: str, chunks: list[str]) -> None:
    """Persist chunk text in Redis (shared) and on local disk (fallback).

    Qdrant holds the same text inside each point payload. Redis is the
    fast lookup; the disk file keeps single-process dev working when
    Redis is down; Qdrant is the copy that survives both.
    """
    _require_document_id(document_hash)
    _chunk_cache().set_persistent(f"docchunks:{document_hash}", chunks)
    settings = get_settings()
    os.makedirs(settings.upload_dir, exist_ok=True)
    with open(_chunks_path(document_hash), "w") as f:
        json.dump(chunks, f)


def _chunks_from_disk(document_hash: str) -> list[str] | None:
    path = _chunks_path(document_hash)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list) and data:
        return data
    return None


def _chunks_from_redis(document_hash: str) -> list[str] | None:
    data = _chunk_cache().get_json(f"docchunks:{document_hash}")
    if isinstance(data, list) and data:
        return data
    return None


def _chunks_from_qdrant(document_hash: str) -> list[str] | None:
    from app.agent.vectorstore import VectorStore

    try:
        store = VectorStore()
        if not store.is_indexed(document_hash):
            return None
        chunks = store.fetch_chunks(document_hash)
    except Exception as exc:
        raise RetrievalBackendError(
            f"qdrant unreachable while loading document: {exc}"
        ) from exc
    return chunks or None


def load_chunks(document_hash: str) -> list[str]:
    document_hash = _require_document_id(document_hash)
    chunks = _chunks_from_redis(document_hash) or _chunks_from_disk(document_hash)
    if chunks:
        return chunks
    chunks = _chunks_from_qdrant(document_hash)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"Unknown document_id: {document_hash}")
    try:
        save_chunks(document_hash, chunks)
    except Exception:
        logger.warning("Could not backfill chunk text for %s", document_hash, exc_info=True)
    return chunks


def index_document(document_hash: str, chunks: list[str]) -> None:
    from app.agent.vectorstore import VectorStore

    VectorStore().index_chunks(document_hash, chunks)


def _upload_stats(chunks: list[str]) -> UploadResponse:
    policy = describe_chunking()
    avg = round(sum(len(chunk) for chunk in chunks) / len(chunks))
    return UploadResponse(
        document_id="",
        num_chunks=len(chunks),
        num_pages_estimate=0,
        indexed=True,
        strategy_id=policy["strategy_id"],
        strategy_label=policy["strategy_label"],
        chunk_size=policy["chunk_size"],
        chunk_overlap=policy["chunk_overlap"],
        overlap_ratio=policy["overlap_ratio"],
        avg_chunk_chars=avg,
    )


@router.get("/chunking", response_model=ChunkingResponse)
async def chunking_policy() -> ChunkingResponse:
    return ChunkingResponse(**describe_chunking())


async def _receive_pdf(file: UploadFile) -> tuple[str, str]:
    """Stream the upload to a temp file inside upload_dir, enforcing type and size."""
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    os.makedirs(settings.upload_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(suffix=".pdf", dir=settings.upload_dir)
    os.close(fd)
    try:
        written = 0
        digest = hashlib.sha256()
        with open(temp_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                digest.update(chunk)
                written += len(chunk)
                if written > max_bytes:
                    raise PayloadTooLargeError(
                        f"File exceeds the {settings.max_upload_size_mb}MB upload limit."
                    )
                f.write(chunk)
        with open(temp_path, "rb") as f:
            magic = f.read(5)
        if not magic.startswith(b"%PDF"):
            raise UnreadableDocumentError("The file is not a PDF.")
    except BaseException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    return temp_path, digest.hexdigest()


def _narrate(facts: str) -> str | None:
    """Two or three sentences, in the agent's voice, from the measured facts.

    The strategy is already decided by measurement. This only explains it,
    so a Groq outage costs the prose and never the decision.
    """
    if not get_settings().groq_api_key:
        return None
    from app.agent.llm import ResilientLLMClient

    try:
        result = ResilientLLMClient().chat_text(
            system_prompt=(
                "You are a RAG agent explaining, in first person, why you chose a chunking "
                "strategy for the file you just read. Two or three plain sentences. Quote "
                "specific numbers from the facts. Mention the runner-up and why it lost. "
                "Never claim a strategy that is marked not built. No hype, no markdown."
            ),
            user_prompt=facts,
            model=get_settings().groq_judge_model,
            temperature=0.3,
        )
        return result.content.strip() or None
    except Exception:
        logger.warning("Chunking narration failed; using the templated rationale", exc_info=True)
        return None


def _embed_fn():
    from app.agent.vectorstore import embed_texts

    return embed_texts


def _finish(plan, document_hash: str, pages: int) -> UploadResponse:
    body = _upload_stats(plan.chunks)
    return body.model_copy(
        update={
            "document_id": document_hash,
            "num_pages_estimate": pages,
            "strategy_id": plan.strategy_id,
            "strategy_label": plan.strategy_label,
            "rationale": plan.rationale,
        }
    )


@router.post("", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    temp_path, sha = await _receive_pdf(file)
    known = _known_upload(sha)
    if known:
        os.remove(temp_path)
        return UploadResponse(**known["upload"])
    try:
        await _acquire_upload_slot()
    except BaseException:
        os.remove(temp_path)
        raise
    try:
        try:
            loaded = await asyncio.to_thread(load_document, temp_path)
        except UnreadableDocumentError:
            raise
        except Exception as exc:
            logger.error("PDF parse failed: %s", exc, exc_info=True)
            raise UnreadableDocumentError(
                "Could not read this PDF. The file may be damaged or encrypted."
            ) from exc

        try:
            await asyncio.to_thread(index_document, loaded.document_hash, loaded.chunks)
        except Exception as exc:
            logger.error("Indexing failed: %s", exc, exc_info=True)
            raise RetrievalBackendError(f"index failed: {exc}") from exc

        save_chunks(loaded.document_hash, loaded.chunks)
        plan = loaded.plan
        body = _upload_stats(loaded.chunks)
        update = {"document_id": loaded.document_hash, "num_pages_estimate": loaded.num_pages_estimate}
        if plan is not None:
            update.update(strategy_id=plan.strategy_id, strategy_label=plan.strategy_label,
                          rationale=plan.rationale)
        result = body.model_copy(update=update)
        _remember_upload(sha, [], result)
        return result
    finally:
        _slots().release()
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _sse(event: dict) -> bytes:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()


@router.post("/stream", response_model=None)
async def upload_document_stream(file: UploadFile = File(...)) -> StreamingResponse:
    """Same as POST /documents, but narrates every decision as server-sent events."""
    temp_path, sha = await _receive_pdf(file)
    known = _known_upload(sha)
    if not known:
        try:
            await _acquire_upload_slot()
        except BaseException:
            os.remove(temp_path)
            raise

    async def events():
        slot_held = not known
        try:
            yield _sse({"type": "stage", "stage": "upload", "text": f"Received {file.filename}."})
            if known:
                yield _sse({"type": "think", "text": "I have read this exact file before, so I am reusing that analysis."})
                for event in known["events"]:
                    if event["type"] not in ("stage", "done"):
                        yield _sse(event)
                yield _sse({"type": "done", "upload": known["upload"]})
                return
            seen: list[dict] = []
            try:
                pages = await asyncio.to_thread(extract_pages, temp_path)
            except UnreadableDocumentError as exc:
                yield _sse({"type": "error", "detail": str(exc)})
                return

            holder: dict = {}
            gen = plan_chunking(pages, embed_fn=_embed_fn(), narrate=_narrate, holder=holder)
            async for event in iterate_in_threadpool(gen):
                seen.append(event)
                yield _sse(event)
            plan = holder.get("plan")
            if plan is None:
                yield _sse({"type": "error", "detail": "This PDF has no extractable text. Scanned pages need OCR, which this app does not run."})
                return

            document_hash = hash_chunks(plan.chunks)
            yield _sse({"type": "stage", "stage": "embed",
                        "text": f"Embedding {len(plan.chunks)} chunks into Qdrant (dense + BM25)."})
            try:
                await asyncio.to_thread(index_document, document_hash, plan.chunks)
            except Exception as exc:
                logger.error("Indexing failed: %s", exc, exc_info=True)
                yield _sse({"type": "error", "detail": retrieval_error_detail(RetrievalBackendError(str(exc)))})
                return
            save_chunks(document_hash, plan.chunks)
            body = _finish(plan, document_hash, len(pages))
            _remember_upload(sha, [e for e in seen if e["type"] != "stage"], body)
            yield _sse({"type": "done", "upload": body.model_dump()})
        except Exception:
            logger.error("Upload stream failed", exc_info=True)
            yield _sse({"type": "error", "detail": "Something went wrong while reading this file."})
        finally:
            if slot_held:
                _slots().release()
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
