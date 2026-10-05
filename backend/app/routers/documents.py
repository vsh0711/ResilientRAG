from __future__ import annotations

import json
import os

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.agent.document_loader import load_document
from app.config import get_settings
from app.exceptions import PayloadTooLargeError
from app.schemas import UploadResponse

router = APIRouter(prefix="/documents", tags=["documents"])


def _chunks_path(document_hash: str) -> str:
    settings = get_settings()
    return os.path.join(settings.upload_dir, f"{document_hash}.json")


def save_chunks(document_hash: str, chunks: list[str]) -> None:
    settings = get_settings()
    os.makedirs(settings.upload_dir, exist_ok=True)
    with open(_chunks_path(document_hash), "w") as f:
        json.dump(chunks, f)


def load_chunks(document_hash: str) -> list[str]:
    path = _chunks_path(document_hash)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Unknown document_id: {document_hash}")
    with open(path) as f:
        return json.load(f)


@router.post("", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    os.makedirs(settings.upload_dir, exist_ok=True)
    temp_path = os.path.join(settings.upload_dir, f"_upload_{file.filename}")
    try:
        written = 0
        with open(temp_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise PayloadTooLargeError(
                        f"File exceeds the {settings.max_upload_size_mb}MB upload limit."
                    )
                f.write(chunk)

        loaded = load_document(temp_path)
        save_chunks(loaded.document_hash, loaded.chunks)

        return UploadResponse(
            document_id=loaded.document_hash,
            num_chunks=len(loaded.chunks),
            num_pages_estimate=loaded.num_pages_estimate,
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
