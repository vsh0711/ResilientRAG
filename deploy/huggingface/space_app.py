"""Entry point for a Hugging Face Space that uses the Gradio SDK (no Docker).

The Gradio SDK just runs this file and expects something listening on port
7860. We never import Gradio; we start the FastAPI app directly. Defaults below
make it a memory-only demo: vectors in this process, no Redis, one worker.
Every value can be overridden with a Space variable or secret.
"""
import os

for key, value in {
    "QDRANT_USE_MEMORY": "true",
    "CACHE_ENABLED": "false",
    "UPLOAD_DIR": "/tmp/resilientrag_uploads",
    "MAX_CONCURRENT_QUERIES": "4",
    "MAX_CONCURRENT_UPLOADS": "2",
    "TRUST_PROXY": "true",
    "WARMUP_MODELS": "true",  # download the three embedding models at start, not on the first upload
}.items():
    os.environ.setdefault(key, value)

# ZeroGPU Spaces refuse to start unless some function is marked @spaces.GPU.
# This app never uses a GPU; the no-op below only satisfies that startup check.
# On CPU hardware the `spaces` package is absent and this is skipped.
try:
    import spaces  # type: ignore[import-not-found]

    @spaces.GPU
    def _satisfy_zerogpu_check() -> None:
        return None
except ImportError:
    pass

import uvicorn  # noqa: E402

from app.main import app  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")), workers=1)
