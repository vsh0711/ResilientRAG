"""Entry point for a Hugging Face Space that uses the Gradio SDK (no Docker).

The Gradio SDK just runs this file and expects something listening on port
7860. We start the FastAPI app directly with uvicorn; Gradio is used only to
satisfy ZeroGPU's startup check (see below). Defaults below
make it a memory-only demo: vectors in this process, no Redis, one worker.
Every value can be overridden with a Space variable or secret.
"""
try:
    import spaces  # noqa: F401  (must be imported before anything else on ZeroGPU)
except ImportError:
    pass
import os

for key, value in {
    "QDRANT_USE_MEMORY": "true",
    "CACHE_ENABLED": "false",
    "UPLOAD_DIR": "/tmp/resilientrag_uploads",
    "MAX_CONCURRENT_QUERIES": "4",
    "MAX_CONCURRENT_UPLOADS": "2",
    "TRUST_PROXY": "true",
    # one worker, so the in-process bookkeeping is correct: delete a document when its
    # last tab reloads or closes, or after 2 idle hours
    "DOCUMENT_EXPIRY_ENABLED": "true",
    "WARMUP_MODELS": "true",  # download the three embedding models at start, not on the first upload
}.items():
    os.environ.setdefault(key, value)

import uvicorn  # noqa: E402

from app.main import app  # noqa: E402

# ZeroGPU Spaces only start if a function marked @spaces.GPU has been registered
# through Gradio's launch(). This app never uses a GPU. So: the official pattern
# below with a no-op function, launched on a private port nobody connects to,
# while uvicorn serves the real API on the public port. On CPU hardware the
# `spaces` package is absent and none of this runs.
try:
    import spaces  # type: ignore[import-not-found]
    import gradio as gr

    @spaces.GPU
    def predict(text: str = "") -> str:
        # Your model code here (there is none: the API runs on CPU)
        return "ok"

    demo = gr.Interface(fn=predict, inputs=gr.Textbox(), outputs=gr.Textbox(),
                        title="ResilientRAG API", description="This Space serves an API. See /docs.")
    import socket

    with socket.socket() as probe:  # any free port; fixed ones can be taken on the Space
        probe.bind(("127.0.0.1", 0))
        private_port = probe.getsockname()[1]
    try:
        demo.launch(server_name="127.0.0.1", server_port=private_port, prevent_thread_lock=True, quiet=True)
    except OSError as exc:
        # ZeroGPU has already been told about the GPU function by this point,
        # and the API must come up regardless of this placeholder page.
        print(f"Placeholder Gradio page not started: {exc}")
except ImportError:
    pass

if __name__ == "__main__":
    # APP_PORT, not PORT: ZeroGPU sets PORT to a port it uses itself.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("APP_PORT", "7860")), workers=1)
