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
    demo.launch(server_name="127.0.0.1", server_port=7861, prevent_thread_lock=True,
                show_api=False, quiet=True)
except ImportError:
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")), workers=1)
