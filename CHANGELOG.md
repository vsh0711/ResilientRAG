# Changelog

## 1.0.0

Rebuild of [gurezende/SelfHealingRAG](https://github.com/gurezende/SelfHealingRAG) into a deployable service.

### Added
- Adaptive chunking: the file is measured, a strategy is scored and chosen, and the reasoning streams to the UI (five strategies built, two described).
- Notebook-style UI with self-drawing doodles (bring your own via `frontend/public/doodles/`), streamed upload reasoning and streamed question/healing trace.
- `/documents/stream` and `/query/stream` (server-sent events).
- Split relevance/faithfulness judges; four-tier retrieval ladder with hybrid search; query rewriting.
- Access codes, per-session rate limits, in-process limiter when Redis is absent.
- Docker production overlay with Caddy HTTPS; free-demo packaging for Vercel + a Hugging Face Gradio-SDK Space (no Docker).
- Real-PDF benchmark (`eval/`), load-test scenario (`loadtest/`), and `docs/`.

### Added (later)
- Documents expire when their last browser tab reloads or closes, or after 2 idle hours (opt-in, on in the free Space).
- Re-uploading the same file is instant, with or without Redis; the "Already read" list was removed.

### Changed
- Default chunk size 1,600 → 500 characters, from measurement.
- Default models replaced (the previous ones were retired).
- Same-file uploads are recognised by hash and not re-analysed.

### Fixed
- Backend crashed on a plain `API_CORS_ORIGINS` value.
- A failed judge no longer triggers healing.
- Unbounded upload concurrency; blocking readiness probe; thread-pool starvation.
- Abandoned requests kept spending LLM quota; a spent daily quota is now detected.

### Known gaps
See `docs/ARCHITECTURE.md` §8 and `eval/results/RESULTS.md` ("Not measured").
