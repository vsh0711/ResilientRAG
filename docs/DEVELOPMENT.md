# Development guide

## Layout

```
backend/app/
  main.py            app, middleware wiring, health, exception handlers
  config.py          every setting, from the environment
  middleware.py      request id, rate limiting (Redis + local fallback), access codes
  routers/           documents.py (upload, SSE), query.py (ask, SSE)
  agent/
    chunking.py      profile → score → choose → split (the "thinks out loud" part)
    document_loader.py  pypdf extraction, chunk_pages
    vectorstore.py   Qdrant dense + BM25, RRF, rerank, in-memory mode
    retrieval.py     the four retrieval modes
    graph.py nodes.py state.py   the LangGraph agent and healing policy
    judges.py llm.py query_rewrite.py   judges, Groq client with deadline, rewriting
    cache.py         Redis wrapper that fails open
backend/tests/       135 tests, offline: fake LLM and store, Qdrant in memory
frontend/app/        Next.js UI (see ARCHITECTURE.md §7)
eval/                benchmark generator, runners, results
loadtest/            Locust scenario
deploy/huggingface/  Space image and bundle script
docs/                this folder
```

## Run locally

```bash
docker compose up -d qdrant redis                 # or QDRANT_USE_MEMORY=true CACHE_ENABLED=false
cd backend && uv sync --extra dev
uv run uvicorn app.main:app --reload              # http://localhost:8000/docs
cd ../frontend && npm install && npm run dev      # http://localhost:3000
```

Put `GROQ_API_KEY` in a repo-root `.env` (git-ignored). The first upload downloads three small embedding models.

## Tests

```bash
cd backend && uv run python -m pytest tests -q          # ~3 s, no network, no keys
cd frontend && npx tsc --noEmit                         # type check
```

What the tests pin down (each exists because something broke or could):
- healing loop terminates, heals, never re-embeds across retries (`test_graph_retry_logic.py`)
- a judge outage never triggers healing (`TestJudgeOutage`)
- retries, fallback, `Retry-After`, daily-quota switching, deadlines (`test_llm_resilience.py`, `test_deadline.py`)
- chunking: heading detection, each splitter, the decision for each document shape, event order (`test_adaptive_chunking.py`)
- upload safety, SSE framing, same-file shortcut, slot release, 429s (`test_upload_and_chunking.py`, `test_adaptive_chunking.py`)
- access codes, session rate limits, Redis-less limiter, env parsing, in-memory Qdrant

## Evaluation

All numbers published in this repo come from real runs. There are no mocked results.

```bash
cd backend
uv run --extra dev python ../eval/make_bench.py                 # 4 PDFs + 160 questions with exact keys
uv run --extra dev python ../eval/run_bench.py chunking --tag sweep \
      --configs recursive_character@500 recursive_character semantic auto
uv run --extra dev python ../eval/run_bench.py e2e --per-doc 8 --workers 1   # real Groq calls
uv run --extra dev python ../eval/run_bench.py unanswerable --n 30
```

- Questions have an exact `key`; correctness is a string match, never an LLM opinion.
- `--configs name@size` forces a strategy and chunk size; `auto` lets the agent choose.
- **Mind the quota.** One `e2e` run is ~190 questions × 3 arms and can spend a free key's daily tokens. Use `--workers 1`.
- Load test (needs a running stack): `ACCESS_CODE=… locust -f loadtest/locustfile.py --headless -u 100 -r 10 -t 4m --host https://localhost/api/proxy`. Task weights are overridable with `W_ASK`, `W_POLICY`, `W_UPLOAD`, `W_READY`, `WAIT_MIN`, `WAIT_MAX`.
- `eval/run_retrieval_eval.py` and `run_generation_eval.py` are the older 16-passage harness; their `--dry-run` uses fakes and must never be published.

## Conventions

- Comments say why, not what. Failure behaviour is written down where it is decided.
- Operational failures are typed (`exceptions.py`) and mapped to clean responses in `main.py`; never leak a stack trace.
- Anything that blocks must not run on the event loop: use `to_thread`/`iterate_in_threadpool`.
- Any work started on behalf of a request must stop when the request does (see `Deadline`).
- New behaviour comes with a test that fails without it.

## Adding a chunking strategy

1. Write the splitter in `chunking.py` (`split_<name>(text, size, overlap)`), add it to `apply_strategy` and `STRATEGIES` with `built: True`.
2. Add a scoring rule in `score_strategies` with a one-line, number-bearing reason.
3. Add a test for the splitter and one for the decision.
4. Benchmark it: `run_bench.py chunking --configs <name>@500 recursive_character@500`. Keep it only if it beats recursive on some document shape; the sweep showed size usually matters more.
