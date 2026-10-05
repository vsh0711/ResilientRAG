# 🩹 ResilientRAG

A self-healing Retrieval-Augmented Generation agent that detects *why* an answer is bad — bad retrieval or bad grounding — and automatically escalates its own strategy before retrying.

This is a from-scratch rebuild of [gurezende/SelfHealingRAG](https://github.com/gurezende/SelfHealingRAG), restructured as a production-shaped portfolio project: a FastAPI + LangGraph backend, a Next.js frontend, Groq for inference, persistent Qdrant + Redis, Docker, a 55-test suite, and an evaluation harness. Credit to [Gustavo R. Santos](https://gustavorsantos.me) for the original concept and graph design.

## Why the rename

"Self-Healing RAG" described the *goal*. "ResilientRAG" describes what actually got built: resilience isn't just the retry loop — it's the LLM-call backoff/fallback, the graceful cache degradation, and the split failure diagnosis that make the healing trustworthy instead of just a budget bump in a loop.

## What changed vs. the original repo

| Area | Original repo | ResilientRAG |
|---|---|---|
| Re-embedding | Re-embeds the **entire document on every retry** | Documents are embedded once per content hash and persisted in Qdrant; retries reuse the index |
| Retrieval strategies | Budget + binary rerank toggle only | 4-tier escalation ladder: `dense → dense_rerank → hybrid → hybrid_rerank`, with hybrid (dense+BM25, RRF-fused) **wired in and used** — the original shipped this code but never called it from the graph |
| Query on repeated failure | Unchanged, just re-run with a bigger budget | After the configured retry threshold, an LLM **rewrites the query** before the next retrieval attempt |
| Judge | One blended score from one LLM call, parsed with a bare `json.loads` | **Two independent judges** (relevance, faithfulness), each schema-validated via Pydantic from JSON-mode output — lets the agent tell "wrong docs" apart from "hallucinated answer" |
| LLM calls | Unguarded `client.chat.completions.create(...)` — any API error crashes the run | Retry with exponential backoff, then fall back to a smaller model, with per-call latency/token capture |
| UI coupling | Business logic (`nodes.py`) imports `streamlit` directly — can't run headlessly | Nodes are pure functions returning plain dicts; UI is a separate Next.js app talking to a FastAPI backend |
| Caching | None — same question re-embeds and re-asks every time | Redis-backed embedding/answer cache, degrades to a no-op if Redis is unreachable rather than crashing (see [Caching](#caching)) |
| Tests | None | 67 tests, 93% coverage (see [Testing](#testing)) |
| Evaluation | None | Labeled 20-question eval set, offline retrieval metrics + baseline-vs-healed generation metrics (see [Evaluation](#evaluation)) |
| Deployment | `streamlit run app.py` only | Dockerized (backend, frontend, Qdrant, Redis via `docker-compose`) |
| API hardening | None — single-user local script | Request-ID correlation, Redis-backed rate limiting, typed exception handling (no leaked stack traces), upload size limits, liveness + readiness health checks |

## Architecture

```mermaid
flowchart TD
    A[User asks a question] --> B[retrieve]
    B --> C[generate]
    C --> D["score (relevance judge + faithfulness judge)"]
    D -->|score >= 0.8 OR retries exhausted| E[END — return answer + trace]
    D -->|score < 0.8 AND retries remain| F[retry: diagnose + escalate strategy]
    F --> G[increment_retry]
    G --> B

    style D fill:#1d4e48,color:#5eead4
    style F fill:#4a2323,color:#f16565
```

The `retry` node is where the healing policy lives:

| Failure reason | Meaning | Healing action |
|---|---|---|
| `irrelevant_docs` | Retrieved docs don't match the query topic at all | Escalate retrieval mode one tier up the ladder, increase budget by 2; rewrite the query starting from the 2nd retry |
| `missing_context` | Docs are on-topic but incomplete | Escalate retrieval mode, increase budget by 3; rewrite the query starting from the 2nd retry |
| `unfaithful_answer` | Docs were fine, but the generated answer made unsupported claims | **Retrieval mode is left unchanged** (it's not a retrieval problem) — budget bumped by 1 and the next generation attempt uses a stricter grounding prompt |
| `none` | Answer passed both judges | Stop |

Splitting `irrelevant_docs`/`missing_context` (retrieval failures) from `unfaithful_answer` (a generation failure) is the piece the original repo's single blended judge couldn't do — it would see a low score either way and have no way to tell whether changing the retrieval strategy would even help.

## Caching

Two independent caching layers, both in Redis, both designed to **fail open** — if Redis is down, the app degrades to "no caching" rather than erroring out (`SafeRedisCache` in `backend/app/agent/cache.py` wraps every Redis call in a try/except and checks connectivity once at startup):

1. **Embedding/index persistence (the big one).** This is the direct fix for the original repo's worst bug: `retrieve_node` called `embed_docs(text)` unconditionally on *every single call*, including every retry — so a 3-retry healing run re-embedded the whole document 4 times. Here, each document is content-hashed (`hash_chunks()`), and `VectorStore.index_chunks()` checks `collection_exists()` before doing any embedding work at all. The collection is keyed by that hash, so the same document uploaded twice — or retried 5 times in one healing loop — is embedded exactly once. This isn't really "caching" in the request/response sense; it's closer to memoization backed by Qdrant itself as the persistent store, which is why it doesn't even need Redis to work.
2. **Answer cache (`AnswerCache`).** Keyed on `(document_hash, normalized_query, retrieval_mode)` — same document, same question (case/whitespace-insensitive), same retrieval strategy → skip the LLM call entirely and return the cached answer. This is the one that actually saves API cost/latency on repeat questions. It's deliberately *not* semantic (a rephrased question is a cache miss) — see [Trade-offs](#trade-offs-and-things-deliberately-left-out) for why.

Why `retrieval_mode` is part of the cache key rather than just `(document, query)`: the same question can legitimately get different answers depending on which retrieval strategy served it (that's the entire premise of self-healing) — caching across modes would mean a cached answer from a `dense`-mode pass could get served even after the healing loop escalated to `hybrid_rerank`, silently undoing the healing.

What's *not* cached, on purpose: judge scores. Caching the judge's verdict alongside the answer would mean a document that gets re-embedded with different content (same hash is unlikely to collide, but a *different* retrieval_mode on the same question is a real path) could serve a stale faithfulness verdict for a different actual answer. The answer and the judgment of that answer are cached as one unit or not at all.

## Tech stack

- **Orchestration:** LangGraph (kept — it's the right tool for explicit retry/branching state machines; this is the one piece of the original stack that wasn't swapped out)
- **Backend:** FastAPI (Python)
- **Frontend:** Next.js / React (TypeScript)
- **Model layer:** Groq (free tier) — `llama-3.3-70b-versatile` primary, `llama-3.1-8b-instant` fallback and judge model
- **Embeddings:** FastEmbed, local (`all-MiniLM-L6-v2` dense, Qdrant's BM25 sparse) — no API key or cost
- **Retrieval & memory:** Qdrant (persistent, named dense + sparse vectors)
- **Data layer:** Redis (embedding/answer cache)
- **Deployment:** Docker + docker-compose (backend, frontend, Qdrant, Redis as four services)

### A note on the originally-suggested stack

The brief also offered Next.js/Supabase/HumanLoop/Phoenix/Vercel. I kept LangGraph+Qdrant (proven fit, already in the source repo) and added FastAPI/Next.js/Groq/Redis/Docker on top, rather than also introducing Supabase+Postgres+Vercel — see [Trade-offs](#trade-offs-and-things-deliberately-left-out) for why.

## Project structure

```
ResilientRAG/
├── backend/
│   ├── app/
│   │   ├── agent/           # state, nodes, graph, retrieval, judges, llm, cache, vectorstore
│   │   ├── routers/         # FastAPI endpoints (documents, query)
│   │   ├── config.py        # env-driven settings
│   │   ├── schemas.py       # API request/response models
│   │   ├── exceptions.py    # typed operational errors (RetrievalBackendError, etc.)
│   │   ├── middleware.py    # request-ID correlation, Redis-backed rate limiting
│   │   └── main.py          # exception handlers, health/readiness checks
│   └── tests/                # 67 tests, mocked LLM/vector store, no network required
├── frontend/
│   └── app/                  # Next.js app router: upload panel, chat panel, healing trace view
├── eval/
│   ├── corpus.json            # 16-passage labeled knowledge base
│   ├── dataset.jsonl           # 20 questions with gold relevant chunks + difficulty tags
│   ├── run_retrieval_eval.py  # precision@k / recall@k per retrieval mode
│   ├── run_generation_eval.py # baseline vs. self-healing, scored end-to-end
│   └── results/                # output of both scripts (dry-run + real)
├── docker-compose.yml
└── .env.example
```

## Running it

### Docker (recommended)

```bash
cp .env.example .env   # fill in GROQ_API_KEY
docker compose up --build
```

Backend: `http://localhost:8000` (docs at `/docs`). Frontend: `http://localhost:3000`.

### Locally

```bash
# Backend
cd backend
pip install -e ".[dev]"
export GROQ_API_KEY=your_key_here
# Qdrant + Redis need to be running (docker compose up qdrant redis, or set
# QDRANT_USE_MEMORY=true / CACHE_ENABLED=false to skip them for a quick local run)
uvicorn app.main:app --reload

# Frontend, in another shell
cd frontend
npm install
npm run dev
```

## Testing

Unit + integration tests use a fake LangGraph-compatible `VectorStore` and `LLMClient` (see `backend/tests/conftest.py`), so the whole suite runs offline with no API key and no Docker services — including the real Qdrant engine in `:memory:` mode for the vector-store plumbing tests.

```
$ cd backend && pytest tests/ --cov=app --cov-report=term-missing
67 passed in 2.80s
```

| Module | Coverage |
|---|---|
| `agent/judges.py`, `agent/llm.py`, `agent/query_rewrite.py`, `agent/state.py`, `config.py`, `exceptions.py` | 100% |
| `agent/graph.py` | 100% |
| `middleware.py` | 98% |
| `agent/vectorstore.py` | 90% |
| `routers/documents.py` | 91% |
| `agent/cache.py` | 87% |
| `agent/nodes.py` | 93% |
| `main.py` | 83% |
| **Overall** | **93%** (718 statements, 52 missed — mostly FastAPI route plumbing not reachable without live infra) |

What's actually exercised, not just covered:
- **The healing loop really heals**: `test_heals_after_one_failed_attempt` scripts a judge that fails on attempt 1 (`missing_context`, score 0.6) and passes on attempt 2 (score 0.9), and asserts the graph terminates with the improved score and a one-entry healing trace.
- **The re-embedding bug is fixed**: `test_document_embedded_only_once_across_all_retries` runs a 3-retry healing sequence and asserts the vector store's `index_chunks` was called exactly once.
- **LLM resilience is real**: `test_llm_resilience.py` mocks the Groq client to raise `RateLimitError`/`APITimeoutError`, asserting correct backoff-then-retry, fallback-model switchover after retries are exhausted, and a typed `LLMCallError` (not a crash) when every attempt fails.
- **Relevance and faithfulness are genuinely independent**: `test_unfaithful_detected_independently_of_relevance` scripts a judge returning perfect relevance (1.0) alongside poor faithfulness (0.2) and asserts the failure reason is correctly attributed to generation, not retrieval.
- **Failures don't leak internals**: `test_hardening.py` forces a retrieval-backend exception through the real FastAPI app and asserts the response is a clean 503 with no exception class name or traceback text in the body — and does the same for an arbitrary unhandled exception (clean 500 + correlation ID).
- **The rate limiter fails open**: `test_unavailable_redis_fails_open` asserts requests are still served when Redis itself is down, so the limiter can't become a single point of failure.

## Completeness / what's actually been verified

Every code path below was genuinely exercised, not just written — here's exactly how, so you can judge where the line between "tested" and "assumed" actually sits:

| Layer | Verified how | Result |
|---|---|---|
| Core agent logic (healing loop, judges, retries, LLM resilience, caching) | 67 automated tests, run clean on a fresh `pip install` | 67/67 passing, 93% coverage |
| Document upload → PDF parsing → chunking | **Live HTTP request** against a running `uvicorn` server, with a real generated PDF, through the real `/documents` endpoint | Real response: `{"num_chunks": 9, ...}` |
| API hardening (clean errors, request IDs, health checks) | **Live HTTP requests** against the running server, including deliberately triggering a real network failure | Confirmed: clean 503 (not a raw stack trace), `X-Request-ID` header present, `/health/ready` correctly reports per-dependency status |
| Frontend build | `npm run build` | Compiles clean, 104KB first-load JS |
| Retrieval quality (precision/recall per mode), generation quality (baseline vs. healed scores) | Harness code runs end-to-end in `--dry-run` (fake models) | Harness is proven to run correctly; **quality numbers are not real** — see [Evaluation](#evaluation) |
| Docker Compose (Qdrant + Redis + backend + frontend together) | **Not run** — this build environment has no Docker daemon available | Compose file is written and each image builds from a Dockerfile that mirrors the locally-verified install steps, but the 4-service wiring itself hasn't been exercised live |
| `/query` end-to-end with a real LLM | **Not run** — this build environment's network reaches PyPI/npm/GitHub only, not HuggingFace Hub or the Groq API | Code path is unit-tested with fake LLM/embedding clients (see judge/graph tests above); the live network call itself is unverified until you run it |

**Bottom line:** the agent logic, the API layer, and the hardening are genuinely tested, including live HTTP calls where that was possible in this sandbox. The two things that remain unverified are specifically the two that need outbound internet this environment doesn't have: a live multi-container `docker compose up`, and an actual Groq-backed `/query` call. Both should just work — the code paths they'd exercise are the same ones already covered by the mocked tests — but "should work" and "verified" are different claims, and this table is here so you know which is which before you put specific numbers in front of anyone.

## Evaluation

### What dataset is used

`eval/corpus.json` + `eval/dataset.jsonl` — **a hand-authored dataset, not a public benchmark.** 16 short passages covering RAG concepts (dense vs. sparse retrieval, RRF fusion, cross-encoder reranking, chunking, LLM-as-judge, faithfulness, self-healing, caching, resilience, precision/recall, Docker), and 20 questions against them with gold-labeled relevant-chunk indices and a difficulty tag per question (`easy`/`medium`/`hard`).

Why hand-authored instead of a public dataset like SQuAD or Natural Questions: **precision/recall evaluation requires knowing which chunks are actually relevant to each question**, and public QA datasets don't come with "relevant chunk indices for a 16-passage corpus you invented" — that labeling has to happen regardless of where the source text comes from. Writing the corpus myself also let me build in the thing that actually tests this project's premise: the `hard`-tier questions (q17-q20) are deliberately paraphrased away from the corpus's own wording ("my app returned relevant passages but the answer still contained made-up facts" instead of "what is a hallucination") specifically so that plain dense retrieval is expected to struggle and hybrid/reranking/query-rewriting have something real to demonstrate improvement on. A random public PDF wouldn't give you that difficulty gradient for free.

If you want a more "recognizable" dataset for portfolio purposes — say, a real public-domain PDF or a known benchmark subset — I can swap or add one; paste a URL or tell me which source and I'll rebuild the gold labels against it. The methodology (precision@k/recall@k, baseline-vs-healed scoring) doesn't change either way, only the source text does.

### The two scripts

See **[`eval/README.md`](eval/README.md)** for full detail. Short version:

- `run_retrieval_eval.py` — precision@3 / recall@3 per retrieval mode against the 20 labeled questions.
- `run_generation_eval.py` — baseline (no healing) vs. self-healing, scored end-to-end through Groq.

**Both were authored and smoke-tested with `--dry-run`** inside this build environment, whose outbound network is restricted to PyPI/npm/GitHub — HuggingFace Hub (needed for real embeddings) and the Groq API are both unreachable from here. The dry-run mode swaps in deterministic fake embedding/LLM components (the same fixtures the test suite uses) purely to prove the harness runs end-to-end; **the numbers below are not real retrieval or answer quality** and are labeled `"dry_run": true` in the output files. Real numbers require running the two commands in `eval/README.md` on a machine with normal internet access and a Groq key — do that before citing these in a portfolio write-up.

<details>
<summary>Dry-run harness output (proof it runs, not a quality measurement)</summary>

Generation eval (baseline vs. self-healing), scripted with a judge that fails easy questions never, medium questions once, hard questions twice:

| Metric | Baseline | Self-Healing |
|---|---|---|
| Mean combined score | 0.758 | 0.925 |
| Pass rate (≥0.8) | 65.0% | 100.0% |

Questions improved: 7/20. Mean retries: 0.55/question. Mean token overhead: 24.8/question.

Full output: [`eval/results/generation_eval_report_dry_run.md`](eval/results/generation_eval_report_dry_run.md), [`eval/results/retrieval_eval_report_dry_run.md`](eval/results/retrieval_eval_report_dry_run.md).
</details>

## Trade-offs and things deliberately left out

- **ColBERT late-interaction retrieval** was in the original repo's dead code (`retrieve_docs_hybrid.py`) alongside dense+BM25, but I didn't wire it in. It's a third retrieval path with a large model download and real added latency, for marginal gain over dense+BM25+cross-encoder-rerank at this corpus scale. Documented as a clear next extension instead of built half-heartedly.
- **Supabase/Postgres instead of Qdrant+Redis** wasn't adopted. Qdrant was already proven in the source repo for this exact use case (named dense+sparse vectors, which Supabase's pgvector doesn't support as cleanly), and adding a third stateful service (Postgres, on top of Qdrant and Redis) for a single-document-at-a-time demo app would be infrastructure for its own sake.
- **Phoenix/HumanLoop instead of a custom eval harness**: chosen because it produces a self-contained, dependency-free artifact (plain JSON/Markdown) that doesn't require standing up another service or an external account to reproduce — at the cost of not getting a hosted tracing UI for free.
- **Combined score is a simple 50/50 average** of relevance and faithfulness. A production system would likely weight these per use case (e.g. faithfulness matters more for medical/legal domains) — left as a config knob (`score_pass_threshold`) rather than a fixed weighting scheme, since the right weighting is domain-specific.
- **The answer cache is keyed on exact (document, query, retrieval_mode)**, not semantic similarity — a rephrased question is a cache miss. A semantic cache (embed the query, check for a near-duplicate) would catch more repeats but adds another embedding call on the hot path for a demo-scale app.

## Possible extensions

- Multi-document routing (the original repo's suggestion, still open)
- Semantic answer caching
- ColBERT as a 5th retrieval tier for corpora large enough to need it
- Per-domain score weighting instead of a fixed 50/50 relevance/faithfulness blend
- Streaming the healing trace to the frontend as it happens, instead of returning it all at the end

## License

MIT, same as the original repo.
