# API reference

Base URL: wherever the backend runs (`http://localhost:8000` in dev; behind Docker's Caddy it is `https://<domain>/api/proxy`). Interactive docs at `/docs`.

When `ACCESS_CODES` is set, every route except `/health`, `/health/ready` and `/auth/status` needs the header `X-Access-Code: <code>`; otherwise the API answers `401 {"detail":"A valid access code is required."}`. Send `X-Session-Id: <8–64 chars of [A-Za-z0-9_-]>` so rate limits are per browser session instead of per IP.

## Endpoints

| method | path | purpose |
|---|---|---|
| GET | `/health` | liveness; never checks dependencies |
| GET | `/health/ready` | Qdrant, Redis, Groq key and **configured models**; 503 when Qdrant is down or a configured model is missing |
| GET | `/auth/status` | `{"required": bool}` so the UI knows whether to ask for a code |
| GET | `/documents/chunking` | default chunking policy and the seven strategies |
| POST | `/documents` | upload a PDF, wait, get the result as JSON |
| POST | `/documents/stream` | same upload, narrated as server-sent events |
| POST | `/query` | ask a question, get the final answer as JSON |
| POST | `/query/stream` | same, narrated node by node |

## POST /documents

Multipart form, field `file` (PDF, ≤ 20 MB).

```json
{
  "document_id": "abfcf026089311aa",
  "num_chunks": 48, "num_pages_estimate": 11, "indexed": true,
  "strategy_id": "structure", "strategy_label": "Document-structure-aware",
  "chunk_size": 500, "chunk_overlap": 62, "overlap_ratio": 0.124,
  "avg_chunk_chars": 637,
  "rationale": "I chose the document-structure-aware strategy because …"
}
```

## POST /query

```json
{ "document_id": "abfcf026089311aa", "question": "What does error E-265 mean?", "max_retries": 3 }
```

`question`: 1–2000 chars. `max_retries`: 0–3 (0 = no healing).

```json
{
  "answer": "…", "final_score": 1.0, "relevance_score": 1.0, "faithfulness_score": 1.0,
  "failure_reason": "none", "retry_count": 0, "retrieval_mode": "dense",
  "healing_trace": [ { "retry_number": 1, "failure_reason": "irrelevant_docs",
      "action_taken": "…", "previous_retrieval_mode": "dense", "new_retrieval_mode": "dense_rerank",
      "previous_budget": 3, "new_budget": 5, "query_rewritten": false, "rewritten_query": null } ],
  "latency_ms": { "retrieve": 18.7, "generate": 1012.6, "score": 700.1 },
  "token_usage": { "generate": { "prompt_tokens": 0, "completion_tokens": 0 } },
  "cache_hits": {}, "sources": ["passage text …"]
}
```

`failure_reason: "judge_unavailable"` means the answer was returned but could not be checked (`final_score` is 0).

## Server-sent events

Both `/stream` endpoints return `text/event-stream`; each frame is `data: <json>\n\n`. Validation errors (wrong type, too big, unknown document, busy) are returned **before** streaming starts as ordinary HTTP errors.

**Upload events** (`type`): `stage` {stage, text} · `think` {text} · `profile` {profile} · `scores` {rows[{id,name,built,score,verdict,reason}]} · `decision` {strategy_id,label,why} · `chunked` {num_chunks,avg,min,max,sample,sizes[]} · `done` {upload} · `error` {detail}. A file seen before replays `think`/`scores`/`decision`/`chunked` instantly.

**Query events**: `start` · `retrieve` {attempt,mode,budget,query,docs[]} · `generate` {attempt,answer,cached} · `score` {attempt,relevance,faithfulness,combined,failure_reason,passed,threshold} · `heal` {step} · `result` {result} (same shape as POST /query) · `error` {detail}.

## Status codes

| code | when |
|---|---|
| 400 | not a PDF, unreadable or password-protected PDF |
| 401 | missing or wrong access code |
| 404 | unknown `document_id` |
| 413 | file over the size limit |
| 422 | invalid request body |
| 429 | rate limit, or all upload/query slots busy (`Retry-After` set) |
| 503 | Qdrant unreachable, or a question exceeded its deadline |
| 500 | unexpected; body carries an `error_id` to find the log line |

Every response carries `X-Request-ID`; send your own to correlate.
