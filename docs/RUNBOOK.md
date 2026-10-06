# Runbook

For whoever is on call, including future you. Commands assume the Docker production stack:
`alias dc='docker compose -f docker-compose.yml -f docker-compose.prod.yml'`.

## First look

```bash
dc ps                                                    # all five services up, backend (healthy)
curl -s https://$DOMAIN/api/proxy/health/ready -H "X-Access-Code: $CODE"
dc logs --since 10m backend | grep -E "ERROR|WARNING"
```

Every log line carries `[request_id=…]`. A user-facing error with an `error_id` maps to the log line containing that id.

## Symptoms

| symptom | likely cause | check / fix |
|---|---|---|
| answers say "temporarily unable to generate an answer" | LLM provider failing or quota spent | `dc logs backend \| grep "Rate limit"`. A `(TPD)` line means the **daily** token budget is gone: wait for the window to roll, or add a paid key. The client already falls back to the smaller model. |
| every question returns 503 "took too long" after 90 s | provider throttling with retries, or Qdrant slow | same grep; `curl /health/ready` for Qdrant |
| `/health/ready` is 503 with `llm_models.missing` | a configured Groq model was retired | `curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $KEY"`, then fix `GROQ_*_MODEL` and restart the backend |
| uploads return 429 "Many files are being read" | all upload slots busy (2 per worker) | normal under bursts; clients retry after `Retry-After`. Raise `MAX_CONCURRENT_UPLOADS` only if CPU and memory allow. |
| 429 "Rate limit exceeded" | one session over 60/min, or one IP over 1,200/min | raise `RATE_LIMIT_PER_MINUTE`; check for abuse in logs |
| 401 for everyone | wrong `ACCESS_CODES` after an edit | the UI keeps the old code in localStorage; users re-enter it |
| UI shows "The API is not reachable" | backend down or CORS | `dc logs backend`; confirm `API_CORS_ORIGINS` has the exact origin |
| backend container restarts | out of memory | `docker stats`; the backend used ~2 GB at rest and ~3 GB under load. Give the host 8 GB or lower `UVICORN_WORKERS`. |
| document "not found" after a restart (demo) | in-memory vectors were lost | expected in demo mode; re-upload |
| first upload after a deploy is slow | models downloading | `WARMUP_MODELS=true` (Docker default) does it at start |

## Routine tasks

**Rotate or revoke an access code.** Edit `ACCESS_CODES` in `.env` (comma-separated), then `dc up -d backend`. Old codes stop working immediately.

**Upgrade.**
```bash
git pull && dc up -d --build
curl -s https://$DOMAIN/api/proxy/health/ready -H "X-Access-Code: $CODE"
```
Collections are keyed by a content hash of the chunks. If you change chunking defaults, old documents keep working; re-uploading creates a new collection.

**Back up.** Qdrant and Redis state sit in the Docker volumes `qdrant_data` and `redis_data`.
```bash
docker run --rm -v resilientrag_qdrant_data:/d -v "$PWD":/b alpine tar czf /b/qdrant-$(date +%F).tgz -C /d .
docker run --rm -v resilientrag_redis_data:/d -v "$PWD":/b alpine tar czf /b/redis-$(date +%F).tgz -C /d .
```
(Qdrant is safest backed up while stopped, or via its snapshot API.)

**Free disk / forget documents.** There is no retention policy. To wipe everything: `dc down -v` (this deletes Qdrant, Redis and Caddy certificate volumes). To delete one document, delete its Qdrant collection `resilientrag_chunks_<document_id>` and the Redis keys `docchunks:<id>` and any `upload:*` record pointing at it.

**Move to a bigger LLM plan.** Set a paid key in `GROQ_API_KEY`, restart, and re-run the benchmarks (see DEVELOPMENT.md); the published accuracy numbers predate several fixes.

## Capacity facts (measured)

- 100 simulated users, no LLM calls, 6 CPUs / 4 GB: 1,125 requests, 0 failures; light endpoints p95 ≤ 0.28 s; uploads queue (2 per worker) with a median 20 s under that load, ~1.6 s alone.
- LLM throughput is the real limit: see the provider quota section in DEPLOYMENT.md.
- Full tables: `eval/results/RESULTS.md`.

## Alerting suggestions (not set up)

Probe `/health/ready` every minute (it returns 503 on a missing model or Qdrant outage); alert on the 5xx rate and on repeated `Daily token quota exhausted` log lines.
