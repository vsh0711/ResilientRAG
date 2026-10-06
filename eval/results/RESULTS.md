# Results so far

Everything here was measured on real PDFs with real embeddings and real Groq calls. Nothing is mocked. Anything not measured is listed under "Not measured".

## Setup

Four generated PDFs (numbered manual, continuous prose, topic-shifting field notes, Python source) with invented facts, so a model cannot answer from memory. 160 questions, each with an exact answer key; correct means the answer contains the key (string match, no LLM judge). Each question is asked in the document's wording and as a paraphrase. Intervals are 95% Wilson. Scripts: `eval/make_bench.py`, `eval/run_bench.py`.

## 1. Retrieval precision (complete run)

Metric: top-3 retrieved chunks contain the whole answer sentence. 80 queries per document per cell. Full tables: `bench_chunking_sweep.md`.

| chunking (all four documents) | dense | hybrid + rerank |
|---|---|---|
| recursive, 500 chars | 98% (95–99) | 100% (99–100) |
| semantic, 500 chars | 92% (88–94) | 98% (95–99) |
| fixed, 500 chars | 88% (84–91) | 98% (95–99) |
| recursive, 900 chars | 91% (87–93) | 100% (98–100) |
| recursive, 1,600 chars (old default) | 80% (75–84) | 89% (85–92) |
| semantic, 1,600 chars | 81% (76–85) | 95% (92–97) |

Findings:
- Chunk size mattered more than strategy. Going from 1,600 to 500 characters raised recursive splitting by 11 points with hybrid search and 18 points with dense only.
- At 500 characters, semantic chunking did not beat plain recursive splitting. It only looked good against the 1,600-character default.
- The default is now 500 characters. Caveat: this is single-fact lookup. Questions that need a wide passage may prefer larger chunks.
- The sweep's `auto` column used the 1,600 default and the earlier, miscalibrated strategy picker, so it is not a measurement of the current picker. The picker was recalibrated afterwards and has not been re-benchmarked.

## 2. End-to-end accuracy, latency, healing (partial run)

File: `bench_e2e_run1.md` (+ `.json` with every attempt). 8 questions × 4 documents × 2 wordings × 3 arms = 192 runs. 27 of 192 failed upstream (Groq rate limits) and are excluded, never counted as correct, so arms have different n.

| arm | answered | correct | median latency | p95 latency | mean tokens |
|---|---|---|---|---|---|
| baseline (1,600-char chunks, dense, no healing) | 53 | 83% (71–91) | 40.8 s | 105 s | 2,987 |
| agent, no healing | 62 | 87% (77–93) | 16.5 s | 67 s | 1,480 |
| agent, full healing loop | 50 | 96% (87–99) | 29.9 s | 104 s | 2,663 |

Paired on the 48 questions answered by both baseline and full agent: the agent fixed 6, broke 2, tied 40. Of 15 questions where the loop actually retried, 13 (87%) ended correct.

Healing traces (3 captured in the report) show the loop climbing dense → dense+rerank → hybrid → hybrid+rerank and rewriting the query, e.g. "Which person was in charge of the Priprimir initiative?" failed twice with "I didn't find any relevant documents", then after hybrid search plus a rewritten query returned the correct name.

Caveats, stated plainly:
- The intervals overlap between baseline and agent. 96% vs 83% is suggestive, not proven at this sample size.
- Latency is dominated by rate-limit backoff on a free key. It is a ceiling, not what the code costs on a paid tier.
- Defect found by this run: when the judge call failed, the old code scored the answer 0.00 and kept escalating retrieval on correct answers. That is why some traces show 0.00 scores next to correct answers. Fixed afterwards (`judge_unavailable`, commit "Do not heal when the judge itself is unavailable"); the fix has a unit test but this benchmark has not been re-run with it.
- The `agent` accuracy of 96% includes those wasted retries, so it does not reflect the fixed behaviour.

## 3. Load test, 100 concurrent users (complete, infrastructure only)

Real production stack in Docker (Caddy HTTPS → Next.js → 2 uvicorn workers → Qdrant + Redis), 6 CPUs and 4 GB for the Docker VM, Locust running on the same machine. 100 users ramped at 10/s for 4 minutes, 10–30 s think time. Mix: 70% policy lookups, 15% readiness checks, 8% PDF uploads (each upload is byte-different, so the server's "same file" shortcut cannot hide the analysis cost). **No LLM calls in this run**, so Groq's quota cannot skew it. File: `load_infra_stats.csv`.

| endpoint | requests | failures | p50 | p95 | p99 |
|---|---|---|---|---|---|
| GET /documents/chunking | 849 | 0 | 0.03 s | 0.12 s | 0.16 s |
| GET /health/ready | 188 | 0 | 0.13 s | 0.28 s | 0.89 s |
| POST /documents (upload + analyse + embed) | 81 | 0 | 20 s | 30 s | 33 s |
| POST /documents answered "busy" (deliberate 429 after 20 s) | 7 | 0 | 20 s | 20 s | 20 s |
| **all** | **1,125** | **0** | 0.04 s | 16 s | 29 s |

Upload latency is high because at most 4 uploads run at once (2 per worker) and the rest queue; solo, an upload takes about 1.6 s. Locust shares the 6 cores, so a dedicated host will do better. Backend memory settled at about 2 GB.

Four defects this test found, all fixed and covered by tests:
1. **Unbounded upload concurrency.** The first run pegged all 6 cores at 2.9 GB and returned 503 to every upload after about 70 s. Uploads are now capped and answered with a fast 429.
2. **Readiness probe blocked the event loop.** `/health/ready` made synchronous Qdrant/Redis calls on the loop (up to 38 s). It now runs in worker threads.
3. **Thread-pool starvation.** The default pool (about 10 threads) was filled by questions waiting on the LLM, starving the probe (p95 68 s). The pool is now sized to the concurrency caps and probes have their own threads.
4. **Abandoned questions kept spending quota.** A timed-out question's thread kept retrying LLM calls. Requests now carry a deadline the client honours, cancelled on timeout or client disconnect.

## 4. What limits real use: the Groq free tier

Read from response headers and error messages on this key, per model: **8,000 tokens per minute, 200,000 tokens per day, 1,000 requests per day.** One question costs roughly 1,500 to 4,000 tokens, so the daily token cap alone allows on the order of 50 to 130 questions per day, and the per-minute cap sustains only a few per minute. Running these benchmarks and load tests exhausted the judge model's daily budget ("Used 199732 of 200000"). The client now detects a spent daily budget and switches to the fallback model immediately instead of retrying.

Because of this, **question throughput at 100 users was not measured successfully**: the load-test runs that included questions failed on quota, not on the server. A paid Groq tier is required before 100 people use this. That is a provider limit, not a code limit, but it is the deciding one.

A second end-to-end benchmark run collapsed into upstream failures for the same reason and was discarded. A third was stopped. No numbers from them are used.

## Not measured

- Question latency and failure rate at 100 concurrent users (blocked by the quota above).
- Refusal rate on unanswerable questions (`run_bench.py unanswerable` exists, never run).
- End-to-end accuracy after the judge fix, Retry-After backoff, the 500-character default, the recalibrated picker and the deadline fix.
- Statistical significance of the agent vs baseline gap.
- Behaviour on a dedicated host (the load generator shared the machine).
