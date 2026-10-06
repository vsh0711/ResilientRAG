# Decisions and lessons

Short records of choices that are not obvious from the code, with the evidence behind them. Newest context first within each group.

## Choices

**D1. Two judges, not one.** Relevance ("did we fetch the right passages?") and faithfulness ("does the answer stay inside them?") are separate calls with schema-validated JSON. One blended score cannot tell you whether to change retrieval or generation. The judge is a different model family (Qwen) from the answerer (gpt-oss) to reduce self-preference.

**D2. A failed judge ends the loop; it does not trigger healing.** The first real benchmark showed a rate-limited judge scoring correct answers 0.00 and the loop escalating retrieval on them. A broken judge says nothing about the answer. It now returns the answer with score 0 and `failure_reason=judge_unavailable`.

**D3. Chunk size defaults to 500 characters.** On the benchmark, recursive splitting at 500 characters found the whole answer sentence in the top 3 100% of the time with hybrid search (98% dense-only); the usual ~1,600 characters (~400 tokens) managed 89% (80% dense). Strategy mattered less than size. Caveat: the benchmark is single-fact lookup; explanatory questions may prefer larger chunks, which the healing loop's growing retrieval budget partly covers. Data: `eval/results/bench_chunking_sweep.md`.

**D4. The strategy is chosen by measurement; the LLM only narrates.** Headings, code ratio, digit ratio, sentence length and a topic-shift probe feed explicit scoring rules. A Groq outage costs the prose, not the decision, and the decision is testable. The semantic threshold was recalibrated after the benchmark showed the probe read 19% on a topic-shifting document and 0% on uniform ones.

**D5. Two chunking strategies are described but not built.** Sentence-window and parent-child embed one text and return another, which needs a second index. They are scored and displayed, never selected, and labelled "not built" in the UI and docs.

**D6. Qdrant + Redis, not Postgres/pgvector.** Named dense and sparse vectors in one collection give hybrid search without a third service. Redis holds chunk text, answers and rate counters, and every use fails open.

**D7. Answer cache keyed on retrieval mode.** The same question can legitimately get different answers from different strategies; caching across modes would serve a `dense` answer after the loop escalated. Judge scores are not cached separately: answer and verdict are one unit.

**D8. Shared access codes instead of accounts.** The aim is to stop a public URL being a free LLM proxy, not to model users. A code is revoked by editing an environment variable. Accounts, history and document ownership are listed as gaps.

**D9. Rate limits per session, with an IP ceiling.** Offices put many people behind one IP; a per-IP limit would throttle them together. The ceiling only exists against session-id spraying.

**D10. SSE over WebSockets.** The reasoning is one-way, resumable by re-asking, and passes through proxies unchanged once buffering is off. The Next.js proxy and Caddy are both configured not to buffer.

**D11. Vercel calls the API directly.** A Vercel function caps request bodies at 4.5 MB and runtime at 60 s on the free plan, which breaks PDF uploads and long answers. `NEXT_PUBLIC_API_URL` makes the browser talk to the backend; CORS is the price.

**D12. The free demo is memory-only.** An in-process Qdrant, no Redis, one worker, on a free Space. Documents vanish on restart. That is acceptable for a demo and wrong for production, so it is a documented mode, not the default.

**D13. The UI paces the reasoning.** The server analyses in well under a second; the player replays events at reading speed with a skip button, because a result nobody can follow is not "thinking out loud".

## Lessons from running it for real

Each of these was invisible until the stack was actually run, and each now has a test or a safeguard.

| found by | what happened | fix |
|---|---|---|
| calling Groq | the repo's default models no longer existed for the key; every answer and judge call would fail | new defaults; `/health/ready` reports missing models |
| `docker compose up` | a plain `API_CORS_ORIGINS=http://…` crashed settings parsing, so the original compose file could never start | `NoDecode` + comma parsing + a test |
| 100-user load test | unbounded concurrent uploads pegged 6 cores / 2.9 GB, every upload 503 after 70 s | upload semaphore, fast 429, same-file shortcut |
| same test | `/health/ready` took up to 38 s: sync clients on the event loop | worker threads |
| same test | readiness got *worse* (68 s p95): the default thread pool (~10) was full of questions waiting on the LLM | sized pool; probes get their own threads |
| after the test | even a single question hung: timed-out requests left threads retrying LLM calls and spending quota | `Deadline`, cancelled on timeout/disconnect |
| reading the 429 text | the judge model had hit its **daily** token cap and the client kept sleeping on it | detect `TPD`/`RPD`, switch model at once |
| first benchmark | correct answers marked wrong by over-strict answer keys; a paired comparison that silently dropped everything with three arms | keys and report logic fixed before any number was published |
| benchmark design | semantic chunking looked 6 points better, but its chunks were smaller | added a chunk-size control; size explained most of the gap |

## Things deliberately left out

- **ColBERT late interaction**: large model, extra latency, marginal gain over dense + BM25 + cross-encoder at this scale.
- **Semantic answer cache**: needs an embedding on the hot path; exact-match caching is enough here.
- **OCR for scanned PDFs**: rejected with a clear message instead.
- **Per-domain score weighting**: the 50/50 blend and 0.8 threshold are configurable, not tuned per domain.
- **Multi-document routing**: one document per conversation.
