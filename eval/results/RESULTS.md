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

## 3. What limits real use: the Groq free tier

Read from response headers on this key: 8,000 tokens per minute and 1,000 requests per day, per model. One question costs about 1,500 to 4,000 tokens, so this key sustains roughly 2 to 5 questions per minute in total. That is the actual ceiling for a 100-user deployment; it is a provider quota, not a property of the code. A paid Groq tier, or a shorter prompt per judge, is required before 100 people use it.

A second benchmark run, started with 2 workers, collapsed into upstream failures for this reason and was discarded. A third run with 1 worker and Retry-After handling was started and then stopped before finishing; no numbers from it are used anywhere.

## Not measured

- Refusal rate on unanswerable questions (`run_bench.py unanswerable` exists, never run).
- Load test with 100 concurrent users (`loadtest/locustfile.py` exists, never run; the quota above makes it uninformative on this key).
- End-to-end accuracy after the judge fix, the Retry-After backoff, the 500-character default and the recalibrated picker.
- Statistical significance of the agent vs baseline gap.
