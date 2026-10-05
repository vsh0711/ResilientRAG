# Evaluation Harness

Two scripts, two layers of the system:

| Script | What it measures | Needs |
|---|---|---|
| `run_retrieval_eval.py` | Precision@3 / Recall@3 per retrieval mode (dense, dense_rerank, hybrid, hybrid_rerank) | Internet access to HuggingFace Hub (model download, first run only). No API key. |
| `run_generation_eval.py` | Baseline (no healing) vs self-healing: mean score, pass rate, retries, token/latency overhead | A `GROQ_API_KEY` |

Both scripts run against the same labeled dataset: `corpus.json` (16 hand-written passages about RAG concepts) and `dataset.jsonl` (20 questions, each with gold relevant-chunk indices and a `difficulty` tag — `easy`/`medium`/`hard`, where *hard* questions are deliberately paraphrased away from the corpus's exact wording to require hybrid retrieval or query rewriting).

## What is real

Only measured numbers are published here. Results come from real PDFs, real embeddings and real Groq calls. The older 16-passage scripts (`run_retrieval_eval.py`, `run_generation_eval.py`) keep a `--dry-run` flag that uses fake models to smoke-test the harness; their output is never committed. The headline benchmark is `make_bench.py` + `run_bench.py` (see the top-level README and `results/README.md`).

## Reproducing the real numbers

```bash
cd backend
pip install -e ".[dev]"

cd ../eval
python run_retrieval_eval.py                 # ~1-2 min, downloads 2 small models on first run
export GROQ_API_KEY=your_key_here
python run_generation_eval.py                 # ~1-2 min, makes ~100 Groq API calls (free tier)
```

Results are written to `results/retrieval_eval_report_real.md` and `results/generation_eval_report_real.md`, plus matching `.json` files with full per-question detail. Paste those two markdown files into the README's results section (or link them) once you've run them with your own key.

## Why this dataset design

* **Difficulty tiers exist on purpose.** Easy questions use the corpus's own vocabulary, so `dense` retrieval alone should already do well on them — they're a control group. Medium/hard questions are paraphrased or combine two passages, which is where `hybrid` retrieval, reranking, and query rewriting are expected to pull ahead of plain dense search. If the self-healing numbers don't show a precision/recall gap on the hard tier, that's a signal the healing strategy isn't actually earning its complexity — check that before trusting the summary numbers.
* **Baseline vs self-healing uses the same corpus and the same questions**, varying only `max_retries` (0 vs 3), so the score delta isolates the effect of the healing loop rather than any difference in test data.
* **Token/latency overhead is reported alongside the score improvement** — self-healing is a trade-off, not a free win, and a portfolio write-up should show the cost side too.
