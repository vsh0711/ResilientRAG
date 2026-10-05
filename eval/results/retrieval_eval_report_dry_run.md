# Retrieval Evaluation Report

Dry run: **True**  |  k = 3  |  questions = 20  |  corpus chunks = 16
Indexing latency: 1.92 ms

## Overall (mean across all 20 questions)

| Retrieval mode | Precision@3 | Recall@3 | Mean latency (ms) |
|---|---|---|---|
| dense | 0.1 | 0.242 | 0.15 |
| dense_rerank | 0.1 | 0.242 | 0.22 |
| hybrid | 0.083 | 0.225 | 0.26 |
| hybrid_rerank | 0.083 | 0.225 | 0.36 |

## Breakdown by question difficulty

| Retrieval mode | easy P@3 / R@3 (n) | hard P@3 / R@3 (n) | medium P@3 / R@3 (n) |
|---|---|---|---|
| dense | 0.102 / 0.308 (n=13) | 0.167 / 0.208 (n=4) | 0.0 / 0.0 (n=3) |
| dense_rerank | 0.102 / 0.308 (n=13) | 0.167 / 0.208 (n=4) | 0.0 / 0.0 (n=3) |
| hybrid | 0.102 / 0.308 (n=13) | 0.083 / 0.125 (n=4) | 0.0 / 0.0 (n=3) |
| hybrid_rerank | 0.102 / 0.308 (n=13) | 0.083 / 0.125 (n=4) | 0.0 / 0.0 (n=3) |

## How to reproduce

```bash
cd eval
python run_retrieval_eval.py           # real embeddings, needs internet for first-run model download
python run_retrieval_eval.py --dry-run # offline smoke test of the harness only, not real quality numbers
```