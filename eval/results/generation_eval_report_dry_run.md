# Generation / Self-Healing Evaluation Report

Dry run: **True**  |  questions = 20  |  pass threshold = 0.8

## Baseline (single-pass) vs Self-Healing

| Metric | Baseline | Self-Healing |
|---|---|---|
| Mean combined score | 0.758 | 0.925 |
| Pass rate (score >= 0.8) | 65.0% | 100.0% |

- Questions improved by healing: **7 / 20**
- Mean retries per question: **0.55**
- Mean extra tokens spent per question to heal: **24.8**
- Mean extra latency per question to heal: **0.7 ms**

## Per-question results

| ID | Difficulty | Baseline score | Healed score | Retries | Token overhead |
|---|---|---|---|---|---|
| q01 | easy | 0.925 | 0.925 | 0 | 0 |
| q02 | easy | 0.925 | 0.925 | 0 | 0 |
| q03 | easy | 0.925 | 0.925 | 0 | 0 |
| q04 | easy | 0.925 | 0.925 | 0 | 0 |
| q05 | easy | 0.925 | 0.925 | 0 | 0 |
| q06 | medium | 0.575 | 0.925 | 1 | 45 |
| q07 | easy | 0.925 | 0.925 | 0 | 0 |
| q08 | easy | 0.925 | 0.925 | 0 | 0 |
| q09 | easy | 0.925 | 0.925 | 0 | 0 |
| q10 | medium | 0.575 | 0.925 | 1 | 45 |
| q11 | easy | 0.925 | 0.925 | 0 | 0 |
| q12 | medium | 0.575 | 0.925 | 1 | 45 |
| q13 | easy | 0.925 | 0.925 | 0 | 0 |
| q14 | easy | 0.925 | 0.925 | 0 | 0 |
| q15 | easy | 0.925 | 0.925 | 0 | 0 |
| q16 | easy | 0.925 | 0.925 | 0 | 0 |
| q17 | hard | 0.35 | 0.925 | 2 | 90 |
| q18 | hard | 0.35 | 0.925 | 2 | 90 |
| q19 | hard | 0.35 | 0.925 | 2 | 90 |
| q20 | hard | 0.35 | 0.925 | 2 | 90 |

## How to reproduce

```bash
cd eval
export GROQ_API_KEY=your_key_here
python run_generation_eval.py            # real Groq calls, costs API credits
python run_generation_eval.py --dry-run  # offline smoke test, not real quality numbers
```