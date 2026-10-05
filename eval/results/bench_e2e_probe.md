# End-to-end answer accuracy

Real PDFs, real Groq calls. 3 questions per document, each in two wordings. Correct = the answer contains the exact key (number, name or phrase) from the document. No LLM judges correctness. Calls that failed upstream (rate limit, outage) are listed separately and excluded from accuracy, never counted as right.

Agent's chunking picks: **manual** → `structure`, **projects** → `recursive_character`, **fieldnotes** → `semantic`, **codebase** → `code`.

| | n | correct | failed calls | median latency | mean tokens | healed (≥1 retry) |
|---|---|---|---|---|---|---|
| **baseline** | 24 | 88% (69%–96%) | 0 | 59.5s | 4,350 | 0 |
| **agent_no_heal** | 24 | 88% (69%–96%) | 0 | 10.6s | 1,840 | 0 |
| **agent** | 24 | 92% (74%–98%) | 0 | 16.5s | 2,176 | 1 |

## By document

| document | baseline | agent_no_heal | agent |
|---|---|---|---|
| manual | 67% (30%–90%) | 100% (61%–100%) | 100% (61%–100%) |
| projects | 100% (61%–100%) | 83% (44%–97%) | 83% (44%–97%) |
| fieldnotes | 100% (61%–100%) | 83% (44%–97%) | 83% (44%–97%) |
| codebase | 83% (44%–97%) | 83% (44%–97%) | 100% (61%–100%) |

## By wording

| wording | baseline | agent_no_heal | agent |
|---|---|---|---|
| direct | 92% (65%–99%) | 83% (55%–95%) | 92% (65%–99%) |
| paraphrase | 83% (55%–95%) | 92% (65%–99%) | 92% (65%–99%) |

## Paired, same question under both arms

0 questions answered by both. The agent fixed **0**, broke **0**, and tied on 0.
Of the 1 questions where the loop actually retried, 0% (0%–79%) ended correct.

## Sample of the agent's misses

- (paraphrase) wanted `Leona Castellan` got “I didn't find any relevant documents.”
- (direct) wanted `density of 35` got “The density of the mineral Vexfon is **35 grams per cubic centimetre**.”
