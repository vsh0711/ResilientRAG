# End-to-end answer accuracy

Real PDFs, real Groq calls. 8 questions per document, each in two wordings. Correct = the answer contains the exact key (number, name or phrase) from the document. No LLM judges correctness. Calls that failed upstream (rate limit, outage) are listed separately and excluded from accuracy, never counted as right.

Agent's chunking picks: **manual** → `structure`, **projects** → `recursive_character`, **fieldnotes** → `semantic`, **codebase** → `code`.

| | n | correct | failed calls | median latency | mean tokens | healed (≥1 retry) |
|---|---|---|---|---|---|---|
| **baseline** | 53 | 83% (71%–91%) | 11 | 34.2s | 2,987 | 0 |
| **agent_no_heal** | 62 | 87% (77%–93%) | 2 | 15.8s | 1,480 | 0 |
| **agent** | 50 | 96% (87%–99%) | 14 | 29.8s | 2,663 | 15 |

## By document

| document | baseline | agent_no_heal | agent |
|---|---|---|---|
| manual | 88% (64%–97%) | 100% (81%–100%) | 100% (81%–100%) |
| projects | 88% (64%–97%) | 81% (57%–93%) | 100% (81%–100%) |
| fieldnotes | 87% (62%–96%) | 100% (81%–100%) | 88% (64%–97%) |
| codebase | 50% (19%–81%) | 64% (39%–84%) | 100% (34%–100%) |

## By wording

| wording | baseline | agent_no_heal | agent |
|---|---|---|---|
| direct | 81% (63%–92%) | 94% (80%–98%) | 92% (75%–98%) |
| paraphrase | 85% (66%–94%) | 80% (63%–90%) | 100% (87%–100%) |

## Paired, same question under both arms

48 questions answered by both. The agent fixed **6**, broke **2**, and tied on 40.
Of the 15 questions where the loop actually retried, 87% (62%–96%) ended correct.

## Latency (seconds, whole question)

| arm | p50 | p95 | max |
|---|---|---|---|
| baseline | 40.8 | 105.0 | 158.2 |
| agent_no_heal | 16.5 | 66.7 | 76.1 |
| agent | 29.9 | 103.7 | 175.7 |

## Healing traces: poor first answers that got fixed

In the `agent` arm, 3 first attempts were wrong. The loop turned **3** of them into correct answers.

### “Which person was in charge of the Priprimir initiative at Northfield?”  (wanted `Leona Castellan`)

| attempt | search | relevance | faithful | score | correct | answer |
|---|---|---|---|---|---|---|
| 1 | dense k=3 | 0.00 | 0.00 | 0.00 | no | I didn't find any relevant documents. |
| 2 | dense_rerank k=5 | 0.00 | 0.00 | 0.00 | no | I didn't find any relevant documents. |
| 3 | hybrid k=7 | 0.00 | 0.00 | 0.00 | yes | Leona Castellan led the Priprimir initiative. |
| 4 | hybrid_rerank k=9 | 0.00 | 0.00 | 0.00 | yes | Leona Castellan. |

- retry 1: irrelevant_docs → Irrelevant docs -> escalated retrieval mode dense -> dense_rerank, increased budget by 2
- retry 2: irrelevant_docs → Irrelevant docs -> escalated retrieval mode dense_rerank -> hybrid, increased budget by 2; rewrote query -> 'Who headed the Priprimir initiative at Northfield?'
- retry 3: irrelevant_docs → Irrelevant docs -> escalated retrieval mode hybrid -> hybrid_rerank, increased budget by 2; rewrote query -> 'Who was the director of Northfield’s Priprimir initiative?'

### “Which person was in charge of the Quimiros initiative at Northfield?”  (wanted `Corin Vestergaard`)

| attempt | search | relevance | faithful | score | correct | answer |
|---|---|---|---|---|---|---|
| 1 | dense k=3 | 0.60 | 0.00 | 0.30 | no | I didn't find any relevant documents. |
| 2 | dense_rerank k=6 | 0.00 | 0.00 | 0.00 | yes | Corin Vestergaard. |
| 3 | hybrid k=8 | 0.00 | 0.00 | 0.00 | yes | Corin Vestergaard. |
| 4 | hybrid_rerank k=10 | 0.70 | 1.00 | 0.85 | yes | Corin Vestergaard. |

- retry 1: missing_context → Missing context -> increased retrieval budget by 3, escalated retrieval mode dense -> dense_rerank
- retry 2: irrelevant_docs → Irrelevant docs -> escalated retrieval mode dense_rerank -> hybrid, increased budget by 2; rewrote query -> 'Who was the project lead for the Quimiros initiative at Northfield?'
- retry 3: irrelevant_docs → Irrelevant docs -> escalated retrieval mode hybrid -> hybrid_rerank, increased budget by 2; rewrote query -> 'Who was the director of the Quimiros program at Northfield?'

### “Which person was in charge of the Drarokel initiative at Northfield?”  (wanted `Thea Pellegrino`)

| attempt | search | relevance | faithful | score | correct | answer |
|---|---|---|---|---|---|---|
| 1 | dense k=3 | 0.80 | 0.00 | 0.40 | no | I didn't find any relevant documents. |
| 2 | dense k=4 | 0.00 | 0.00 | 0.00 | yes | The documents state that **Project Drarokel was led by Thea Pellegrino**【Document 1】.    H |
| 3 | dense_rerank k=6 | 0.30 | 1.00 | 0.65 | yes | The Drarokel project was led by Thea Pellegrino. |
| 4 | hybrid k=8 | 0.00 | 0.00 | 0.00 | yes | Thea Pellegrino oversaw the Drarokel initiative. |

- retry 1: unfaithful_answer → Unfaithful answer -> widened context by 1 doc and will regenerate with stricter grounding instructions (retrieval mode unchanged: this is a generation-side failure, not a retrieval-side one)
- retry 2: irrelevant_docs → Irrelevant docs -> escalated retrieval mode dense -> dense_rerank, increased budget by 2; rewrote query -> 'Who was the lead of the Drarokel project at Northfield?'
- retry 3: irrelevant_docs → Irrelevant docs -> escalated retrieval mode dense_rerank -> hybrid, increased budget by 2; rewrote query -> 'Who oversaw the Drarokel initiative at Northfield?'


## Upstream failures (14)

- `LLMCallError: All LLM attempts failed after 4 tries across 2 model(s)`
- `LLMCallError: All LLM attempts failed after 4 tries across 2 model(s)`
- `LLMCallError: All LLM attempts failed after 4 tries across 2 model(s)`
- `LLMCallError: All LLM attempts failed after 4 tries across 2 model(s)`
- `LLMCallError: All LLM attempts failed after 4 tries across 2 model(s)`

## Sample of the agent's misses

- (direct) wanted `102 grams` got “I didn't find any relevant documents.”
- (direct) wanted `69 grams per cubic` got “The specific gravity of the mineral Bakel is **69**.”
