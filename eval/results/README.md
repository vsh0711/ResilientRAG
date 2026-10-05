# Results

Every file here comes from `eval/run_bench.py` run against the PDFs that
`eval/make_bench.py` builds. Nothing is a dry run unless the file name says so.

| file | what it measures | needs |
|---|---|---|
| `bench_chunking.md` | retrieval hit rate for every chunking strategy and the agent's own pick | no API key |
| `bench_chunking_sweep.md` | same, plus chunk size 500 / 900 / 1600 | no API key |
| `bench_e2e_run1.md` | answer accuracy (partial run, see RESULTS.md), baseline vs the full agent | Groq key |
| `RESULTS.md` | the write-up: numbers, caveats, what is not measured | |
