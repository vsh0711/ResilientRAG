#!/usr/bin/env python3
"""
Quantitative benchmark on real PDFs, real embeddings, real Groq calls.

    python run_bench.py chunking                 # no API key needed
    python run_bench.py e2e --per-doc 12         # needs GROQ_API_KEY
    python run_bench.py unanswerable --n 30      # does it refuse when the file has no answer?

Inputs come from make_bench.py (four PDFs, questions with exact answer keys).

chunking      For each document and each chunking strategy (plus `auto`, the
              strategy the agent picks), index the real PDF and measure how
              often the top-3 retrieved chunks contain the whole fact
              sentence. Dense and hybrid+rerank are both reported, for the
              direct wording and the paraphrase.
e2e           Full pipeline accuracy. A question is correct when the answer
              contains the exact key (a number, a name, a phrase). No LLM
              decides correctness. Compared across:
                baseline   default recursive chunks, dense search, no healing
                agent      auto-chosen chunks, healing loop (up to 3 retries)
unanswerable  Questions about things the file never mentions. Correct means
              the model refused instead of inventing an answer.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL.parent / "backend"))
os.environ.setdefault("CACHE_ENABLED", "false")  # measure real latency, not Redis

BENCH = EVAL / "bench"
RESULTS = EVAL / "results"
DOCS = ["manual", "projects", "fieldnotes", "codebase"]
STRATEGIES = ["fixed", "recursive_character", "structure", "semantic", "code"]


def load_questions() -> list[dict]:
    with open(BENCH / "questions.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def squash(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[‐-―−]", "-", text)
    return re.sub(r"\s+", " ", text.replace("**", "")).strip().lower()


def contains(chunk: str, needle: str) -> bool:
    return squash(needle) in squash(chunk)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion. Honest error bars for small n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def pct(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k / max(n, 1):.0%} ({lo:.0%}–{hi:.0%})"


# --------------------------------------------------------------------------- chunking

def parse_config(label: str) -> tuple[str, int | None]:
    """'recursive_character@900' -> ('recursive_character', 900)."""
    name, _, size = label.partition("@")
    return name, int(size) if size else None


def run_chunking(args) -> dict:
    from qdrant_client import QdrantClient

    from app.agent.cache import hash_chunks
    from app.agent.chunking import plan_chunking
    from app.agent.document_loader import extract_pages
    from app.agent.retrieval import retrieve
    from app.agent.vectorstore import VectorStore, embed_texts

    questions = load_questions()
    store = VectorStore(client=QdrantClient(":memory:"))
    rows = []
    picks = {}
    for doc in DOCS:
        pages = extract_pages(str(BENCH / "docs" / f"{doc}.pdf"))
        qs = [q for q in questions if q["doc"] == doc]
        for strategy in args.configs:
            name, size = parse_config(strategy)
            holder: dict = {}
            t0 = time.perf_counter()
            for _ in plan_chunking(
                pages, embed_fn=embed_texts, force=None if name == "auto" else name,
                chunk_size=size, holder=holder,
            ):
                pass
            plan = holder["plan"]
            chunk_secs = time.perf_counter() - t0
            if name == "auto":
                picks[doc] = plan.strategy_id
            h = hash_chunks(plan.chunks) + f"_{doc}_{strategy}"
            store.index_chunks(h, plan.chunks)
            sizes = [len(c) for c in plan.chunks]
            for mode in ("dense", "hybrid_rerank"):
                for form in ("direct", "paraphrase"):
                    hits = rr = 0
                    for q in qs:
                        docs = retrieve(store, h, q[form], 3, mode)
                        rank = next((i for i, d in enumerate(docs) if contains(d, q["fact"])), None)
                        if rank is not None:
                            hits += 1
                            rr += 1 / (rank + 1)
                    rows.append({
                        "doc": doc, "strategy": strategy, "chosen": plan.strategy_id,
                        "mode": mode, "form": form, "n": len(qs), "hits": hits,
                        "mrr": rr / len(qs), "chunks": len(plan.chunks),
                        "avg_chars": round(statistics.mean(sizes)), "chunk_secs": round(chunk_secs, 2),
                    })
            print(f"{doc:11s} {strategy:20s} -> {plan.strategy_id:20s} {len(plan.chunks):4d} chunks", flush=True)
    return {"rows": rows, "auto_picks": picks, "k": 3, "configs": args.configs}


def report_chunking(res: dict) -> str:
    rows = res["rows"]
    STRATS = res.get("configs", STRATEGIES + ["auto"])
    out = ["# Chunking strategy benchmark", "",
           "Real PDFs, MiniLM dense + BM25 + cross-encoder rerank, top-3 retrieval. "
           "A hit means one of the three retrieved chunks contains the **entire** fact sentence, "
           "so a strategy that cuts a fact in two scores a miss. 40 questions per document, each asked "
           "in the document's wording and as a paraphrase (80 per cell below). Intervals are 95% Wilson.", ""]
    out.append(f"The agent's own picks: {', '.join(f'**{d}** → `{s}`' for d, s in res['auto_picks'].items())}.")
    out.append("")
    for mode in ("dense", "hybrid_rerank"):
        out += [f"## Retrieval: `{mode}`", "",
                "| document | " + " | ".join(STRATS) + " |",
                "|---|" + "---|" * len(STRATS)]
        for doc in DOCS:
            cells = []
            best = 0
            sel = [r for r in rows if r["doc"] == doc and r["mode"] == mode]
            for s in STRATS:
                rs = [r for r in sel if r["strategy"] == s]
                k, n = sum(r["hits"] for r in rs), sum(r["n"] for r in rs)
                best = max(best, k) if s != "auto" else best
                cells.append((s, k, n))
            row = []
            for s, k, n in cells:
                text = pct(k, n)
                if s != "auto" and k == best:
                    text = f"**{text}**"
                if s == "auto":
                    text += f" · picked `{res['auto_picks'][doc]}`"
                row.append(text)
            out.append(f"| {doc} | " + " | ".join(row) + " |")
        tot = {}
        for s in STRATS:
            rs = [r for r in rows if r["strategy"] == s and r["mode"] == mode]
            tot[s] = (sum(r["hits"] for r in rs), sum(r["n"] for r in rs))
        out.append("| **all four** | " + " | ".join(pct(*tot[s]) for s in STRATS) + " |")
        out.append("")
    out += ["## Paraphrase only (the hard half), `hybrid_rerank`", "",
            "| strategy | hits |", "|---|---|"]
    for s in STRATS:
        rs = [r for r in rows if r["strategy"] == s and r["mode"] == "hybrid_rerank" and r["form"] == "paraphrase"]
        out.append(f"| {s} | {pct(sum(r['hits'] for r in rs), sum(r['n'] for r in rs))} |")
    out += ["", "## Chunk counts", "", "| document | " + " | ".join(STRATS) + " |",
            "|---|" + "---|" * len(STRATS)]
    for doc in DOCS:
        cells = []
        for s in STRATS:
            r = next(r for r in rows if r["doc"] == doc and r["strategy"] == s and r["mode"] == "dense" and r["form"] == "direct")
            cells.append(f"{r['chunks']} × {r['avg_chars']}")
        out.append(f"| {doc} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------ pipeline

REFUSAL = re.compile(r"didn.t find|not (?:mentioned|found|provided|contain|available)|no (?:relevant|information|mention)|cannot (?:be )?(?:found|answer)|unable to (?:find|answer)|does not (?:say|mention|contain|include)|isn.t (?:mentioned|in)", re.I)


def build_plan(doc: str, strategy: str | None, size: int | None = None):
    from app.agent.chunking import plan_chunking
    from app.agent.document_loader import extract_pages
    from app.agent.vectorstore import VectorStore, embed_texts
    from app.agent.cache import hash_chunks

    pages = extract_pages(str(BENCH / "docs" / f"{doc}.pdf"))
    holder: dict = {}
    for _ in plan_chunking(pages, embed_fn=embed_texts, force=strategy, chunk_size=size, holder=holder):
        pass
    plan = holder["plan"]
    VectorStore().index_chunks(hash_chunks(plan.chunks), plan.chunks)
    return plan


def run_one(graph, chunks, question: str, retries: int) -> dict:
    """Run the graph node by node so every attempt is recorded, not just the last."""
    from app.agent.graph import initial_state

    t0 = time.perf_counter()
    state = dict(initial_state(chunks=chunks, query=question, max_retries=retries))
    attempts: list[dict] = []
    cur: dict = {}
    try:
        for update in graph.stream(state, stream_mode="updates"):
            for node, delta in update.items():
                state.update(delta)
                if node == "retrieve":
                    cur = {"mode": state["retrieval_mode"], "budget": state["retrieval_budget"],
                           "query": state["query"], "docs": len(delta.get("retrieved_docs", []))}
                elif node == "generate":
                    cur["answer"] = delta["answer"]
                elif node == "score":
                    cur.update(relevance=delta["relevance_score"], faithfulness=delta["faithfulness_score"],
                               score=delta["score"], failure_reason=delta["failure_reason"])
                    attempts.append(cur)
    except Exception as exc:  # rate limits, outages: counted, never hidden
        return {"error": f"{type(exc).__name__}: {str(exc)[:120]}", "secs": time.perf_counter() - t0}
    tokens = sum(t["prompt_tokens"] + t["completion_tokens"] for t in state.get("token_usage", {}).values())
    return {
        "answer": state["answer"], "score": state["score"], "retries": state["retry_count"],
        "mode": state["retrieval_mode"], "secs": time.perf_counter() - t0, "tokens": tokens,
        "llm_failed": "temporarily unable" in state["answer"],
        "attempts": attempts, "trace": state.get("healing_trace", []),
        "latency_ms": state.get("latency_ms", {}),
    }


def run_e2e(args) -> dict:
    from app.agent.graph import build_graph

    graph = build_graph()
    questions = load_questions()
    per_doc: dict[str, list[dict]] = {}
    for doc in DOCS:
        qs = [q for q in questions if q["doc"] == doc][: args.per_doc]
        per_doc[doc] = qs
    # baseline: what a typical tutorial ships. agent_no_heal isolates the chunking
    # gain from the healing gain.
    arms = {"baseline": ("recursive_character", 1600, 0), "agent_no_heal": (None, None, 0), "agent": (None, None, 3)}
    rows = []
    jobs = []
    plans = {}
    for doc in DOCS:
        for arm, (strategy, size, retries) in arms.items():
            plans[(doc, arm)] = build_plan(doc, strategy, size)
            print(f"{doc:11s} {arm:9s} chunks={len(plans[(doc, arm)].chunks)} strategy={plans[(doc, arm)].strategy_id}", flush=True)
            for q in per_doc[doc]:
                for form in ("direct", "paraphrase"):
                    jobs.append((doc, arm, retries, q, form))

    def work(job):
        doc, arm, retries, q, form = job
        out = run_one(graph, plans[(doc, arm)].chunks, q[form], retries)
        out.update(doc=doc, arm=arm, form=form, id=q["id"], key=q["key"], question=q[form])
        if "answer" in out:
            out["correct"] = contains(out["answer"], q["key"])
            for a in out["attempts"]:
                a["correct"] = contains(a.get("answer", ""), q["key"])
        return out

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, out in enumerate(pool.map(work, jobs), 1):
            rows.append(out)
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    return {"rows": rows, "per_doc": args.per_doc,
            "picks": {d: plans[(d, "agent")].strategy_id for d in DOCS}}


def report_e2e(res: dict) -> str:
    rows = res["rows"]
    out = ["# End-to-end answer accuracy", "",
           f"Real PDFs, real Groq calls. {res['per_doc']} questions per document, each in two wordings. "
           "Correct = the answer contains the exact key (number, name or phrase) from the document. "
           "No LLM judges correctness. Calls that failed upstream (rate limit, outage) are listed "
           "separately and excluded from accuracy, never counted as right.", "",
           f"Agent's chunking picks: {', '.join(f'**{d}** → `{s}`' for d, s in res['picks'].items())}.", ""]

    def arm_stats(sel):
        ok = [r for r in sel if "answer" in r and not r.get("llm_failed")]
        errs = len(sel) - len(ok)
        k = sum(r["correct"] for r in ok)
        return ok, errs, k

    out += ["| | n | correct | failed calls | median latency | mean tokens | healed (≥1 retry) |",
            "|---|---|---|---|---|---|---|"]
    for arm in ("baseline", "agent_no_heal", "agent"):
        sel = [r for r in rows if r["arm"] == arm]
        ok, errs, k = arm_stats(sel)
        lat = statistics.median([r["secs"] for r in ok]) if ok else 0
        tok = statistics.mean([r["tokens"] for r in ok]) if ok else 0
        healed = sum(1 for r in ok if r["retries"] > 0)
        out.append(f"| **{arm}** | {len(ok)} | {pct(k, len(ok))} | {errs} | {lat:.1f}s | {tok:,.0f} | {healed} |")
    out += ["", "## By document", "", "| document | baseline | agent_no_heal | agent |", "|---|---|---|---|"]
    for doc in DOCS:
        cells = []
        for arm in ("baseline", "agent_no_heal", "agent"):
            ok, errs, k = arm_stats([r for r in rows if r["arm"] == arm and r["doc"] == doc])
            cells.append(pct(k, len(ok)))
        out.append(f"| {doc} | " + " | ".join(cells) + " |")
    out += ["", "## By wording", "", "| wording | baseline | agent_no_heal | agent |", "|---|---|---|---|"]
    for form in ("direct", "paraphrase"):
        cells = []
        for arm in ("baseline", "agent_no_heal", "agent"):
            ok, errs, k = arm_stats([r for r in rows if r["arm"] == arm and r["form"] == form])
            cells.append(pct(k, len(ok)))
        out.append(f"| {form} | " + " | ".join(cells) + " |")

    # paired comparison: same question, both arms answered
    by = {}
    for r in rows:
        if "answer" in r and not r.get("llm_failed"):
            by.setdefault((r["id"], r["form"]), {})[r["arm"]] = r["correct"]
    both = [v for v in by.values() if "baseline" in v and "agent" in v]
    up = sum(1 for v in both if v["agent"] and not v["baseline"])
    down = sum(1 for v in both if v["baseline"] and not v["agent"])
    out += ["", "## Paired, same question under both arms", "",
            f"{len(both)} questions answered by both. The agent fixed **{up}**, broke **{down}**, "
            f"and tied on {len(both) - up - down}."]
    healed_rows = [r for r in rows if r["arm"] == "agent" and "answer" in r and r["retries"] > 0 and not r.get("llm_failed")]
    if healed_rows:
        k = sum(r["correct"] for r in healed_rows)
        out.append(f"Of the {len(healed_rows)} questions where the loop actually retried, {pct(k, len(healed_rows))} ended correct.")
    # latency percentiles per arm
    out += ["", "## Latency (seconds, whole question)", "", "| arm | p50 | p95 | max |", "|---|---|---|---|"]
    for arm in ("baseline", "agent_no_heal", "agent"):
        xs = sorted(r["secs"] for r in rows if r["arm"] == arm and "answer" in r)
        if xs:
            q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
            out.append(f"| {arm} | {q(0.5):.1f} | {q(0.95):.1f} | {xs[-1]:.1f} |")

    # the healing story: first attempt wrong, final right
    saved = [r for r in rows if r["arm"] == "agent" and r.get("attempts") and len(r["attempts"]) > 1
             and not r["attempts"][0]["correct"] and r.get("correct")]
    judged_bad = [r for r in rows if r["arm"] == "agent" and r.get("attempts") and not r["attempts"][0]["correct"]]
    out += ["", "## Healing traces: poor first answers that got fixed", "",
            f"In the `agent` arm, {len(judged_bad)} first attempts were wrong. The loop turned "
            f"**{len(saved)}** of them into correct answers.", ""]
    for r in saved[:6]:
        out.append(f"### “{r['question']}”  (wanted `{r['key']}`)" if r.get("question") else f"### {r['id']} ({r['form']}) wanted `{r['key']}`")
        out.append("")
        out.append("| attempt | search | relevance | faithful | score | correct | answer |")
        out.append("|---|---|---|---|---|---|---|")
        for i, a in enumerate(r["attempts"], 1):
            ans = a.get("answer", "").replace("\n", " ").replace("|", "/")[:90]
            out.append(f"| {i} | {a['mode']} k={a['budget']} | {a['relevance']:.2f} | {a['faithfulness']:.2f} | "
                       f"{a['score']:.2f} | {'yes' if a['correct'] else 'no'} | {ans} |")
        out.append("")
        for st in r["trace"]:
            out.append(f"- retry {st['retry_number']}: {st['failure_reason']} → {st['action_taken']}")
        out.append("")
    errs = [r for r in rows if "error" in r]
    if errs:
        out += ["", f"## Upstream failures ({len(errs)})", ""]
        out += [f"- `{r['error']}`" for r in errs[:5]]
    wrong = [r for r in rows if r["arm"] == "agent" and r.get("correct") is False][:6]
    if wrong:
        out += ["", "## Sample of the agent's misses", ""]
        out += [f"- ({r['form']}) wanted `{r['key']}` got “{r['answer'][:110]}”" for r in wrong]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------- unanswerable

def run_unanswerable(args) -> dict:
    from app.agent.graph import build_graph

    graph = build_graph()
    import random

    rng = random.Random(11)
    qs = []
    for i in range(args.n):
        kind = i % 3
        if kind == 0:
            qs.append(("manual", f"What does error E-{rng.randrange(900, 999)} mean on the Zephyr X200?"))
        elif kind == 1:
            qs.append(("projects", f"Who led Project {rng.choice(['Zyxqor', 'Wumbal', 'Frelnik', 'Ostrava'])}{rng.randrange(10, 99)}?"))
        else:
            qs.append(("codebase", f"How many times does rebuild_{rng.choice(['vaults', 'cursors', 'hooks'])}{rng.randrange(10, 99)} retry?"))
    plans = {d: build_plan(d, "recursive_character") for d in {d for d, _ in qs}}

    def work(item):
        doc, question = item
        out = run_one(graph, plans[doc].chunks, question, 3)
        out.update(doc=doc, question=question)
        if "answer" in out:
            out["refused"] = bool(REFUSAL.search(out["answer"]))
        return out

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(work, qs))
    return {"rows": rows}


def report_unanswerable(res: dict) -> str:
    rows = [r for r in res["rows"] if "answer" in r and not r.get("llm_failed")]
    k = sum(r["refused"] for r in rows)
    out = ["# Refusal on unanswerable questions", "",
           "Questions about error codes, projects and functions that do **not** exist in the file. "
           "Correct = the answer says it could not find it. Anything else is an invented answer.", "",
           f"Refused: {pct(k, len(rows))} of {len(rows)} (failed upstream calls excluded).", ""]
    bad = [r for r in rows if not r["refused"]]
    if bad:
        out += ["## Answers that should have been refusals", ""]
        out += [f"- “{r['question']}” → “{r['answer'][:140]}”" for r in bad[:8]]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["chunking", "e2e", "unanswerable"])
    ap.add_argument("--per-doc", type=int, default=12)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--configs", nargs="+", default=STRATEGIES + ["auto"],
                    help="strategy or strategy@size, e.g. recursive_character@900")
    ap.add_argument("--tag", default="", help="suffix for the results file name")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    fn, rp = {"chunking": (run_chunking, report_chunking), "e2e": (run_e2e, report_e2e),
              "unanswerable": (run_unanswerable, report_unanswerable)}[args.task]
    res = fn(args)
    stem = f"bench_{args.task}{('_' + args.tag) if args.tag else ''}"
    (RESULTS / f"{stem}.json").write_text(json.dumps(res, indent=1))
    report = rp(res)
    (RESULTS / f"{stem}.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
