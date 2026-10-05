#!/usr/bin/env python3
"""
End-to-end generation evaluation: baseline (single-pass, no self-healing,
max_retries=0) vs the self-healing agent (max_retries=3), across the same
20-question dataset, run through the real LangGraph pipeline.

Requires GROQ_API_KEY (both generation and the two LLM judges call Groq).

Usage:
    export GROQ_API_KEY=...
    python run_generation_eval.py                 # real Groq calls
    python run_generation_eval.py --dry-run        # fake LLM, no network,
                                                    # smoke-tests the harness only

NOTE ON WHERE THIS WAS RUN: this harness was authored and smoke-tested
with --dry-run inside a sandboxed build environment with no outbound
access to the Groq API. The real (non-dry-run) numbers in
eval/results/generation_eval_report.md were produced by running this
script WITHOUT --dry-run with a real GROQ_API_KEY -- see eval/README.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"


def load_corpus() -> list[str]:
    with open(EVAL_DIR / "corpus.json") as f:
        return json.load(f)


def load_dataset() -> list[dict]:
    items = []
    with open(EVAL_DIR / "dataset.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _dry_run_judge_sequence(difficulty: str) -> list[dict]:
    """Deterministic, difficulty-correlated judge script for the fake LLM,
    so the dry-run smoke test actually exercises the healing path for
    medium/hard questions instead of trivially passing everything."""
    good = {"relevant_docs": True, "sufficient_context": True, "relevance_score": 0.92,
            "is_faithful": True, "faithfulness_score": 0.93}
    missing_ctx = {"relevant_docs": True, "sufficient_context": False, "relevance_score": 0.55,
                   "is_faithful": True, "faithfulness_score": 0.6}
    irrelevant = {"relevant_docs": False, "sufficient_context": False, "relevance_score": 0.3,
                  "is_faithful": True, "faithfulness_score": 0.4}

    if difficulty == "easy":
        return [good]
    if difficulty == "medium":
        return [missing_ctx, good]
    return [irrelevant, missing_ctx, good]  # hard


def run(dry_run: bool, max_retries_healed: int) -> dict:
    corpus = load_corpus()
    dataset = load_dataset()

    from app.agent.graph import build_graph, initial_state

    if dry_run:
        sys.path.insert(0, str(BACKEND_DIR / "tests"))
        from conftest import FakeLLMClient, FakeVectorStore  # type: ignore

    rows = []
    for item in dataset:
        if dry_run:
            store = FakeVectorStore()
            store.index_chunks("doc", corpus)
            llm_baseline = FakeLLMClient(judge_sequence=_dry_run_judge_sequence(item["difficulty"]))
            llm_healed = FakeLLMClient(judge_sequence=_dry_run_judge_sequence(item["difficulty"]))
            graph_baseline = build_graph(store=store, llm=llm_baseline)
            graph_healed = build_graph(store=store, llm=llm_healed)
        else:
            from app.agent.llm import ResilientLLMClient
            from app.agent.vectorstore import VectorStore
            store = VectorStore()
            llm = ResilientLLMClient()
            graph_baseline = build_graph(store=store, llm=llm)
            graph_healed = build_graph(store=store, llm=llm)

        t0 = time.perf_counter()
        baseline_result = graph_baseline.invoke(
            initial_state(chunks=corpus, query=item["question"], max_retries=0)
        )
        baseline_latency_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        healed_result = graph_healed.invoke(
            initial_state(chunks=corpus, query=item["question"], max_retries=max_retries_healed)
        )
        healed_latency_ms = (time.perf_counter() - t1) * 1000

        rows.append({
            "id": item["id"],
            "difficulty": item["difficulty"],
            "baseline_score": round(baseline_result["score"], 3),
            "healed_score": round(healed_result["score"], 3),
            "score_delta": round(healed_result["score"] - baseline_result["score"], 3),
            "healed_retry_count": healed_result["retry_count"],
            "baseline_latency_ms": round(baseline_latency_ms, 1),
            "healed_latency_ms": round(healed_latency_ms, 1),
            "baseline_tokens": sum(
                v.get("prompt_tokens", 0) + v.get("completion_tokens", 0)
                for v in baseline_result.get("token_usage", {}).values()
            ),
            "healed_tokens": sum(
                v.get("prompt_tokens", 0) + v.get("completion_tokens", 0)
                for v in healed_result.get("token_usage", {}).values()
            ),
            "healing_trace": healed_result.get("healing_trace", []),
        })

    n = len(rows)
    mean_baseline_score = sum(r["baseline_score"] for r in rows) / n
    mean_healed_score = sum(r["healed_score"] for r in rows) / n
    pass_threshold = 0.8
    baseline_pass_rate = sum(1 for r in rows if r["baseline_score"] >= pass_threshold) / n
    healed_pass_rate = sum(1 for r in rows if r["healed_score"] >= pass_threshold) / n
    improved_count = sum(1 for r in rows if r["score_delta"] > 0)
    mean_retries = sum(r["healed_retry_count"] for r in rows) / n
    mean_token_overhead = sum(r["healed_tokens"] - r["baseline_tokens"] for r in rows) / n
    mean_latency_overhead_ms = sum(r["healed_latency_ms"] - r["baseline_latency_ms"] for r in rows) / n

    return {
        "dry_run": dry_run,
        "num_questions": n,
        "pass_threshold": pass_threshold,
        "summary": {
            "mean_baseline_score": round(mean_baseline_score, 3),
            "mean_healed_score": round(mean_healed_score, 3),
            "baseline_pass_rate": round(baseline_pass_rate, 3),
            "healed_pass_rate": round(healed_pass_rate, 3),
            "questions_improved_by_healing": improved_count,
            "mean_retries_per_question": round(mean_retries, 2),
            "mean_token_overhead_per_question": round(mean_token_overhead, 1),
            "mean_latency_overhead_ms_per_question": round(mean_latency_overhead_ms, 1),
        },
        "per_question": rows,
    }


def write_markdown_report(results: dict, path: Path) -> None:
    s = results["summary"]
    lines = [
        "# Generation / Self-Healing Evaluation Report",
        "",
        f"Dry run: **{results['dry_run']}**  |  questions = {results['num_questions']}  |  "
        f"pass threshold = {results['pass_threshold']}",
        "",
        "## Baseline (single-pass) vs Self-Healing",
        "",
        "| Metric | Baseline | Self-Healing |",
        "|---|---|---|",
        f"| Mean combined score | {s['mean_baseline_score']} | {s['mean_healed_score']} |",
        f"| Pass rate (score >= {results['pass_threshold']}) | {s['baseline_pass_rate']*100:.1f}% | "
        f"{s['healed_pass_rate']*100:.1f}% |",
        "",
        f"- Questions improved by healing: **{s['questions_improved_by_healing']} / {results['num_questions']}**",
        f"- Mean retries per question: **{s['mean_retries_per_question']}**",
        f"- Mean extra tokens spent per question to heal: **{s['mean_token_overhead_per_question']}**",
        f"- Mean extra latency per question to heal: **{s['mean_latency_overhead_ms_per_question']} ms**",
        "",
        "## Per-question results",
        "",
        "| ID | Difficulty | Baseline score | Healed score | Retries | Token overhead |",
        "|---|---|---|---|---|---|",
    ]
    for r in results["per_question"]:
        lines.append(
            f"| {r['id']} | {r['difficulty']} | {r['baseline_score']} | {r['healed_score']} | "
            f"{r['healed_retry_count']} | {r['healed_tokens'] - r['baseline_tokens']} |"
        )

    lines += [
        "",
        "## How to reproduce",
        "",
        "```bash",
        "cd eval",
        "export GROQ_API_KEY=your_key_here",
        "python run_generation_eval.py            # real Groq calls, costs API credits",
        "python run_generation_eval.py --dry-run  # offline smoke test, not real quality numbers",
        "```",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = run(dry_run=args.dry_run, max_retries_healed=args.max_retries)

    suffix = "dry_run" if args.dry_run else "real"
    json_path = RESULTS_DIR / f"generation_eval_{suffix}.json"
    md_path = RESULTS_DIR / f"generation_eval_report_{suffix}.md"
    json_path.write_text(json.dumps(results, indent=2))
    write_markdown_report(results, md_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(results["summary"], indent=2))
