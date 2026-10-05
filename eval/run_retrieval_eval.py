#!/usr/bin/env python3
"""
Offline retrieval quality evaluation: precision@k / recall@k for each
retrieval mode (dense, dense_rerank, hybrid, hybrid_rerank) against a
hand-labeled 20-question dataset over eval/corpus.json.

This does NOT require a GROQ_API_KEY — it only exercises the embedding +
vector-search layer, which runs locally via fastembed (downloads small
models from HuggingFace Hub on first run).

Usage:
    python run_retrieval_eval.py                 # real embeddings, real Qdrant
    python run_retrieval_eval.py --dry-run        # fake embeddings, no network
                                                   # (smoke-tests the harness only)

NOTE ON WHERE THIS WAS RUN: this harness was authored and smoke-tested
with --dry-run inside a sandboxed build environment whose outbound network
is restricted to PyPI/npm/GitHub (HuggingFace Hub is unreachable there).
The real (non-dry-run) numbers in eval/results/retrieval_eval_report.md
were produced by running this script WITHOUT --dry-run on a machine with
normal internet access -- see eval/README.md for exact reproduction steps.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"

RETRIEVAL_MODES = ["dense", "dense_rerank", "hybrid", "hybrid_rerank"]
K = 3
DOCUMENT_HASH = "eval_corpus_v1"


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


def precision_recall_at_k(retrieved_texts: list[str], relevant_texts: set[str], k: int) -> tuple[float, float]:
    top_k = retrieved_texts[:k]
    hits = sum(1 for t in top_k if t in relevant_texts)
    precision = hits / max(len(top_k), 1)
    recall = hits / max(len(relevant_texts), 1)
    return precision, recall


def run(dry_run: bool) -> dict:
    corpus = load_corpus()
    dataset = load_dataset()

    if dry_run:
        from tests.test_vectorstore import FakeDenseModel, FakeReranker, FakeSparseModel
        import app.agent.vectorstore as vs_module
        vs_module._dense_model = lambda: FakeDenseModel()
        vs_module._sparse_model = lambda: FakeSparseModel()
        vs_module._reranker_model = lambda: FakeReranker()

    from app.agent.retrieval import retrieve
    from app.agent.state import RetrievalMode
    from app.agent.vectorstore import VectorStore
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:") if dry_run else None
    store = VectorStore(client=client)

    t0 = time.perf_counter()
    store.index_chunks(DOCUMENT_HASH, corpus)
    index_latency_ms = (time.perf_counter() - t0) * 1000

    results: dict = {
        "dry_run": dry_run,
        "k": K,
        "num_questions": len(dataset),
        "num_corpus_chunks": len(corpus),
        "index_latency_ms": round(index_latency_ms, 2),
        "per_mode": {},
    }

    for mode in RETRIEVAL_MODES:
        per_question = []
        mode_latencies = []
        for item in dataset:
            relevant_texts = {corpus[i] for i in item["relevant_chunk_indices"]}
            t1 = time.perf_counter()
            retrieved = retrieve(store, DOCUMENT_HASH, item["question"], k=K, mode=RetrievalMode(mode))
            latency_ms = (time.perf_counter() - t1) * 1000
            mode_latencies.append(latency_ms)

            precision, recall = precision_recall_at_k(retrieved, relevant_texts, K)
            per_question.append({
                "id": item["id"],
                "difficulty": item["difficulty"],
                "precision_at_k": round(precision, 3),
                "recall_at_k": round(recall, 3),
                "latency_ms": round(latency_ms, 2),
            })

        mean_precision = sum(q["precision_at_k"] for q in per_question) / len(per_question)
        mean_recall = sum(q["recall_at_k"] for q in per_question) / len(per_question)

        by_difficulty: dict[str, list] = {}
        for q in per_question:
            by_difficulty.setdefault(q["difficulty"], []).append(q)
        difficulty_breakdown = {
            diff: {
                "mean_precision_at_k": round(sum(x["precision_at_k"] for x in qs) / len(qs), 3),
                "mean_recall_at_k": round(sum(x["recall_at_k"] for x in qs) / len(qs), 3),
                "n": len(qs),
            }
            for diff, qs in by_difficulty.items()
        }

        results["per_mode"][mode] = {
            "mean_precision_at_k": round(mean_precision, 3),
            "mean_recall_at_k": round(mean_recall, 3),
            "mean_latency_ms": round(sum(mode_latencies) / len(mode_latencies), 2),
            "by_difficulty": difficulty_breakdown,
            "per_question": per_question,
        }

    return results


def write_markdown_report(results: dict, path: Path) -> None:
    lines = [
        "# Retrieval Evaluation Report",
        "",
        f"Dry run: **{results['dry_run']}**  |  k = {results['k']}  |  "
        f"questions = {results['num_questions']}  |  corpus chunks = {results['num_corpus_chunks']}",
        f"Indexing latency: {results['index_latency_ms']} ms",
        "",
        "## Overall (mean across all 20 questions)",
        "",
        "| Retrieval mode | Precision@3 | Recall@3 | Mean latency (ms) |",
        "|---|---|---|---|",
    ]
    for mode, data in results["per_mode"].items():
        lines.append(
            f"| {mode} | {data['mean_precision_at_k']} | {data['mean_recall_at_k']} | "
            f"{data['mean_latency_ms']} |"
        )

    lines += ["", "## Breakdown by question difficulty", ""]
    difficulties = sorted({d for mode_data in results["per_mode"].values() for d in mode_data["by_difficulty"]})
    header = "| Retrieval mode | " + " | ".join(f"{d} P@3 / R@3 (n)" for d in difficulties) + " |"
    sep = "|---|" + "---|" * len(difficulties)
    lines += [header, sep]
    for mode, data in results["per_mode"].items():
        row = [mode]
        for d in difficulties:
            if d in data["by_difficulty"]:
                dd = data["by_difficulty"][d]
                row.append(f"{dd['mean_precision_at_k']} / {dd['mean_recall_at_k']} (n={dd['n']})")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## How to reproduce",
        "",
        "```bash",
        "cd eval",
        "python run_retrieval_eval.py           # real embeddings, needs internet for first-run model download",
        "python run_retrieval_eval.py --dry-run # offline smoke test of the harness only, not real quality numbers",
        "```",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Use fake embedders, no network required.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = run(dry_run=args.dry_run)

    suffix = "dry_run" if args.dry_run else "real"
    json_path = RESULTS_DIR / f"retrieval_eval_{suffix}.json"
    md_path = RESULTS_DIR / f"retrieval_eval_report_{suffix}.md"

    json_path.write_text(json.dumps(results, indent=2))
    write_markdown_report(results, md_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    for mode, data in results["per_mode"].items():
        print(f"{mode:16s} P@{K}={data['mean_precision_at_k']:.3f}  R@{K}={data['mean_recall_at_k']:.3f}  "
              f"latency={data['mean_latency_ms']:.1f}ms")
