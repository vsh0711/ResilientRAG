"""
Load test: what 100 real people look like to the API.

Each simulated user pauses 15 to 45 seconds between actions, the way a person
reads an answer before asking the next question. Mix per user action:

    70%  ask a question (full self-healing pipeline, real Groq calls)
    10%  ask a question over the streaming endpoint
    10%  open the page's policy endpoint
     5%  upload a PDF
     5%  poll /health/ready

Questions are unique (taken from the benchmark), so the Redis answer cache
does not flatter the numbers. One document is uploaded once at test start and
shared by every user.

    cd backend
    uv run --extra dev locust -f ../loadtest/locustfile.py --headless \
        -u 100 -r 5 -t 4m --host http://localhost:8000 \
        --csv ../eval/results/load --html ../eval/results/load.html
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

from locust import HttpUser, between, events, task

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = [json.loads(l) for l in open(ROOT / "eval/bench/questions.jsonl") if l.strip()]
QUESTIONS = [q for q in QUESTIONS if q["doc"] == "manual"]
FORMS = [(q, f) for q in QUESTIONS for f in ("direct", "paraphrase")]
random.Random(3).shuffle(FORMS)
_next = itertools.cycle(FORMS)

SMALL_PDF = ROOT / "eval/bench/docs/fieldnotes.pdf"
state: dict = {"doc": None}


@events.test_start.add_listener
def upload_shared_document(environment, **_):
    import requests

    with open(ROOT / "eval/bench/docs/manual.pdf", "rb") as f:
        r = requests.post(f"{environment.host}/documents", files={"file": ("manual.pdf", f, "application/pdf")}, timeout=300)
    r.raise_for_status()
    state["doc"] = r.json()["document_id"]
    print(f"shared document: {state['doc']} ({r.json()['strategy_id']})")


class Person(HttpUser):
    wait_time = between(15, 45)

    def next_question(self) -> str:
        q, form = next(_next)
        return q[form]

    @task(70)
    def ask(self):
        with self.client.post(
            "/query",
            json={"document_id": state["doc"], "question": self.next_question()},
            name="POST /query",
            catch_response=True,
            timeout=200,
        ) as r:
            if r.status_code == 429:
                r.failure("429 busy")
            elif r.status_code != 200:
                r.failure(f"{r.status_code}")

    @task(10)
    def ask_stream(self):
        with self.client.post(
            "/query/stream",
            json={"document_id": state["doc"], "question": self.next_question()},
            name="POST /query/stream",
            catch_response=True,
            stream=True,
            timeout=200,
        ) as r:
            if r.status_code != 200:
                r.failure(f"{r.status_code}")
                return
            last = ""
            for line in r.iter_lines():
                if line.startswith(b"data:"):
                    last = line.decode()
            if '"type": "result"' not in last:
                r.failure("stream ended without a result")

    @task(10)
    def policy(self):
        self.client.get("/documents/chunking", name="GET /documents/chunking")

    @task(5)
    def upload(self):
        with open(SMALL_PDF, "rb") as f:
            self.client.post(
                "/documents", files={"file": ("notes.pdf", f, "application/pdf")},
                name="POST /documents", timeout=200,
            )

    @task(5)
    def ready(self):
        self.client.get("/health/ready", name="GET /health/ready")
