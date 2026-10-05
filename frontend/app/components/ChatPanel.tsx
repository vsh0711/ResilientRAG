"use client";

import { useEffect, useRef, useState } from "react";
import { askQuestionStream, QueryEvent, QueryResponse, UploadResponse } from "../lib/api";
import Typed from "./Typed";

type Beat = Exclude<QueryEvent, { type: "start" | "result" | "error" }>;

const MODE_LABEL: Record<string, string> = {
  dense: "dense vectors",
  dense_rerank: "dense vectors + cross-encoder rerank",
  hybrid: "dense + BM25, fused",
  hybrid_rerank: "dense + BM25, fused, reranked",
};

const REASON_LABEL: Record<string, string> = {
  irrelevant_docs: "the retrieved passages were about something else",
  missing_context: "the passages were on topic but incomplete",
  unfaithful_answer: "the answer claimed more than the passages support",
  judge_unavailable: "the judge could not run (rate limit or outage), so this answer is unverified",
  none: "all good",
};

function Bold({ text }: { text: string }) {
  const parts = text.split(/\*\*(.+?)\*\*/g);
  return (
    <>
      {parts.map((part, i) => (i % 2 ? <strong key={i}>{part}</strong> : <span key={i}>{part}</span>))}
    </>
  );
}

function Meter({ label, value }: { label: string; value: number }) {
  const tone = value >= 0.8 ? "good" : value >= 0.5 ? "mid" : "bad";
  return (
    <div className={`meter tone-${tone}`}>
      <span className="meter-label">{label}</span>
      <span className="meter-track">
        <i style={{ width: `${Math.round(value * 100)}%` }} />
      </span>
      <strong>{value.toFixed(2)}</strong>
    </div>
  );
}

function BeatView({ beat, live }: { beat: Beat; live: boolean }) {
  switch (beat.type) {
    case "retrieve":
      return (
        <li className="beat beat-retrieve">
          <span className="beat-tag">attempt {beat.attempt} · search</span>
          <p>
            Looking up {beat.budget} passages with {MODE_LABEL[beat.mode] ?? beat.mode}
            {beat.attempt > 1 && <> for “{beat.query}”</>}.
          </p>
          <ul className="snips">
            {beat.docs.map((d, i) => (
              <li key={i}>{d.replace(/\s+/g, " ")}…</li>
            ))}
          </ul>
        </li>
      );
    case "generate":
      return (
        <li className="beat beat-generate">
          <span className="beat-tag">attempt {beat.attempt} · draft{beat.cached ? " · from cache" : ""}</span>
          <p className="draft">
            <Typed text={beat.answer.replace(/\*\*/g, "")} instant={!live} cps={160} />
          </p>
        </li>
      );
    case "score":
      return (
        <li className={`beat beat-score ${beat.passed ? "pass" : "fail"}`}>
          <span className="beat-tag">attempt {beat.attempt} · two judges</span>
          <Meter label="right passages?" value={beat.relevance} />
          <Meter label="stays in the source?" value={beat.faithfulness} />
          <p>
            {beat.passed
              ? `Combined ${beat.combined.toFixed(2)}, above the ${beat.threshold} bar. Done.`
              : `Combined ${beat.combined.toFixed(2)}, below ${beat.threshold}: ${REASON_LABEL[beat.failure_reason] ?? beat.failure_reason}.`}
          </p>
        </li>
      );
    case "heal":
      return (
        <li className="beat beat-heal">
          <span className="beat-tag">healing · retry {beat.step.retry_number}</span>
          <p>{beat.step.action_taken}</p>
        </li>
      );
  }
}

export default function ChatPanel({ doc }: { doc: UploadResponse }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [beats, setBeats] = useState<Beat[]>([]);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    document.body.dataset.busy = loading ? "1" : "0";
    return () => {
      document.body.dataset.busy = "0";
    };
  }, [loading]);

  async function handleAsk() {
    const q = question.trim();
    if (!q || loading) return;
    setLoading(true);
    setElapsed(0);
    setError(null);
    setResult(null);
    setBeats([]);
    const started = Date.now();
    const timer = window.setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 500);
    abort.current = new AbortController();
    try {
      await askQuestionStream(
        doc.document_id,
        q,
        (ev) => {
          if (ev.type === "result") setResult(ev.result);
          else if (ev.type === "error") setError(ev.detail);
          else if (ev.type !== "start") setBeats((b) => [...b, ev]);
        },
        3,
        abort.current.signal
      );
    } catch (err) {
      if (!(err instanceof DOMException && err.name === "AbortError")) {
        setError(err instanceof Error ? err.message : "Query failed");
      }
    } finally {
      window.clearInterval(timer);
      setLoading(false);
    }
  }

  const firstScore = beats.find((b) => b.type === "score");
  const healed = result && result.retry_count > 0 && firstScore && firstScore.type === "score";

  return (
    <div className="chat-panel">
      <div className="ask-row">
        <input
          className="question-input"
          placeholder="Ask anything the file should be able to answer"
          value={question}
          maxLength={2000}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && void handleAsk()}
        />
        {loading ? (
          <button className="ghost" onClick={() => abort.current?.abort()}>
            Stop · {elapsed}s
          </button>
        ) : (
          <button onClick={() => void handleAsk()} disabled={!question.trim()}>
            Ask
          </button>
        )}
      </div>

      {error && <p className="error">{error}</p>}

      {beats.length > 0 && (
        <ol className="beats">
          {beats.map((beat, i) => (
            <BeatView key={i} beat={beat} live={loading && i === beats.length - 1} />
          ))}
          {loading && (
            <li className="beat beat-wait">
              <span className="pencil" aria-hidden /> working…
            </li>
          )}
        </ol>
      )}

      {result && (
        <div className="result">
          <span className="eyebrow">answer</span>
          <p className="answer">
            <Bold text={result.answer} />
          </p>

          <div className="meters">
            <Meter label="confidence" value={result.final_score} />
            <Meter label="right passages" value={result.relevance_score} />
            <Meter label="grounded" value={result.faithfulness_score} />
          </div>

          {healed && firstScore.type === "score" && (
            <p className="healed">
              First attempt scored {firstScore.combined.toFixed(2)}. After {result.retry_count}{" "}
              {result.retry_count === 1 ? "retry" : "retries"} it scored {result.final_score.toFixed(2)}.
            </p>
          )}

          <div className="chips">
            {Object.entries(result.latency_ms).map(([k, v]) => (
              <span key={k} className="chip">{k} {Math.round(v)} ms</span>
            ))}
            <span className="chip">
              {Object.values(result.token_usage).reduce((a, t) => a + t.prompt_tokens + t.completion_tokens, 0)} tokens
            </span>
          </div>

          {result.sources?.length > 0 && (
            <details className="sources">
              <summary>{result.sources.length} passages this answer came from</summary>
              {result.sources.map((s, i) => (
                <blockquote key={i}>{s}</blockquote>
              ))}
            </details>
          )}
        </div>
      )}
    </div>
  );
}

