"use client";

import { useState } from "react";
import { askQuestion, QueryResponse, UploadResponse } from "../lib/api";
import HealingTrace from "./HealingTrace";

export default function ChatPanel({ document }: { document: UploadResponse }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);

  async function handleAsk() {
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await askQuestion(document.document_id, question);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="chat-panel">
      <div className="ask-row">
        <input
          className="question-input"
          placeholder="e.g. What is hybrid retrieval?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleAsk()}
        />
        <button onClick={handleAsk} disabled={loading}>
          {loading ? "Thinking…" : "Ask"}
        </button>
      </div>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="result">
          <h2>Answer</h2>
          <p className="answer">{result.answer}</p>

          <div className="scores">
            <ScorePill label="Combined" value={result.final_score} />
            <ScorePill label="Relevance" value={result.relevance_score} />
            <ScorePill label="Faithfulness" value={result.faithfulness_score} />
          </div>

          <HealingTrace steps={result.healing_trace} />
        </div>
      )}
    </div>
  );
}

function ScorePill({ label, value }: { label: string; value: number }) {
  const cls = value >= 0.8 ? "pill-good" : value >= 0.5 ? "pill-mid" : "pill-bad";
  return (
    <div className={`pill ${cls}`}>
      <span>{label}</span>
      <strong>{value.toFixed(2)}</strong>
    </div>
  );
}
