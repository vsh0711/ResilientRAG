"use client";

import { useEffect, useState } from "react";
import type { ThinkState } from "./useEventPlayer";
import Typed from "./Typed";

const STAGES: [string, string][] = [
  ["read", "Reading"],
  ["profile", "Measuring"],
  ["probe", "Probing"],
  ["score", "Voting"],
  ["chunk", "Splitting"],
  ["embed", "Embedding"],
];

function pct(n: number) {
  return `${Math.round(n * 100)}%`;
}

function Vote({ rows }: { rows: NonNullable<ThinkState["scores"]> }) {
  const [grown, setGrown] = useState(false);
  useEffect(() => {
    const t = window.setTimeout(() => setGrown(true), 60);
    return () => window.clearTimeout(t);
  }, []);
  return (
    <div className="vote">
      <h3>The vote</h3>
      <ul>
        {rows.map((row, i) => (
          <li key={row.id} className={`vote-row verdict-${row.verdict.replace(" ", "-")}`}>
            <div className="vote-head">
              <span className="vote-name">{row.name}</span>
              <span className="vote-tag">{row.verdict}</span>
            </div>
            <div className="vote-bar">
              <span
                style={{
                  width: grown ? `${Math.max(row.score, 0.02) * 100}%` : "0%",
                  transitionDelay: `${i * 90}ms`,
                }}
              />
            </div>
            <p className="vote-reason">{row.reason}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ChunkStrip({ chunked }: { chunked: NonNullable<ThinkState["chunked"]> }) {
  const sizes = chunked.sizes ?? [];
  const max = Math.max(...sizes, 1);
  return (
    <div className="chunk-strip-wrap">
      <h3>
        {chunked.num_chunks} chunks <small>avg {chunked.avg} · smallest {chunked.min} · largest {chunked.max} chars</small>
      </h3>
      <div className="chunk-strip" aria-label="Chunk sizes">
        {sizes.map((size, i) => (
          <i
            key={i}
            style={{ height: `${14 + (size / max) * 54}px`, animationDelay: `${Math.min(i * 14, 900)}ms` }}
            title={`${size} characters`}
          />
        ))}
      </div>
      <blockquote className="chunk-sample">{chunked.sample}…</blockquote>
    </div>
  );
}

export default function ThinkingStream({
  fileName,
  state,
  playing,
  onSkip,
}: {
  fileName: string;
  state: ThinkState;
  playing: boolean;
  onSkip: () => void;
}) {
  const lastId = state.lines.length ? state.lines[state.lines.length - 1].id : 0;
  const active = state.stages[state.stages.length - 1];
  const p = state.profile;

  return (
    <section className="think-card" aria-live="polite">
      <header className="think-head">
        <div>
          <span className="eyebrow">{playing ? "thinking out loud" : "what I decided"}</span>
          <h2>{fileName}</h2>
        </div>
        {playing && (
          <button className="link-btn" onClick={onSkip}>
            skip the commentary
          </button>
        )}
      </header>

      <ol className="rail">
        {STAGES.map(([id, label]) => {
          const seen = state.stages.includes(id);
          return (
            <li key={id} className={seen ? (id === active && playing ? "on now" : "on") : ""}>
              {label}
            </li>
          );
        })}
      </ol>

      <div className="notebook">
        {state.lines.map((line) =>
          line.kind === "stage" ? (
            <p key={line.id} className="nb-stage">
              {line.text}
            </p>
          ) : (
            <p key={line.id} className="nb-line">
              <Typed text={line.text} instant={line.id !== lastId || !playing} />
              {line.id === lastId && playing && <span className="pencil" aria-hidden />}
            </p>
          )
        )}
        {playing && state.lines.length === 0 && <p className="nb-stage">Opening the file…</p>}
      </div>

      {p && (
        <dl className="stats">
          <div><dt>pages</dt><dd>{p.pages}</dd></div>
          <div><dt>words</dt><dd>{p.words.toLocaleString()}</dd></div>
          <div><dt>headings</dt><dd>{p.headings}</dd></div>
          <div><dt>avg paragraph</dt><dd>{p.avg_paragraph_chars}</dd></div>
          <div><dt>code lines</dt><dd>{pct(p.code_ratio)}</dd></div>
          <div><dt>digits</dt><dd>{pct(p.digit_ratio)}</dd></div>
        </dl>
      )}

      {state.scores && <Vote rows={state.scores} />}

      {state.decision && (
        <div className="verdict">
          <span className="eyebrow">my pick</span>
          <h3>{state.decision.label}</h3>
          <p>
            <Typed text={state.decision.why} instant={!playing} cps={110} />
          </p>
        </div>
      )}

      {state.chunked && <ChunkStrip chunked={state.chunked} />}

      {state.error && <p className="error">{state.error}</p>}
    </section>
  );
}
