"use client";

import { useEffect, useState } from "react";
import { ChunkingPolicy, UploadResponse, fetchChunking } from "../lib/api";

/**
 * Reference, not the main event: the live reasoning above is the real
 * answer. This lists what the agent can choose from and what it ignores.
 */
export default function ChunkingGuide({ upload }: { upload: UploadResponse | null }) {
  const [policy, setPolicy] = useState<ChunkingPolicy | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchChunking()
      .then((next) => !cancelled && setPolicy(next))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  if (!policy) return null;
  const picked = upload?.strategy_id ?? policy.strategy_id;

  return (
    <details className="guide">
      <summary>Field guide: the seven ways I can split a file</summary>
      <p className="lede">
        Chunk size and boundaries decide what search can find. Too small loses the sentence around
        an answer; too large mixes topics into one vector. Every upload gets measured first.
      </p>
      <div className="cards">
        {policy.strategies.map((row) => {
          const built = !["sentence_window", "parent_child"].includes(row.id);
          return (
            <article key={row.id} className={`card${row.id === picked ? " picked" : ""}${built ? "" : " unbuilt"}`}>
              <h3>{row.name}</h3>
              <p>{row.how}</p>
              <small>Best for: {row.best_for}</small>
              {row.id === picked && <span className="stamp">{upload ? "picked for your file" : "default"}</span>}
              {!built && <span className="stamp muted">not built yet</span>}
            </article>
          );
        })}
      </div>
      <div className="notes">
        <p><b>Size</b> {policy.chunk_size} characters, about {policy.approx_tokens} tokens. Smaller than the usual 300–500 token advice: on this project's benchmark it found the right passage far more often.</p>
        <p><b>Overlap</b> {policy.chunk_overlap} characters, {Math.round(policy.overlap_ratio * 1000) / 10}% (10–15% stops a sentence being lost at a boundary).</p>
        <p><b>Separators</b> <code>{policy.separators_display}</code></p>
      </div>
    </details>
  );
}
