"use client";

import { useEffect, useState } from "react";
import { checkAccess, saveAccessCode } from "../lib/api";

/** Shows the page when the server is open or the saved code works; otherwise asks for a code. */
export default function AccessGate({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<"checking" | "locked" | "open" | "down">("checking");
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    checkAccess()
      .then((r) => setState(r.ok ? "open" : "locked"))
      .catch(() => setState("down"));
  }, []);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      const r = await checkAccess(code.trim());
      if (r.ok) {
        saveAccessCode(code.trim());
        setState("open");
      } else {
        setError("That code was not accepted.");
      }
    } catch {
      setState("down");
    }
  }

  if (state === "open") return <>{children}</>;
  if (state === "checking") return <p className="gate-note">One moment…</p>;
  if (state === "down") {
    return <p className="gate-note error">The API is not reachable. Start the backend, Qdrant and Redis, then reload.</p>;
  }
  return (
    <form className="gate" onSubmit={submit}>
      <span className="eyebrow">private beta</span>
      <h1>Resilient<span className="hl">RAG</span></h1>
      <p className="subtitle">Enter your access code to continue.</p>
      <div className="ask-row">
        <input
          className="question-input"
          type="password"
          autoComplete="off"
          aria-label="Access code"
          value={code}
          onChange={(e) => setCode(e.target.value)}
        />
        <button type="submit" disabled={!code.trim()}>Enter</button>
      </div>
      {error && <p className="error">{error}</p>}
    </form>
  );
}
