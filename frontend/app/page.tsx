"use client";

import { useCallback, useEffect, useState } from "react";
import ChatPanel from "./components/ChatPanel";
import ChunkingGuide from "./components/ChunkingGuide";
import Doodles from "./components/Doodles";
import UploadPanel from "./components/UploadPanel";
import { UploadResponse } from "./lib/api";

type Recent = { name: string; doc: UploadResponse };
const RECENT_KEY = "rrag-recent";

function loadRecent(): Recent[] {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(RECENT_KEY) ?? "[]");
    return Array.isArray(parsed) ? parsed.slice(0, 5) : [];
  } catch {
    return [];
  }
}

export default function Home() {
  const [doc, setDoc] = useState<UploadResponse | null>(null);
  const [recent, setRecent] = useState<Recent[]>([]);

  useEffect(() => setRecent(loadRecent()), []);

  const remember = useCallback((next: UploadResponse, name: string) => {
    setDoc(next);
    setRecent((prev) => {
      const list = [{ name, doc: next }, ...prev.filter((r) => r.doc.document_id !== next.document_id)].slice(0, 5);
      try {
        window.localStorage.setItem(RECENT_KEY, JSON.stringify(list));
      } catch {
        /* private mode: the list just will not survive a reload */
      }
      return list;
    });
  }, []);

  return (
    <>
      <Doodles />
      <main className="container">
        <header className="hero">
          <span className="eyebrow">self-healing retrieval</span>
          <h1>
            Resilient<span className="hl">RAG</span>
          </h1>
          <p className="subtitle">
            Drop in a PDF. I read it, work out how it should be split, and tell you why. Ask a
            question and I show every search, every score, and every time I fix my own mistake.
          </p>
        </header>

        <section className="step">
          <h2><span className="num">1</span> Give me a file</h2>
          <UploadPanel onUploaded={remember} />
          {recent.length > 0 && (
            <div className="recents">
              <span>Already read:</span>
              {recent.map((r) => (
                <button
                  key={r.doc.document_id}
                  className={`recent${doc?.document_id === r.doc.document_id ? " on" : ""}`}
                  onClick={() => setDoc(r.doc)}
                  title={`${r.doc.num_chunks} chunks, ${r.doc.strategy_label}`}
                >
                  {r.name}
                </button>
              ))}
            </div>
          )}
        </section>

        {doc && (
          <section className="step">
            <h2><span className="num">2</span> Ask it something</h2>
            <ChatPanel doc={doc} />
          </section>
        )}

        <ChunkingGuide upload={doc} />

        <footer className="foot">
          Dense + BM25 hybrid search · cross-encoder rerank · two independent judges · Qdrant + Redis
        </footer>
      </main>
    </>
  );
}
