"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import AccessGate from "./components/AccessGate";
import ChatPanel from "./components/ChatPanel";
import ChunkingGuide from "./components/ChunkingGuide";
import Doodles from "./components/Doodles";
import UploadPanel from "./components/UploadPanel";
import { releaseDocument, UploadResponse } from "./lib/api";

export default function Home() {
  const [doc, setDoc] = useState<UploadResponse | null>(null);
  const current = useRef<string | null>(null);

  const onUploaded = useCallback((next: UploadResponse) => {
    // A new file replaces the old one: let the server drop whatever is no longer in use.
    if (current.current && current.current !== next.document_id) releaseDocument(current.current);
    current.current = next.document_id;
    setDoc(next);
  }, []);

  // Reloading or closing the page ends this tab's use of the document. A reload
  // starts with no document, so the server deletes it unless another tab shares it.
  useEffect(() => {
    const leave = () => {
      if (current.current) releaseDocument(current.current);
    };
    window.addEventListener("pagehide", leave);
    return () => window.removeEventListener("pagehide", leave);
  }, []);

  return (
    <>
      <Doodles />
      <AccessGate>
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
          <UploadPanel onUploaded={onUploaded} />
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
      </AccessGate>
    </>
  );
}
