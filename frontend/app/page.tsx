"use client";

import { useState } from "react";
import ChatPanel from "./components/ChatPanel";
import UploadPanel from "./components/UploadPanel";
import { UploadResponse } from "./lib/api";

export default function Home() {
  const [doc, setDoc] = useState<UploadResponse | null>(null);

  return (
    <main className="container">
      <header>
        <h1>ResilientRAG</h1>
        <p className="subtitle">
          A self-healing RAG agent: when it detects a bad answer, it diagnoses why and
          automatically escalates its retrieval strategy before trying again.
        </p>
      </header>

      <section className="panel">
        <h2>1. Upload a PDF</h2>
        <UploadPanel onUploaded={setDoc} />
        {doc && (
          <p className="muted">
            Indexed {doc.num_chunks} chunks from ~{doc.num_pages_estimate} page(s).
          </p>
        )}
      </section>

      {doc && (
        <section className="panel">
          <h2>2. Ask a question</h2>
          <ChatPanel document={doc} />
        </section>
      )}
    </main>
  );
}
