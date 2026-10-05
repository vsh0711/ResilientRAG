"use client";

import { useRef, useState } from "react";
import { uploadDocument, UploadResponse } from "../lib/api";

export default function UploadPanel({
  onUploaded,
}: {
  onUploaded: (doc: UploadResponse) => void;
}) {
  const [status, setStatus] = useState<"idle" | "uploading" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileName(file.name);
    setStatus("uploading");
    setError(null);
    try {
      const doc = await uploadDocument(file);
      setStatus("idle");
      onUploaded(doc);
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "Upload failed");
    }
  }

  return (
    <div className="upload-panel">
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        onChange={handleFileChange}
        className="file-input"
      />
      {status === "uploading" && <p className="muted">Processing {fileName}…</p>}
      {status === "error" && <p className="error">{error}</p>}
    </div>
  );
}
