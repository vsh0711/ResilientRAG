"use client";

import { useEffect, useRef, useState } from "react";
import { uploadDocumentStream, UploadResponse } from "../lib/api";
import ThinkingStream from "./ThinkingStream";
import { useEventPlayer } from "./useEventPlayer";

export default function UploadPanel({
  onUploaded,
}: {
  onUploaded: (doc: UploadResponse, fileName: string) => void;
}) {
  const [fileName, setFileName] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [over, setOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const nameRef = useRef<string>("");
  const player = useEventPlayer((doc) => onUploaded(doc, nameRef.current));

  useEffect(() => {
    document.body.dataset.busy = busy || player.playing ? "1" : "0";
    return () => {
      document.body.dataset.busy = "0";
    };
  }, [busy, player.playing]);

  async function start(file: File) {
    player.reset();
    setFileName(file.name);
    nameRef.current = file.name;
    setBusy(true);
    try {
      await uploadDocumentStream(file, player.push);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Upload failed";
      player.push({ type: "error", detail: message });
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  function onChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) void start(file);
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file && !busy) void start(file);
  }

  return (
    <div className="upload-panel">
      <label
        className={`dropzone${over ? " over" : ""}${busy ? " busy" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          onChange={onChange}
          className="file-input"
          disabled={busy}
        />
        <span className="drop-title">{busy ? "Reading your file…" : "Drop a PDF here, or click to choose one"}</span>
        <span className="drop-sub">Up to 20 MB. I read it first, then tell you how I plan to split it.</span>
      </label>

      {fileName && (
        <ThinkingStream
          fileName={fileName}
          state={player.state}
          playing={player.playing}
          onSkip={player.skip}
        />
      )}
    </div>
  );
}
