"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { DocProfile, StrategyScore, UploadEvent, UploadResponse } from "../lib/api";

export interface ThinkLine {
  id: number;
  kind: "stage" | "think";
  text: string;
}

export interface ThinkState {
  stages: string[];
  lines: ThinkLine[];
  profile?: DocProfile;
  scores?: StrategyScore[];
  decision?: { strategy_id: string; label: string; why: string };
  chunked?: { num_chunks: number; avg: number; min: number; max: number; sample: string; sizes: number[] };
  upload?: UploadResponse;
  error?: string;
}

export const EMPTY: ThinkState = { stages: [], lines: [] };

function apply(state: ThinkState, ev: UploadEvent, id: number): ThinkState {
  switch (ev.type) {
    case "stage":
      return {
        ...state,
        stages: state.stages.includes(ev.stage) ? state.stages : [...state.stages, ev.stage],
        lines: [...state.lines, { id, kind: "stage", text: ev.text }],
      };
    case "think":
      return { ...state, lines: [...state.lines, { id, kind: "think", text: ev.text }] };
    case "reused":
      return state;
    case "profile":
      return { ...state, profile: ev.profile };
    case "scores":
      return { ...state, scores: ev.rows };
    case "decision":
      return { ...state, decision: { strategy_id: ev.strategy_id, label: ev.label, why: ev.why } };
    case "chunked":
      return { ...state, chunked: ev };
    case "done":
      return { ...state, upload: ev.upload };
    case "error":
      return { ...state, error: ev.detail };
  }
}

function delayFor(ev: UploadEvent): number {
  switch (ev.type) {
    case "stage":
      return 650;
    case "think":
      return Math.min(1700, 320 + ev.text.length * 15);
    case "scores":
      return 1500;
    case "decision":
      return Math.min(4200, 700 + ev.why.length * 14);
    case "chunked":
      return 1500;
    default:
      return 0;
  }
}

/**
 * The server finishes its analysis in well under a second. Playing the
 * events back at reading speed is what makes the reasoning followable.
 * `skip()` drains the queue immediately.
 */
export function useEventPlayer(onUpload: (u: UploadResponse) => void) {
  const [state, setState] = useState<ThinkState>(EMPTY);
  const [playing, setPlaying] = useState(false);
  const queue = useRef<UploadEvent[]>([]);
  const running = useRef(false);
  const nextId = useRef(1);
  const timer = useRef<number | null>(null);
  const fast = useRef(false);
  const onUploadRef = useRef(onUpload);
  onUploadRef.current = onUpload;

  const step = useCallback(() => {
    const ev = queue.current.shift();
    if (!ev) {
      running.current = false;
      setPlaying(false);
      return;
    }
    // A file the server has already read: no point narrating at reading speed.
    if (ev.type === "reused") fast.current = true;
    setState((s) => apply(s, ev, nextId.current++));
    if (ev.type === "done") onUploadRef.current(ev.upload);
    const wait = fast.current ? 0 : delayFor(ev);
    timer.current = window.setTimeout(step, wait);
  }, []);

  const push = useCallback(
    (ev: UploadEvent) => {
      queue.current.push(ev);
      if (!running.current) {
        running.current = true;
        setPlaying(true);
        step();
      }
    },
    [step]
  );

  const skip = useCallback(() => {
    fast.current = true;
  }, []);

  const reset = useCallback(() => {
    if (timer.current) window.clearTimeout(timer.current);
    queue.current = [];
    running.current = false;
    fast.current = false;
    nextId.current = 1;
    setState(EMPTY);
    setPlaying(false);
  }, []);

  useEffect(() => () => {
    if (timer.current) window.clearTimeout(timer.current);
  }, []);

  return { state, playing, push, skip, reset };
}
