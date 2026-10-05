"use client";

import { useEffect, useState } from "react";

/** Reveals text like handwriting. Renders the full text at once when `instant` or reduced motion. */
export default function Typed({
  text,
  cps = 70,
  instant = false,
  onDone,
}: {
  text: string;
  cps?: number;
  instant?: boolean;
  onDone?: () => void;
}) {
  const [shown, setShown] = useState(instant ? text.length : 0);

  useEffect(() => {
    const reduced =
      typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (instant || reduced) {
      setShown(text.length);
      onDone?.();
      return;
    }
    setShown(0);
    const started = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const n = Math.min(text.length, Math.floor(((now - started) / 1000) * cps));
      setShown(n);
      if (n < text.length) raf = requestAnimationFrame(tick);
      else onDone?.();
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text, instant, cps]);

  return <>{text.slice(0, shown)}</>;
}
