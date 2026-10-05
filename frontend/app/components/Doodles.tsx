"use client";

import { useEffect, useMemo, useState } from "react";

type Piece = {
  id: string;
  d: string[];
  x: number; // % from left
  y: number; // % from top
  size: number;
  rot: number;
  color: string;
  delay: number;
  dur: number;
  fill?: boolean;
};

type Custom = { file: string; x: number; y: number; size?: number; rotate?: number };

/** A hand-drawn spiral. Seeded wobble so it looks the same on every render. */
function spiral(turns = 3.2): string {
  const pts: string[] = [];
  for (let i = 0; i <= 90; i++) {
    const t = (i / 90) * turns * Math.PI * 2;
    const r = 3 + (i / 90) * 42 + Math.sin(i * 1.7) * 0.8;
    pts.push(`${i === 0 ? "M" : "L"}${(50 + Math.cos(t) * r).toFixed(1)} ${(50 + Math.sin(t) * r).toFixed(1)}`);
  }
  return pts.join(" ");
}

const SHAPES: Record<string, string[]> = {
  spiral: [spiral()],
  star: ["M50 8 L61 38 L93 40 L68 60 L77 92 L50 74 L23 92 L32 60 L7 40 L39 38 Z"],
  arrow: ["M8 78 C 24 20, 62 14, 88 54", "M88 54 L72 52", "M88 54 L82 38"],
  magnifier: ["M62 40 a22 22 0 1 0 -0.1 0", "M77 56 L94 76", "M38 30 C 42 24, 48 22, 54 24"],
  bulb: [
    "M50 10 C 26 10, 20 38, 35 52 C 41 58, 41 64, 41 70 L59 70 C 59 64, 59 58, 65 52 C 80 38, 74 10, 50 10",
    "M42 79 L58 79", "M44 87 L56 87", "M50 0 L50 4", "M10 24 L16 28", "M90 24 L84 28",
  ],
  plane: ["M6 52 L94 10 L60 90 L48 58 Z", "M94 10 L48 58", "M6 52 L48 58"],
  page: [
    "M24 8 L62 8 L78 24 L78 92 L24 92 Z", "M62 8 L62 24 L78 24",
    "M34 42 L68 42", "M34 56 L68 56", "M34 70 L56 70",
  ],
  squiggle: ["M2 50 C 12 14, 24 86, 36 50 S 60 14, 72 50 S 90 86, 98 50"],
  sparkle: ["M50 6 L55 44 L94 50 L55 56 L50 94 L45 56 L6 50 L45 44 Z"],
  brackets: [
    "M32 14 L16 14 L16 86 L32 86", "M68 14 L84 14 L84 86 L68 86",
    "M36 38 L64 38", "M36 52 L64 52", "M36 66 L56 66",
  ],
  cloud: ["M24 72 C 6 72, 6 46, 28 46 C 28 24, 62 20, 68 42 C 90 38, 96 70, 74 72 Z"],
  dots: ["M10 50 L10.5 50", "M30 40 L30.5 40", "M50 50 L50.5 50", "M70 60 L70.5 60", "M90 50 L90.5 50"],
};

const PIECES: Piece[] = [
  { id: "p1", d: SHAPES.magnifier, x: 3, y: 12, size: 92, rot: -8, color: "var(--teal)", delay: 0, dur: 11 },
  { id: "p2", d: SHAPES.spiral, x: 88, y: 8, size: 110, rot: 12, color: "var(--violet)", delay: 1.2, dur: 13 },
  { id: "p3", d: SHAPES.bulb, x: 91, y: 44, size: 84, rot: 8, color: "var(--amber)", delay: 2.4, dur: 12 },
  { id: "p4", d: SHAPES.brackets, x: 2, y: 46, size: 80, rot: -4, color: "var(--coral)", delay: 3.1, dur: 10 },
  { id: "p5", d: SHAPES.plane, x: 84, y: 76, size: 96, rot: -14, color: "var(--teal)", delay: 0.8, dur: 12 },
  { id: "p6", d: SHAPES.page, x: 6, y: 78, size: 80, rot: 6, color: "var(--violet)", delay: 4.2, dur: 14 },
  { id: "p7", d: SHAPES.star, x: 22, y: 4, size: 44, rot: 10, color: "var(--amber)", delay: 5.0, dur: 9 },
  { id: "p8", d: SHAPES.squiggle, x: 70, y: 92, size: 120, rot: -3, color: "var(--coral)", delay: 2.0, dur: 11 },
  { id: "p9", d: SHAPES.sparkle, x: 94, y: 24, size: 38, rot: 0, color: "var(--violet)", delay: 6.0, dur: 8 },
  { id: "p10", d: SHAPES.arrow, x: 10, y: 30, size: 96, rot: 14, color: "var(--ink)", delay: 3.6, dur: 10 },
  { id: "p11", d: SHAPES.cloud, x: 78, y: 4, size: 90, rot: 0, color: "var(--teal)", delay: 7.2, dur: 15 },
  { id: "p12", d: SHAPES.dots, x: 40, y: 94, size: 70, rot: 0, color: "var(--ink)", delay: 1.6, dur: 9 },
  { id: "p13", d: SHAPES.sparkle, x: 4, y: 64, size: 30, rot: 0, color: "var(--amber)", delay: 4.8, dur: 8 },
  { id: "p14", d: SHAPES.star, x: 95, y: 62, size: 34, rot: -12, color: "var(--coral)", delay: 8.0, dur: 9 },
];

/**
 * Background layer: line doodles that draw and erase themselves, plus any
 * images the owner drops into public/doodles (see manifest.json there).
 * While the agent works, <body data-busy="1"> speeds the drawing up.
 */
export default function Doodles() {
  const [custom, setCustom] = useState<Custom[]>([]);

  useEffect(() => {
    let live = true;
    fetch("/doodles/manifest.json")
      .then((r) => (r.ok ? r.json() : []))
      .then((list: Custom[]) => live && Array.isArray(list) && setCustom(list))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, []);

  const pieces = useMemo(() => PIECES, []);

  return (
    <div className="doodles" aria-hidden="true">
      {pieces.map((p) => (
        <svg
          key={p.id}
          className="doodle"
          viewBox="0 0 100 100"
          style={{
            left: `${p.x}%`,
            top: `${p.y}%`,
            width: p.size,
            height: p.size,
            color: p.color,
            ["--rot" as string]: `${p.rot}deg`,
            ["--delay" as string]: `${p.delay}s`,
            ["--dur" as string]: `${p.dur}s`,
          }}
        >
          {p.d.map((d, i) => (
            <path key={i} d={d} pathLength={1} style={{ animationDelay: `calc(var(--delay) + ${i * 0.35}s)` }} />
          ))}
        </svg>
      ))}
      {custom.map((c) => (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          key={c.file}
          className="doodle-img"
          src={`/doodles/${c.file}`}
          alt=""
          style={{
            left: `${c.x}%`,
            top: `${c.y}%`,
            width: c.size ?? 140,
            transform: `rotate(${c.rotate ?? 0}deg)`,
          }}
        />
      ))}
    </div>
  );
}
