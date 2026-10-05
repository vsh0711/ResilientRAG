"""
Adaptive chunking: read the uploaded file, decide how to split it, say why.

The strategy table is general knowledge about chunking. What this module
adds is the decision: it measures the file in front of it (headings,
code, tables, paragraph length, topic shifts), scores each strategy
against those measurements, and emits every step as an event so the UI
can show the reasoning while it happens.

Built strategies: fixed, recursive, structure, semantic, code.
Described but not built: sentence_window and parent_child. Both embed one
text and return another, which needs a second index. They are scored and
shown so the choice is honest, but never selected.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

STRATEGY_ID = "recursive_character"  # default / fallback
STRATEGY_LABEL = "Recursive character"

# Tried in order. ". " rather than a bare period so decimals survive.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
SEPARATORS_DISPLAY = '\\n\\n  →  \\n  →  ". "  →  " "'
CODE_SEPARATORS = ["\nclass ", "\ndef ", "\nfunction ", "\n\n", "\n", " ", ""]

STRATEGIES: list[dict] = [
    {"id": "fixed", "name": "Fixed-size", "built": True,
     "how": "Split every N characters, with overlap",
     "best_for": "Quick prototypes, uniform text"},
    {"id": STRATEGY_ID, "name": STRATEGY_LABEL, "built": True,
     "how": "Splits on paragraph, then sentence, then word",
     "best_for": "General-purpose default"},
    {"id": "semantic", "name": "Semantic chunking", "built": True,
     "how": "Splits where embedding similarity drops between sentences",
     "best_for": "Topic-shifting documents"},
    {"id": "structure", "name": "Document-structure-aware", "built": True,
     "how": "Splits on headings, markdown, and HTML tags",
     "best_for": "Docs with clear structure (wikis, manuals)"},
    {"id": "sentence_window", "name": "Sentence window", "built": False,
     "how": "Embed single sentences, retrieve with a surrounding window",
     "best_for": "High-precision Q&A"},
    {"id": "parent_child", "name": "Parent-child chunking", "built": False,
     "how": "Small chunk for retrieval, larger parent for context",
     "best_for": "Balancing precision and context"},
    {"id": "code", "name": "Code-aware chunking", "built": True,
     "how": "Splits at function and class boundaries",
     "best_for": "Codebases, technical docs"},
]
_BY_ID = {row["id"]: row for row in STRATEGIES}
BUILT_IDS = [row["id"] for row in STRATEGIES if row["built"]]

# ---------------------------------------------------------------------------
# Text cleanup and measurement
# ---------------------------------------------------------------------------

_MD_HEADING = re.compile(r"^#{1,6}\s+\S")
_NUM_HEADING = re.compile(r"^(\d+(\.\d+){0,3}|[A-Z]|[IVX]+)[.)]?\s+[A-Z][^.!?]{2,70}$")
_BULLET = re.compile(r"^\s*([-*•▪◦]|\d+[.)])\s+")
_CODE_LINE = re.compile(
    r"^\s*(def \w+\(|class \w+.*[:{(]\s*$|import [\w.]+(\s+as \w+)?\s*$|from [\w.]+ import |"
    r"function \w*\(|(const|let|var) \w+\s*=|(public|private|protected) [\w<>\[\]]+ \w+.*[({;]\s*$|"
    r"#include|(if|for|while) \(.*\)\s*\{?\s*$|return .*;\s*$|[{}]\s*$)"
    r"|=>|\w\(.*\);\s*$"
)
_INDENTED = re.compile(r"^(\s{2,}|\t)\S")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def is_heading(line: str) -> bool:
    text = line.strip()
    if not 3 <= len(text) <= 80 or text.endswith((",", ";")):
        return False
    if _MD_HEADING.match(text):
        return True
    if _BULLET.match(text) and not _NUM_HEADING.match(text):
        return False
    words = text.split()
    if len(words) > 10:
        return False
    if _NUM_HEADING.match(text) and not text.endswith("."):
        return True
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 4 and text.isupper() and len(words) >= 1:
        return True
    if text.endswith((".", "?", "!")):
        return False
    big = [w for w in words if len(w) > 3]
    if not big or len(words) > 8:
        return False
    return all(w[0].isupper() for w in big) and len(words) >= 2


def normalize_text(pages: list[str]) -> str:
    """Rebuild paragraphs from PDF line breaks.

    PDF extraction breaks lines at the page margin, not at paragraph ends.
    A line that runs near full width and has no closing punctuation is
    joined to the next one; blank lines and headings stay as breaks.
    """
    raw_lines: list[str] = []
    for page in pages:
        page = page.replace("\r", "").replace("-\n", "")
        raw_lines.extend(page.split("\n"))
        raw_lines.append("")
    lengths = [len(l.strip()) for l in raw_lines if len(l.strip()) > 25]
    wide = (statistics.median(lengths) * 0.8) if lengths else 60

    paragraphs: list[str] = []
    buf = ""        # prose being joined
    code: list[str] = []  # a run of consecutive code lines, kept line by line

    def flush() -> None:
        nonlocal buf, code
        if buf:
            paragraphs.append(buf)
            buf = ""
        if code:
            paragraphs.append("\n".join(code))
            code = []

    for line in raw_lines:
        text = line.strip()
        if not text:
            flush()
            continue
        if _CODE_LINE.search(line) or (code and _INDENTED.match(line)):
            if buf:
                paragraphs.append(buf)
                buf = ""
            code.append(line.rstrip())
            continue
        if code:
            paragraphs.append("\n".join(code))
            code = []
        if is_heading(text):
            flush()
            paragraphs.append(text)
        elif _BULLET.match(text):
            flush()
            paragraphs.append(text)
        elif buf:
            prev_wide = len(buf.rsplit("\n", 1)[-1]) >= wide
            if prev_wide and (not buf.endswith((".", "!", "?", ":")) or text[:1].islower()):
                buf = f"{buf} {text}"
            else:
                paragraphs.append(buf)
                buf = text
        else:
            buf = text
    flush()
    return "\n\n".join(p for p in paragraphs if p.strip())


@dataclass
class DocProfile:
    pages: int
    chars: int
    words: int
    paragraphs: int
    avg_paragraph_chars: int
    headings: int
    heading_density: float  # per 1,000 words
    heading_pages: int
    code_ratio: float
    list_ratio: float
    digit_ratio: float
    avg_sentence_words: float
    sample_headings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def profile_document(pages: list[str], clean: str) -> DocProfile:
    paragraphs = [p for p in clean.split("\n\n") if p.strip()]
    lines = [l for l in clean.split("\n") if l.strip()]
    words = len(clean.split())
    headings = [p for p in paragraphs if is_heading(p) and "\n" not in p]
    heading_set = {h.strip() for h in headings}
    pages_with = sum(
        1 for page in pages if any(line.strip() in heading_set for line in page.split("\n"))
    )
    code_lines = 0
    in_code = False
    for l in lines:
        in_code = bool(_CODE_LINE.search(l)) or (in_code and bool(_INDENTED.match(l)))
        code_lines += in_code
    list_lines = sum(1 for l in lines if _BULLET.match(l) and l.strip() not in heading_set)
    chars = len(clean)
    digits = sum(c.isdigit() for c in clean)
    body = [p for p in paragraphs if p not in heading_set]
    sentences = [s for p in body for s in _SENTENCE_END.split(p) if s.strip()]
    avg_sentence = (
        sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0.0
    )
    return DocProfile(
        pages=len(pages),
        chars=chars,
        words=words,
        paragraphs=len(paragraphs),
        avg_paragraph_chars=round(chars / max(len(paragraphs), 1)),
        headings=len(headings),
        heading_density=len(headings) / max(words / 1000, 0.001),
        heading_pages=pages_with,
        code_ratio=code_lines / max(len(lines), 1),
        list_ratio=list_lines / max(len(lines), 1),
        digit_ratio=digits / max(chars, 1),
        avg_sentence_words=avg_sentence,
        sample_headings=headings[:4],
    )


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------

def _clean(chunks: list[str]) -> list[str]:
    return [c.strip() for c in chunks if c and c.strip()]


def split_fixed(text: str, size: int, overlap: int) -> list[str]:
    step = max(size - overlap, 1)
    return _clean([text[i : i + size] for i in range(0, len(text), step)])


def split_recursive(text: str, size: int, overlap: int, separators=None) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=separators or SEPARATORS
    )
    return _clean(splitter.split_text(text))


def split_structure(text: str, size: int, overlap: int, min_chars: int = 300) -> list[str]:
    """One chunk per section. Long sections are split, each part keeps its heading."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for para in text.split("\n\n"):
        if is_heading(para) and "\n" not in para:
            sections.append((para.strip(), []))
        else:
            sections[-1][1].append(para)

    out: list[str] = []
    carry = ""
    for heading, body in sections:
        content = "\n\n".join(body).strip()
        if not heading and not content:
            continue
        title = heading.lstrip("# ").strip()
        block = f"{title}\n{content}".strip() if title else content
        if carry:
            block = f"{carry}\n\n{block}"
            carry = ""
        if len(block) < min_chars:
            carry = block
            continue
        if len(block) <= size:
            out.append(block)
            continue
        parts = split_recursive(content, size - len(title) - 1, overlap)
        out.extend(f"{title}\n{part}" if title else part for part in parts)
    if carry:
        if out and len(out[-1]) + len(carry) <= size * 1.25:
            out[-1] = f"{out[-1]}\n\n{carry}"
        else:
            out.append(carry)
    return _clean(out)


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for para in text.split("\n\n"):
        out.extend(s.strip() for s in _SENTENCE_END.split(para) if s.strip())
    return out


def split_semantic(
    text: str,
    size: int,
    overlap: int,
    embed_fn: Callable[[list[str]], np.ndarray],
    percentile: float = 80.0,
    min_chars: int = 400,
) -> list[str]:
    """Break where the meaning moves most. Needs one embedding per sentence."""
    sentences = _sentences(text)
    if len(sentences) < 4:
        return split_recursive(text, size, overlap)
    vecs = embed_fn(sentences)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.clip(norms, 1e-9, None)
    # Compare sliding 3-sentence windows so one odd sentence does not cut.
    window = np.array([vecs[max(0, i - 1) : i + 2].mean(axis=0) for i in range(len(vecs))])
    drops = np.array([1 - float(window[i] @ window[i + 1]) for i in range(len(window) - 1)])
    # A break is a local peak of the drop, above the chosen percentile. Without
    # the floor, a text where most neighbours are identical puts the percentile
    # at zero and every sentence boundary would qualify.
    floor = max(float(np.percentile(drops, percentile)), 1e-3)
    peaks = {
        i
        for i in range(len(drops))
        if drops[i] >= floor
        and drops[i] >= drops[max(i - 1, 0)]
        and drops[i] >= drops[min(i + 1, len(drops) - 1)]
    }

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for i, sent in enumerate(sentences):
        cur.append(sent)
        cur_len += len(sent) + 1
        if cur_len >= size or (i in peaks and cur_len >= min_chars):
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
    if cur:
        if chunks and cur_len < min_chars // 2:
            chunks[-1] += " " + " ".join(cur)
        else:
            chunks.append(" ".join(cur))
    final: list[str] = []
    for chunk in chunks:
        final.extend(split_recursive(chunk, size, overlap) if len(chunk) > size * 1.3 else [chunk])
    return _clean(final)


def apply_strategy(
    strategy_id: str,
    text: str,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    chunk_size: int | None = None,
) -> list[str]:
    s = get_settings()
    size = chunk_size or s.chunk_size
    overlap = round(size * s.chunk_overlap / s.chunk_size)  # keep the overlap ratio
    if strategy_id == "fixed":
        return split_fixed(text, size, overlap)
    if strategy_id == "structure":
        return split_structure(text, size, overlap)
    if strategy_id == "code":
        return split_recursive(text, size, overlap, CODE_SEPARATORS)
    if strategy_id == "semantic":
        if embed_fn is None:
            raise ValueError("semantic chunking needs an embedding function")
        return split_semantic(text, size, overlap, embed_fn)
    return split_recursive(text, size, overlap)


# ---------------------------------------------------------------------------
# Topic-shift probe and scoring
# ---------------------------------------------------------------------------

def topic_shift_ratio(clean: str, embed_fn: Callable[[list[str]], np.ndarray], cap: int = 48):
    """Embed a spread of paragraphs; return (fraction of sharp drops, mean sim, n)."""
    paras = [p[:600] for p in clean.split("\n\n") if len(p) > 120 and not is_heading(p)]
    if len(paras) < 6:
        return None
    if len(paras) > cap:
        idx = np.linspace(0, len(paras) - 1, cap).astype(int)
        paras = [paras[i] for i in idx]
    vecs = embed_fn(paras)
    vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)
    sims = np.array([float(vecs[i] @ vecs[i + 1]) for i in range(len(vecs) - 1)])
    return float((sims < 0.30).mean()), float(sims.mean()), len(paras)


@dataclass
class StrategyScore:
    id: str
    name: str
    built: bool
    score: float
    verdict: str  # "chosen" | "runner-up" | "ruled out" | "not built"
    reason: str


def score_strategies(profile: DocProfile, shift: tuple | None) -> list[StrategyScore]:
    p = profile
    raw: dict[str, tuple[float, str]] = {}

    if p.headings >= 4:
        s = min(1.0, p.heading_density / 5.0) * 0.8 + 0.2 * (p.heading_pages / max(p.pages, 1))
        raw["structure"] = (s, f"{p.headings} headings across {p.heading_pages} of {p.pages} pages "
                               f"({p.heading_density:.1f} per 1,000 words), so sections are real boundaries")
    else:
        raw["structure"] = (0.05, (f"only {p.headings} heading-like lines found" if p.headings else "no heading-like lines found") + ", nothing to split on")

    if p.code_ratio >= 0.10:
        raw["code"] = (min(1.0, p.code_ratio / 0.25), f"{p.code_ratio:.0%} of lines look like code")
    else:
        raw["code"] = (0.0, f"{p.code_ratio:.0%} of lines look like code, this is not a codebase")

    if shift is None:
        raw["semantic"] = (0.2, "too few long paragraphs to measure topic shifts")
    else:
        ratio, mean_sim, n = shift
        s = min(1.0, ratio / 0.25) * 0.9
        if p.headings >= 4:
            s *= 0.6  # headings already mark the topic changes, for free
        raw["semantic"] = (s, f"{ratio:.0%} of {n - 1} neighbouring paragraph pairs drop below 0.30 "
                              f"similarity (mean {mean_sim:.2f})")

    unstructured = p.avg_paragraph_chars > 3500 or p.digit_ratio > 0.30 or p.avg_sentence_words < 4
    if unstructured:
        why = (f"{p.digit_ratio:.0%} digits and no usable sentence structure"
               if p.digit_ratio > 0.30 or p.avg_sentence_words < 4
               else f"paragraphs average {p.avg_paragraph_chars} characters, no breaks to follow")
        raw["fixed"] = (0.7, why)
    else:
        raw["fixed"] = (0.15, "the text has paragraph and sentence boundaries, a blind cut would ignore them")

    prose = 8 <= p.avg_sentence_words <= 40 and 150 <= p.avg_paragraph_chars <= 3500
    raw[STRATEGY_ID] = (
        0.55 if prose else 0.35,
        (f"sentences average {p.avg_sentence_words:.0f} words and paragraphs {p.avg_paragraph_chars} "
         "characters, ordinary prose that breaks cleanly on paragraph then sentence")
        if prose else "prose shape is unusual, but recursive splitting still degrades safely",
    )

    raw["sentence_window"] = (
        0.4 if p.avg_sentence_words >= 14 else 0.1,
        "would help pinpoint single facts, but needs a second index to return the window",
    )
    raw["parent_child"] = (
        0.45 if p.avg_paragraph_chars > 1200 else 0.1,
        f"long paragraphs ({p.avg_paragraph_chars} chars) fit it, but it needs a second index",
    )

    built = {i: v for i, v in raw.items() if _BY_ID[i]["built"]}
    ranked = sorted(built, key=lambda i: built[i][0], reverse=True)
    winner = ranked[0]
    # Structure and code are only chosen when clearly warranted; otherwise recursive.
    if built[winner][0] < 0.5:
        winner = STRATEGY_ID
    runner = next(i for i in ranked if i != winner)

    out = []
    for sid, (score, reason) in raw.items():
        row = _BY_ID[sid]
        if not row["built"]:
            verdict = "not built"
        elif sid == winner:
            verdict = "chosen"
        elif sid == runner:
            verdict = "runner-up"
        else:
            verdict = "ruled out"
        out.append(StrategyScore(sid, row["name"], row["built"], round(score, 3), verdict, reason))
    out.sort(key=lambda r: (r.verdict != "chosen", -r.score))
    return out


# ---------------------------------------------------------------------------
# The reasoning run
# ---------------------------------------------------------------------------

@dataclass
class ChunkPlan:
    strategy_id: str
    strategy_label: str
    chunks: list[str]
    profile: DocProfile
    scores: list[StrategyScore]
    rationale: str


def _fallback_rationale(winner: StrategyScore, runner: StrategyScore | None, p: DocProfile) -> str:
    text = f"{winner.name}: {winner.reason}."
    if runner:
        text += f" Next closest was {runner.name} ({runner.reason})."
    return text


def plan_chunking(
    pages: list[str],
    *,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    narrate: Callable[[str], str | None] | None = None,
    force: str | None = None,
    chunk_size: int | None = None,
    holder: dict | None = None,
) -> Iterator[dict]:
    """Yield reasoning events. The finished ChunkPlan lands in holder['plan']."""
    settings = get_settings()
    holder = holder if holder is not None else {}
    total_chars = sum(len(p) for p in pages)

    yield {"type": "stage", "stage": "read", "text": f"Reading {len(pages)} page(s), {total_chars:,} characters of text."}
    clean = normalize_text(pages)
    if not clean.strip():
        holder["plan"] = None
        return
    n_par = clean.count("\n\n") + 1
    yield {"type": "think", "text": f"Page breaks cut sentences in half. Rejoined them into {n_par} paragraphs."}

    yield {"type": "stage", "stage": "profile", "text": "Measuring the shape of this document."}
    p = profile_document(pages, clean)
    yield {"type": "think", "text": f"{p.words:,} words. Paragraphs average {p.avg_paragraph_chars} characters; sentences average {p.avg_sentence_words:.0f} words."}
    if p.headings:
        shown = ", ".join(f"“{h[:40]}”" for h in p.sample_headings[:3])
        noun = "heading-like line" if p.headings == 1 else "heading-like lines"
        yield {"type": "think", "text": f"Found {p.headings} {noun} ({shown}…), on {p.heading_pages} of {p.pages} page{'s' if p.pages != 1 else ''}."}
    else:
        yield {"type": "think", "text": "No headings found. Nothing labels where topics change."}
    yield {"type": "think", "text": f"Code-like lines: {p.code_ratio:.0%}. Bullet lines: {p.list_ratio:.0%}. Digits: {p.digit_ratio:.0%} of characters."}
    yield {"type": "profile", "profile": p.as_dict()}

    shift = None
    if embed_fn is not None and force is None:
        yield {"type": "stage", "stage": "probe", "text": "Probing for topic shifts with the embedding model."}
        try:
            shift = topic_shift_ratio(clean, embed_fn)
        except Exception:
            shift = None
        if shift:
            yield {"type": "think", "text": f"Compared {shift[2]} spread-out paragraphs: {shift[0]:.0%} of neighbouring pairs change topic sharply."}
        else:
            yield {"type": "think", "text": "Not enough long paragraphs to measure topic shifts. Skipping that test."}

    yield {"type": "stage", "stage": "score", "text": "Scoring each strategy against what was measured."}
    scores = score_strategies(p, shift)
    by_id = {s.id: s for s in scores}
    if force:
        if force not in BUILT_IDS:
            raise ValueError(f"unknown chunking strategy: {force}")
        winner_id = force
        yield {"type": "think", "text": f"Strategy forced to {by_id[force].name}. Skipping the vote."}
    else:
        winner_id = scores[0].id
    winner = by_id[winner_id]
    runner = next((s for s in scores if s.verdict == "runner-up"), None)
    for s in scores:
        yield {"type": "think", "text": f"{s.name}: {s.verdict}. {s.reason[0].upper()}{s.reason[1:]}."}
    yield {"type": "scores", "rows": [s.__dict__ for s in scores]}

    rationale = None
    if narrate and not force:
        facts = (
            f"Document profile: {p.as_dict()}\n"
            f"Chosen: {winner.name}. Reason: {winner.reason}.\n"
            + (f"Runner-up: {runner.name}. Reason: {runner.reason}.\n" if runner else "")
            + "Not built, never selected: sentence window, parent-child.\n"
            f"Opening of the file:\n{clean[:900]}"
        )
        rationale = narrate(facts)
    rationale = rationale or _fallback_rationale(winner, runner, p)
    yield {"type": "decision", "strategy_id": winner_id, "label": winner.name, "why": rationale}

    size = chunk_size or settings.chunk_size
    yield {"type": "stage", "stage": "chunk", "text": f"Splitting with {winner.name.lower()} at {size} characters, {round(size * settings.chunk_overlap / settings.chunk_size)} overlap."}
    chunks = apply_strategy(winner_id, clean, embed_fn, chunk_size)
    if not chunks:
        holder["plan"] = None
        return
    sizes = [len(c) for c in chunks]
    yield {
        "type": "chunked",
        "num_chunks": len(chunks),
        "avg": round(sum(sizes) / len(sizes)),
        "min": min(sizes),
        "max": max(sizes),
        "sample": chunks[0][:220],
        "sizes": sizes[:400],
    }
    holder["plan"] = ChunkPlan(winner_id, winner.name, chunks, p, scores, rationale)


def describe_chunking() -> dict:
    """Static policy shown before any file is uploaded."""
    settings = get_settings()
    size = settings.chunk_size
    overlap = settings.chunk_overlap
    ratio = round(overlap / size, 3) if size else 0.0
    return {
        "strategy_id": STRATEGY_ID,
        "strategy_label": STRATEGY_LABEL,
        "chunk_size": size,
        "chunk_overlap": overlap,
        "overlap_ratio": ratio,
        "approx_tokens": round(size / 4),
        "separators_display": SEPARATORS_DISPLAY,
        "why": (
            "The default when a file gives no stronger signal. Each upload is "
            "measured and the strategy is chosen from the file itself. "
            f"{size} characters is about {round(size / 4)} tokens. That is smaller than the usual "
            "300–500 token advice on purpose: on this project's benchmark, 500 characters found the "
            "right passage 98–100% of the time against 80–89% at 1,600. "
            f"Overlap is {overlap} characters ({ratio:.1%}), inside the 10–15% band."
        ),
        "strategies": [
            {
                "id": row["id"], "name": row["name"], "how": row["how"],
                "best_for": row["best_for"], "selected": row["id"] == STRATEGY_ID,
            }
            for row in STRATEGIES
        ],
    }
