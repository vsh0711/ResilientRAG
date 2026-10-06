const QUERY_TIMEOUT_MS = 190_000;

export interface ChunkingStrategy {
  id: string;
  name: string;
  how: string;
  best_for: string;
  selected: boolean;
}

export interface ChunkingPolicy {
  strategy_id: string;
  strategy_label: string;
  chunk_size: number;
  chunk_overlap: number;
  overlap_ratio: number;
  approx_tokens: number;
  separators_display: string;
  why: string;
  strategies: ChunkingStrategy[];
}

export interface UploadResponse {
  document_id: string;
  num_chunks: number;
  num_pages_estimate: number;
  indexed: boolean;
  strategy_id: string;
  strategy_label: string;
  chunk_size: number;
  chunk_overlap: number;
  overlap_ratio: number;
  avg_chunk_chars: number;
  rationale: string;
}

export interface HealingStep {
  retry_number: number;
  failure_reason: string;
  action_taken: string;
  previous_retrieval_mode: string;
  new_retrieval_mode: string;
  previous_budget: number;
  new_budget: number;
  query_rewritten: boolean;
  rewritten_query?: string;
}

export interface QueryResponse {
  answer: string;
  final_score: number;
  relevance_score: number;
  faithfulness_score: number;
  failure_reason: string;
  retry_count: number;
  retrieval_mode: string;
  healing_trace: HealingStep[];
  latency_ms: Record<string, number>;
  token_usage: Record<string, { prompt_tokens: number; completion_tokens: number }>;
  cache_hits?: Record<string, boolean>;
  sources: string[];
}

/**
 * With NEXT_PUBLIC_API_URL set (the Vercel deployment) the browser talks to the
 * API directly. Vercel functions cap request bodies at 4.5 MB and run for 60 s
 * at most on the free plan, which would break PDF uploads and long answers if
 * they went through the proxy. Without it, requests use the same-origin proxy.
 */
const DIRECT = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");

function endpoint(path: string): string {
  return DIRECT ? `${DIRECT}${path}` : `/api/proxy${path}`;
}

/** One random ID per browser. The API rate-limits per session, not per office IP. */
function sessionId(): string {
  try {
    let id = window.localStorage.getItem("rrag-session");
    if (!id) {
      id = crypto.randomUUID();
      window.localStorage.setItem("rrag-session", id);
    }
    return id;
  } catch {
    return "";
  }
}

const CODE_KEY = "rrag-code";

export function storedAccessCode(): string {
  try {
    return window.localStorage.getItem(CODE_KEY) ?? "";
  } catch {
    return "";
  }
}

export function saveAccessCode(code: string): void {
  try {
    if (code) window.localStorage.setItem(CODE_KEY, code);
    else window.localStorage.removeItem(CODE_KEY);
  } catch {
    /* private mode: the code just will not be remembered */
  }
}

function withSession(init: RequestInit = {}): RequestInit {
  const headers: Record<string, string> = { ...(init.headers as Record<string, string> | undefined) };
  const id = sessionId();
  if (id) headers["X-Session-Id"] = id;
  const code = storedAccessCode();
  if (code) headers["X-Access-Code"] = code;
  return { ...init, headers };
}

/** null when the server is open, otherwise whether the saved code is accepted. */
export async function checkAccess(code?: string): Promise<{ required: boolean; ok: boolean }> {
  const status = await fetch(endpoint("/auth/status"));
  const { required } = (await status.json()) as { required: boolean };
  if (!required) return { required: false, ok: true };
  const supplied = code ?? storedAccessCode();
  if (!supplied) return { required: true, ok: false };
  const res = await fetch(endpoint("/documents/chunking"), {
    headers: { "X-Access-Code": supplied, "X-Session-Id": sessionId() },
  });
  return { required: true, ok: res.ok };
}

async function readError(res: Response, fallback: string): Promise<string> {
  const body = await res.text();
  try {
    const parsed = JSON.parse(body) as { detail?: unknown };
    if (typeof parsed.detail === "string" && parsed.detail) return parsed.detail;
  } catch {
    /* body was not JSON */
  }
  return body || fallback;
}

export async function fetchChunking(): Promise<ChunkingPolicy> {
  const res = await fetch(endpoint("/documents/chunking"), withSession());
  if (!res.ok) throw new Error(await readError(res, "Could not load chunking settings"));
  return res.json();
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(
    endpoint("/documents"),
    withSession({ method: "POST", body: formData, signal: AbortSignal.timeout(QUERY_TIMEOUT_MS) })
  );
  if (!res.ok) throw new Error(await readError(res, "Upload failed"));
  return res.json();
}

export async function askQuestion(
  documentId: string,
  question: string,
  maxRetries: number = 3
): Promise<QueryResponse> {
  const res = await fetch(
    endpoint("/query"),
    withSession({
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document_id: documentId, question, max_retries: maxRetries }),
      signal: AbortSignal.timeout(QUERY_TIMEOUT_MS),
    })
  );
  if (!res.ok) throw new Error(await readError(res, "Query failed"));
  return res.json();
}

// ---- server-sent events ---------------------------------------------------

export interface StrategyScore {
  id: string;
  name: string;
  built: boolean;
  score: number;
  verdict: "chosen" | "runner-up" | "ruled out" | "not built";
  reason: string;
}

export interface DocProfile {
  pages: number;
  chars: number;
  words: number;
  paragraphs: number;
  avg_paragraph_chars: number;
  headings: number;
  heading_density: number;
  heading_pages: number;
  code_ratio: number;
  list_ratio: number;
  digit_ratio: number;
  avg_sentence_words: number;
}

export type UploadEvent =
  | { type: "stage"; stage: string; text: string }
  | { type: "think"; text: string }
  | { type: "profile"; profile: DocProfile }
  | { type: "scores"; rows: StrategyScore[] }
  | { type: "decision"; strategy_id: string; label: string; why: string }
  | { type: "chunked"; num_chunks: number; avg: number; min: number; max: number; sample: string; sizes: number[] }
  | { type: "done"; upload: UploadResponse }
  | { type: "error"; detail: string };

export type QueryEvent =
  | { type: "start"; question: string; max_retries: number }
  | { type: "retrieve"; attempt: number; mode: string; budget: number; query: string; docs: string[] }
  | { type: "generate"; attempt: number; answer: string; cached: boolean }
  | {
      type: "score";
      attempt: number;
      relevance: number;
      faithfulness: number;
      combined: number;
      failure_reason: string;
      passed: boolean;
      threshold: number;
    }
  | { type: "heal"; step: HealingStep }
  | { type: "result"; result: QueryResponse }
  | { type: "error"; detail: string };

async function readStream<T>(res: Response, onEvent: (event: T) => void): Promise<void> {
  if (!res.body) throw new Error("The server sent no stream");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut: number;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      const data = frame
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data) onEvent(JSON.parse(data) as T);
    }
  }
}

export async function uploadDocumentStream(
  file: File,
  onEvent: (event: UploadEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(endpoint("/documents/stream"), withSession({ method: "POST", body: formData, signal }));
  if (!res.ok) throw new Error(await readError(res, "Upload failed"));
  await readStream<UploadEvent>(res, onEvent);
}

export async function askQuestionStream(
  documentId: string,
  question: string,
  onEvent: (event: QueryEvent) => void,
  maxRetries: number = 3,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(
    endpoint("/query/stream"),
    withSession({
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document_id: documentId, question, max_retries: maxRetries }),
      signal,
    })
  );
  if (!res.ok) throw new Error(await readError(res, "Query failed"));
  await readStream<QueryEvent>(res, onEvent);
}
