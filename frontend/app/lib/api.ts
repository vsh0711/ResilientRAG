const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface UploadResponse {
  document_id: string;
  num_chunks: number;
  num_pages_estimate: number;
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
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_URL}/documents`, { method: "POST", body: formData });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Upload failed (${res.status}): ${body}`);
  }
  return res.json();
}

export async function askQuestion(
  documentId: string,
  question: string,
  maxRetries: number = 3
): Promise<QueryResponse> {
  const res = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_id: documentId, question, max_retries: maxRetries }),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Query failed (${res.status}): ${body}`);
  }
  return res.json();
}
