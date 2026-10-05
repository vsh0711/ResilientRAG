const TARGET = process.env.API_PROXY_TARGET || "http://localhost:8000";

export const dynamic = "force-dynamic";
export const maxDuration = 300;

type RouteContext = { params: Promise<{ path: string[] }> };

async function forward(request: Request, context: RouteContext): Promise<Response> {
  const params = await context.params;
  const url = new URL(request.url);
  const target = `${TARGET}/${params.path.join("/")}${url.search}`;
  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);
  const accept = request.headers.get("accept");
  if (accept) headers.set("accept", accept);
  const forwarded = request.headers.get("x-forwarded-for");
  if (forwarded) headers.set("x-forwarded-for", forwarded);
  const requestId = request.headers.get("x-request-id");
  if (requestId) headers.set("x-request-id", requestId);
  const session = request.headers.get("x-session-id");
  if (session) headers.set("x-session-id", session);

  try {
    const upstream = await fetch(target, {
      method: request.method,
      headers,
      body: request.method === "GET" || request.method === "HEAD" ? undefined : await request.arrayBuffer(),
      signal: AbortSignal.timeout(200_000),
      cache: "no-store",
    });
    const responseHeaders = new Headers();
    const upstreamType = upstream.headers.get("content-type");
    if (upstreamType) responseHeaders.set("content-type", upstreamType);
    const upstreamId = upstream.headers.get("x-request-id");
    if (upstreamId) responseHeaders.set("x-request-id", upstreamId);
    const retryAfter = upstream.headers.get("retry-after");
    if (retryAfter) responseHeaders.set("retry-after", retryAfter);

    // Server-sent events must pass through as they arrive. Buffering the
    // body would hold every "thinking" line until the whole run finished.
    if (upstreamType?.includes("text/event-stream") && upstream.body) {
      responseHeaders.set("cache-control", "no-cache, no-transform");
      responseHeaders.set("x-accel-buffering", "no");
      return new Response(upstream.body, { status: upstream.status, headers: responseHeaders });
    }
    return new Response(await upstream.arrayBuffer(), {
      status: upstream.status,
      headers: responseHeaders,
    });
  } catch {
    return Response.json(
      { detail: "The API is not reachable. Start the backend, Qdrant, and Redis, then try again." },
      { status: 503 }
    );
  }
}

export const GET = forward;
export const POST = forward;
