# Deployment

Two supported shapes. Neither needs a paid Groq key to start, but a free key allows only a handful of questions per day (see [limits](#groq-free-tier-limits)).

| | A. Free demo: Vercel + Hugging Face Space | B. Docker on a VPS |
|---|---|---|
| Cost | $0 | a server (4 CPU / 8 GB) |
| Persistence | none: documents vanish on restart | Qdrant + Redis volumes |
| Users | a few, light use | tens, with a paid LLM tier |
| Setup | ~15 minutes | ~15 minutes |

Vercel runs only the Next.js frontend. The backend needs a long-running process with ~1 GB of RAM for three embedding models, so it cannot live on Vercel.

## A. Free demo

### A1. Backend on a Hugging Face Space (no Docker)

Use the **Gradio SDK**, not the Docker SDK. Docker Spaces are not available on every free account; the Gradio SDK is, and it can run any Python app. The bundle's `space_app.py` starts the FastAPI app directly and never imports Gradio.

1. Create an account at huggingface.co and a **new Space**: SDK **Gradio**, hardware **CPU basic (free)**.
2. Build the bundle and push it to the Space:
   ```bash
   ./deploy/huggingface/build_space.sh          # writes dist/space (add "docker" for the Docker SDK variant)
   cd dist/space
   git init -b main
   git remote add space https://huggingface.co/spaces/<you>/<space-name>
   git add . && git commit -m "deploy"
   git push space main --force                  # username + a HF access token (write) as the password
   ```
3. In the Space → **Settings → Variables and secrets**, add:

   | name | kind | value |
   |---|---|---|
   | `GROQ_API_KEY` | secret | your free Groq key |
   | `ACCESS_CODES` | secret | `openssl rand -base64 18`; protects your quota |
   | `API_CORS_ORIGINS` | variable | your Vercel URL, e.g. `https://resilient-rag.vercel.app` (comma-separate several) |
   | `GROQ_PRIMARY_MODEL` | variable | `openai/gpt-oss-120b` |
   | `GROQ_FALLBACK_MODEL` | variable | `openai/gpt-oss-20b` |
   | `GROQ_JUDGE_MODEL` | variable | `qwen/qwen3.8-27b` |

4. The first start installs packages and downloads three embedding models (a few minutes). Then check
   `https://<you>-<space-name>.hf.space/health/ready`; expect `"status":"ok"`.

How this was verified: the bundle's `requirements.txt` plus the pinned `gradio==4.44.1` were installed into a fresh Python 3.11 virtualenv and `space_app.py` was run exactly as the Space runs it; a real upload and question succeeded. The Space itself was not deployed from this environment (no Hugging Face login), so expect to read the build log once.

Free Spaces sleep after inactivity and lose their memory when they restart, so uploaded documents disappear. For persistence, point `QDRANT_URL` + `QDRANT_API_KEY` at a free Qdrant Cloud cluster and set `QDRANT_USE_MEMORY=false`.

**If you have no Hugging Face access at all**, run the backend on your own machine and expose it over a free tunnel:

```bash
cd backend && uv sync && QDRANT_USE_MEMORY=true CACHE_ENABLED=false \
  ACCESS_CODES=<code> API_CORS_ORIGINS=https://<your-app>.vercel.app \
  uv run uvicorn app.main:app --port 8000
cloudflared tunnel --url http://localhost:8000     # prints a https://….trycloudflare.com URL
```

Use that URL as `NEXT_PUBLIC_API_URL`. It works only while your machine is on, and the URL changes each time the tunnel restarts.

### A2. Frontend on Vercel

1. Push the repo to GitHub (already done if you are reading this there).
2. vercel.com → **Add New → Project** → import the repo.
3. **Root Directory: `frontend`**. Framework preset: Next.js (auto-detected).
4. **Environment variable:** `NEXT_PUBLIC_API_URL` = `https://<you>-<space-name>.hf.space` (no trailing slash).
5. Deploy, then put the resulting Vercel URL into the Space's `API_CORS_ORIGINS` and restart the Space.

The browser calls the Space directly. Going through a Vercel function would cap uploads at 4.5 MB and answers at 60 s on the free plan.

### A3. Check it

Open the Vercel URL → enter your access code → drop a PDF → watch the reasoning → ask a question.

## B. Docker on a VPS

```bash
cp .env.example .env     # set GROQ_API_KEY, DOMAIN, ACCESS_CODES
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
curl https://$DOMAIN/api/proxy/health/ready -H "X-Access-Code: <code>"
```

Point your domain's DNS at the host and open ports 80 and 443; Caddy fetches a Let's Encrypt certificate. Only Caddy publishes ports. See [Operations](RUNBOOK.md) for backups, upgrades and troubleshooting.

Rehearse on your laptop: the same command with `DOMAIN` unset serves `https://localhost` (accept the local certificate).

## Configuration reference

All settings are environment variables read by `backend/app/config.py`; defaults shown. `.env.example` is the annotated template.

| variable | default | meaning |
|---|---|---|
| `GROQ_API_KEY` | – | required |
| `GROQ_PRIMARY_MODEL` / `GROQ_FALLBACK_MODEL` / `GROQ_JUDGE_MODEL` | gpt-oss-120b / gpt-oss-20b / qwen3.8-27b | answering, fallback, judging + narration |
| `LLM_MAX_RETRIES` / `LLM_BACKOFF_BASE_SECONDS` / `LLM_TIMEOUT_SECONDS` | 5 / 2.0 / 30 | per-call retry policy |
| `QDRANT_URL` / `QDRANT_API_KEY` / `QDRANT_USE_MEMORY` | localhost:6333 / – / false | vector store |
| `REDIS_URL` / `CACHE_ENABLED` / `CACHE_TTL_SECONDS` | localhost / true / 3600 | cache, chunk store, rate counters |
| `SCORE_PASS_THRESHOLD` | 0.8 | combined judge score needed to stop |
| `MAX_RETRIES_DEFAULT` | 3 | healing rounds (API caps at 3) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 500 / 62 | characters |
| `API_CORS_ORIGINS` | http://localhost:3000 | comma-separated or JSON list |
| `ACCESS_CODES` | empty (open) | comma-separated shared codes |
| `RATE_LIMIT_PER_MINUTE` / `IP_RATE_LIMIT_PER_MINUTE` | 60 / 1200 | per session / per IP |
| `TRUST_PROXY` | false | read `X-Forwarded-For`; only behind a proxy you control |
| `MAX_UPLOAD_SIZE_MB` | 20 | |
| `MAX_CONCURRENT_QUERIES` / `MAX_CONCURRENT_UPLOADS` | 12 / 2 | per worker |
| `QUERY_TIMEOUT_SECONDS` / `QUERY_QUEUE_TIMEOUT_SECONDS` / `UPLOAD_QUEUE_TIMEOUT_SECONDS` | 90 / 15 / 20 | deadlines |
| `UPLOAD_DIR` | /tmp/resilientrag_uploads | temp + chunk fallback |
| `WARMUP_MODELS` | false (true in Docker) | download models at start |
| `UVICORN_WORKERS` (Docker only) | 2 | |
| frontend: `NEXT_PUBLIC_API_URL` / `API_PROXY_TARGET` | – / http://localhost:8000 | direct base URL / proxy target |

## Groq free-tier limits

Read from this project's key, per model: **8,000 tokens/minute, 200,000 tokens/day, 1,000 requests/day.** A question costs about 1,500–4,000 tokens (answer + two judges), so expect roughly 50–130 questions per day and only a few per minute. The client switches to the fallback model when a daily budget is spent. Consequences for a public demo: keep `ACCESS_CODES` on, share the code selectively, and expect "temporarily unable" answers late in the day. Lowering the cost per question: set `max_retries` to 0 from the client, or use a smaller judge model.

## Pre-flight checklist

- [ ] `GET /health/ready` is `ok` and `llm_models.missing` is empty
- [ ] `ACCESS_CODES` set; a request without a code returns 401
- [ ] `API_CORS_ORIGINS` contains the exact frontend origin (scheme + host, no path)
- [ ] a real PDF uploads and a question gets a grounded answer
- [ ] (Docker) Qdrant, Redis and the backend are **not** reachable from the internet
- [ ] (Docker) volumes are backed up
