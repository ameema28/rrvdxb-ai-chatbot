# RRVDXB AI Shopping Chatbot: Deployment Guide

This guide covers local runs (venv, Docker, docker compose) and a public demo on **Google Cloud Run**.
Numbers marked *measured* come from local runs on Windows 11 with Docker Desktop (2026-11-07). Everything marked *not measured* has not been checked on Cloud Run yet.

---

## 1. Local development (venv)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -r requirements-dev.txt
cp .env.example .env            # then set INTERNAL_JWT_SECRET (and GROQ_API_KEY)
uvicorn app.main:app --reload
```

- Demo page: http://127.0.0.1:8000/ (it needs `DEMO_TOKENS_ENABLED=true` in `.env`)
- API docs: http://127.0.0.1:8000/docs
- `INTERNAL_JWT_SECRET` must be at least 32 characters, or the app refuses to start. Generate one with
  `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
- Without `GROQ_API_KEY` every reply is fixed fallback text (`fallback_used: true`).

---

## 2. Docker

### 2.1 Build and run

```bash
docker build -t rrvdxb-ai-chatbot:phase3 .
```

The image:
- installs **CPU-only torch** (2.15.0+cpu) from the PyTorch CPU index, then the runtime requirements (no dev tools);
- runs as a **non-root user (uid 1000)**;
- contains the embedding model and the FAQ Chroma index (built during `docker build`, which needs internet). At runtime Hugging Face is offline (`HF_HUB_OFFLINE=1`);
- listens on **`$PORT` (default 8080)**;
- has **no secret inside**: `INTERNAL_JWT_SECRET` must be passed at runtime.

```powershell
# PowerShell: generate a secret into an env var without printing it, then pass it by name
$env:INTERNAL_JWT_SECRET = python -c "import secrets; print(secrets.token_urlsafe(48), end='')"
docker run --rm -p 8080:8080 -e INTERNAL_JWT_SECRET -e DEMO_TOKENS_ENABLED=true rrvdxb-ai-chatbot:phase3
```

Without `INTERNAL_JWT_SECRET` the container exits at start-up with
`RuntimeError: INTERNAL_JWT_SECRET must be at least 32 characters` (the value is never printed).

Image defaults: `APP_ENV=production`, `DEBUG=false`, `DATABASE_URL=sqlite:////tmp/rrvdxb.db`, `DB_AUTO_MIGRATE=true`, `RAG_WARMUP=true`, `DEMO_TOKENS_ENABLED=false`.

### 2.2 docker compose

```bash
# SQLite chat history in the named volume rrvdxb-data
docker compose up -d --build

# Optional local PostgreSQL 16 (profile "postgres", internal network only, trust auth)
COMPOSE_DATABASE_URL=postgresql://postgres@db:5432/postgres docker compose --profile postgres up -d --build

docker compose logs -f app
docker compose down          # add -v to delete the volumes
```

PowerShell: `$env:COMPOSE_DATABASE_URL="postgresql://postgres@db:5432/postgres"; docker compose --profile postgres up -d --build`.

- Compose serves the API and the demo page on http://localhost:8000 and turns demo tokens on.
- `INTERNAL_JWT_SECRET` and `GROQ_API_KEY` come from your own `.env` (optional file; without the secret the app does not start).
- Compose uses `COMPOSE_DATABASE_URL`, not `DATABASE_URL`, so a local-dev `DATABASE_URL=sqlite:///./rrvdxb.db` in `.env` cannot point the container at its read-only code folder.

---

## 3. Google Cloud Run (public demo)

The same image runs on Cloud Run unchanged. Visitors open the service URL and get the demo page at `/`. The page fetches an anonymous demo token from `POST /api/v1/auth/demo-token`, keeps it in memory only and calls `/api/v1/ai/chat` on the same origin.

### 3.1 Service settings

| Setting | Value | Why |
|:---|:---|:---|
| Maximum instances | **1** | The rate limiter is in memory, migrations run at start-up (`DB_AUTO_MIGRATE=true`), and the SQLite fallback lives in one instance. More instances would mean separate limits and racing migrations. |
| Minimum instances | **0** | Scales to zero when idle (no cost when idle). The first visitor after an idle period waits for a cold start. |
| Container port | 8080 (Cloud Run default) | The app reads `$PORT`. Do not set `PORT` yourself. |
| Memory | **1 GiB** | Measured locally: 405.3 MiB after one chat (`docker stats`, no Groq key, so no live LLM call). Rule used: the smallest of 1 / 2 GiB that is at least twice that. Cloud Run's `/tmp` (SQLite fallback) is in memory and counts against this limit. Not measured on Cloud Run. |
| CPU | 1, with startup CPU boost (`--cpu-boost`) | One uvicorn worker. The boost shortens model loading during a cold start. |
| Authentication | Allow unauthenticated invocations | Visitors must be able to open the page. The chat API itself needs a bearer token (demo tokens are anonymous and short-lived). |
| Request timeout | Default (300 s) | A chat request is bounded by its own 4.0 s budget. |

### 3.2 Secrets (Secret Manager)

| Secret | Value | Notes |
|:---|:---|:---|
| `GROQ_API_KEY` | Your Groq key | Without it every reply is fallback text. |
| `INTERNAL_JWT_SECRET` | Random, at least 32 characters | Generate it, never paste it into chat, code or tickets. Changing it invalidates all issued tokens (the demo page simply fetches a new one). |
| `DATABASE_URL` | PostgreSQL URL, e.g. Neon's **direct** connection string with `sslmode=require` | Optional. Without it, chat history is SQLite in `/tmp` (see 3.5). |

Each secret is exposed to the service as an environment variable of the same name. The runtime service account needs the **Secret Manager Secret Accessor** role on each secret (commands in 3.7).

### 3.3 Environment variables

| Name | Value | Notes |
|:---|:---|:---|
| `APP_ENV` | `production` | Image default. |
| `DEBUG` | `false` | Image default. |
| `DEMO_TOKENS_ENABLED` | `true` | **Required for the demo page.** The image default is `false` (the endpoint then answers 404). |
| `TRUST_PROXY` | `true` | Take the client IP from `X-Forwarded-For`. |
| `TRUST_PROXY_HOPS` | `1` | The client IP is the right-most `X-Forwarded-For` entry, which Google's front end appends. Entries to its left are client-supplied and ignored. Use `2` if an external HTTPS load balancer sits in front of the service. |
| `DB_AUTO_MIGRATE` | `true` | Image default. Safe only with maximum instances 1. |
| `RAG_WARMUP` | `true` | Image default. |
| `LOG_LEVEL` | optional, default `INFO` | Root log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Logs contain no message text, tokens or keys. |
| `RAG_SIMILARITY_THRESHOLD` | optional, default `0.45` | Minimum FAQ similarity for grounding (see 3.10). |
| `CHAT_IP_RATE_PER_MINUTE` | optional, default `20` | Chat requests per client IP per minute (see 3.6). |
| `RATE_LIMIT_PER_MINUTE` | optional, default `20` | Chat requests per token subject per minute. |
| `DEMO_TOKEN_RATE_PER_MINUTE` | optional, default `5` | Demo tokens per client IP per minute. |
| `DEMO_TOKEN_TTL_SECONDS` | optional, default `1800` | Demo token lifetime. |
| `LLM_MODEL` | optional, default `llama-3.1-8b-instant` | Change it if Groq stops serving that model. |

### 3.4 Health, readiness and cold start

- **`GET /health`** needs no token and is not rate limited. It returns `{"status": "ok" | "degraded", "service": ..., "rag_ready": true | false}` with HTTP 200.
  - `rag_ready: true` means the embedding model and the FAQ index answered a warm-up query at start-up.
  - `status: "degraded"` means the warm-up failed; the app still answers, and FAQ questions take the no-match path.
- The warm-up runs before uvicorn opens the port, so Cloud Run's default TCP startup probe already waits for a warm app.
- **Cold start, measured locally:** 10.1 s from `docker run` to the first `/health` 200 with `rag_ready: true` (Windows PC, Docker Desktop, image already on disk). On Cloud Run a cold start also includes starting the instance and loading a 2.64 GB (unpacked) image: **not measured yet**.
- **First FAQ answer after start-up, measured locally:** `response_time_ms` 38 for "What is your return policy?", with no retrieval timeout in the log. There was no Groq key, so this covers intent, retrieval and the fallback path, not a live LLM call.

### 3.5 What resets, and what persists

- **Without `DATABASE_URL`:** chat history is SQLite in `/tmp/rrvdxb.db`. Cloud Run's `/tmp` is in memory, belongs to one instance and counts against the memory limit. History is **lost on every restart, new revision or scale to zero**.
- **With `DATABASE_URL` (e.g. Neon):** history survives restarts. Neon's free tier suspends an idle database; the first query after that waits for it to wake up (**not measured**).
- **Demo tokens** last 30 minutes. They stay valid across restarts as long as `INTERNAL_JWT_SECRET` is unchanged; the page fetches a new one on a 401.
- **Rate-limit counters** are in memory and reset with the instance.
- The catalog, FAQs and the deal are mock data.

### 3.6 Rate limits

| Limit | Default | Key |
|:---|:---|:---|
| Chat, per user | 20 / minute | Verified token subject (each demo token has a new random subject) |
| Chat, per client IP | 20 / minute | Client IP (proxy-aware) |
| Demo tokens | 5 / minute | Client IP (proxy-aware) |

`/`, `/health`, `/docs`, `/redoc` and `/openapi.json` are not rate limited.

**Shared Wi-Fi / events.** Everyone behind one public IP (an office, a university network, conference Wi-Fi, a mobile carrier's NAT) shares the per-IP chat cap: together they get 20 chats a minute, and 5 new demo tokens a minute. For a live demo to a group, raise the cap temporarily with an environment variable (this creates a new revision, which also resets the in-memory counters and, without `DATABASE_URL`, the SQLite history):

```bash
gcloud run services update rrvdxb-ai-chatbot --region <REGION> --update-env-vars CHAT_IP_RATE_PER_MINUTE=100
# afterwards, back to the default of 20:
gcloud run services update rrvdxb-ai-chatbot --region <REGION> --remove-env-vars CHAT_IP_RATE_PER_MINUTE
```

`DEMO_TOKEN_RATE_PER_MINUTE` can be raised the same way. Every extra chat can be a paid Groq call, so set it back afterwards.

### 3.7 Step by step (commands for the owner)

Replace `<PROJECT_ID>`, `<PROJECT_NUMBER>` and `<REGION>` (for example `europe-west1`). Nothing below is run by the build tooling.

**1. Neon (optional, for durable history).** Create a free project in a region close to `<REGION>`. In the connection details choose the **direct** connection (not "pooled"). Copy the connection string; it contains `sslmode=require`. Keep it out of chat, tickets and the repository. The app converts `postgresql://` URLs to the psycopg 3 driver and keeps the `sslmode` / `channel_binding` parameters.

**2. Project and APIs.**

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud services enable compute.googleapis.com run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
```

**3. Artifact Registry repository.**

```bash
gcloud artifacts repositories create rrvdxb --repository-format=docker --location=<REGION>
```

**4. Secrets.** Simplest: **Console > Security > Secret Manager > Create secret**, name it (`GROQ_API_KEY`, `INTERNAL_JWT_SECRET`, `DATABASE_URL`), paste the value, create. Repeat for each.

Or with gcloud, from files that you delete straight away (PowerShell; `-NoNewline` matters, a trailing newline would become part of the secret):

```powershell
$f = "$env:TEMP\rrvdxb-secret.txt"

# INTERNAL_JWT_SECRET: generated, never printed
python -c "import secrets; print(secrets.token_urlsafe(48), end='')" | Set-Content -NoNewline -Encoding ascii $f
gcloud secrets create INTERNAL_JWT_SECRET --replication-policy=automatic --data-file=$f

# GROQ_API_KEY: paste the key when prompted (input is hidden)
$k = Read-Host -AsSecureString "Groq key"; [Net.NetworkCredential]::new("", $k).Password | Set-Content -NoNewline -Encoding ascii $f
gcloud secrets create GROQ_API_KEY --replication-policy=automatic --data-file=$f

# DATABASE_URL: paste the Neon direct connection string when prompted
$k = Read-Host -AsSecureString "Neon URL"; [Net.NetworkCredential]::new("", $k).Password | Set-Content -NoNewline -Encoding ascii $f
gcloud secrets create DATABASE_URL --replication-policy=automatic --data-file=$f

Remove-Item $f
```

**5. Let the runtime service account read the secrets.** Cloud Run uses the Compute Engine default service account unless you choose another one. Check that it exists (one row expected; it appears once the Compute Engine API is enabled):

```bash
gcloud iam service-accounts list --filter="email~-compute@developer.gserviceaccount.com"
gcloud projects describe <PROJECT_ID> --format="value(projectNumber)"
```

Then, for each of `GROQ_API_KEY`, `INTERNAL_JWT_SECRET` and `DATABASE_URL`:

```bash
gcloud secrets add-iam-policy-binding <SECRET_NAME> \
  --member=serviceAccount:<PROJECT_NUMBER>-compute@developer.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor
```

**6. Push the image built locally.** Build it from the commit you want to deploy (`docker build -t rrvdxb-ai-chatbot:phase3 .`, see 3.11); an image built before a code change does not contain it.

```bash
gcloud auth configure-docker <REGION>-docker.pkg.dev
docker tag rrvdxb-ai-chatbot:phase3 <REGION>-docker.pkg.dev/<PROJECT_ID>/rrvdxb/rrvdxb-ai-chatbot:v1
docker push <REGION>-docker.pkg.dev/<PROJECT_ID>/rrvdxb/rrvdxb-ai-chatbot:v1
```

**7. Deploy.** (Leave out `DATABASE_URL=DATABASE_URL:latest` if you skipped Neon.)

```bash
gcloud run deploy rrvdxb-ai-chatbot \
  --image <REGION>-docker.pkg.dev/<PROJECT_ID>/rrvdxb/rrvdxb-ai-chatbot:v1 \
  --region <REGION> \
  --allow-unauthenticated \
  --max-instances 1 --min-instances 0 \
  --cpu 1 --memory 1Gi --cpu-boost \
  --set-secrets GROQ_API_KEY=GROQ_API_KEY:latest,INTERNAL_JWT_SECRET=INTERNAL_JWT_SECRET:latest,DATABASE_URL=DATABASE_URL:latest \
  --set-env-vars APP_ENV=production,DEBUG=false,DEMO_TOKENS_ENABLED=true,TRUST_PROXY=true,TRUST_PROXY_HOPS=1,DB_AUTO_MIGRATE=true
```

PowerShell: put the command on one line, or end each line with a backtick instead of `\`.

**8. Smoke check, then benchmark** (see 3.8).

**9. Check the client IP** (see 3.9).

### 3.8 Post-deploy checks

`scripts/smoke_check.py` (Python stdlib only, no API key) checks a deployed URL and writes a JSON summary to `--out`, which must be outside the repository:

```bash
python scripts/smoke_check.py --url https://<service-url> --out ../_facts/rrvdxb_smoke.json
```

- It checks `/health` (`status=ok`, `rag_ready=true`), the demo page, a demo token, a chat without a token (401), the 4 example questions (expected intent and `fallback_used=false`; the product question must return products) and a price-guard probe ("Is the iPhone 14 Pro Max only AED 3,999?"; the catalog price is AED 4,699, so the reply must not quote AED 3,999).
- Two **WARN-only scope probes** ("Does Amazon deliver to Riyadh?", "How much is a flight from Karachi to Lahore?"). Both pass the retrieval gate (see 3.10), so the reply should decline under the prompt's scope rule. The check is a text heuristic (the reply mentions RRVDXB and does not repeat the FAQ's delivery time); read the printed reply yourself.
- Unless `--skip-rate-limit`: the per-user chat limit, the per-IP chat cap with spoofed `X-Forwarded-For` entries, and the demo-token limit with spoofed entries. The rate-limit checks send empty messages, which cost no LLM call.
- About 7 live LLM calls; about 3.5 minutes with the rate-limit checks (three waits of 61 s).
- Measured locally against the container without a Groq key: 8 PASS, 4 FAIL, where the FAILs were only `fallback_used: true` on the example questions (expected without a key); all three rate-limit checks got their 429.

`scripts/benchmark.py` (stdlib only) gets one demo token, sends one warm-up request (not counted), then N sequential chats over the 4 example questions, and prints nearest-rank p50 / p95 / max for client time and for the server's `response_time_ms`:

```bash
python scripts/benchmark.py --url https://<service-url> -n 20 --out ../_facts/rrvdxb_benchmark.json
```

- Requests start 3.5 s apart (under 20 per minute). Wait 60 s after the smoke check before starting.
- Client times include your network. `response_time_ms` is the server's own time for the whole chat request.
- It exits non-zero unless every measured request returned 200.

`scripts/eval_retrieval.py` (no LLM, no service needed) measures FAQ retrieval on the dev and held-out sets in `data/eval/` (see 3.10):

```bash
python scripts/eval_retrieval.py --out ../_facts/rrvdxb_retrieval_eval.json
python scripts/eval_retrieval.py --threshold 0.6 --out ../_facts/rrvdxb_retrieval_eval_060.json
```

### 3.9 Checking the client IP the rate limiter sees

**Not verified on Cloud Run yet:** whether the right-most `X-Forwarded-For` entry is the visitor's IP. After the smoke check has triggered a 429 on the per-IP cap, the service log has a line `Rate limit exceeded for client IP <ip>` (Cloud Run > the service > Logs). Compare `<ip>` with your own public IP:
- It matches: the setting is correct.
- It is a Google address: all visitors share one limit. The limit still cannot be bypassed with spoofed headers, but set `TRUST_PROXY_HOPS=2`.

The IP appears in the log only on a per-IP 429.

### 3.10 Retrieval threshold and known leaks

FAQ chunks ground an answer only when their cosine similarity is at least `RAG_SIMILARITY_THRESHOLD` (default **0.45**, previously 0.60). The value was **chosen on the dev set** (`data/eval/retrieval_eval.jsonl`, 30 in-scope + 7 off-topic questions) and **checked on a held-out set** (`data/eval/retrieval_heldout.jsonl`, 15 in-scope + 10 off-topic, several near-topic) that was written before any measurement and not edited afterwards.

| Set | Threshold | Gated hit@1 | Off-topic rejected | Off-topic max similarity |
|:---|:---|:---|:---|:---|
| dev | 0.60 | 0.40 | 7/7 | 0.360 |
| dev | 0.45 | 0.67 | 7/7 | 0.360 |
| held-out | 0.60 | 0.33 | 9/10 | 0.648 |
| held-out | 0.45 | 0.80 | 6/10 | 0.648 |

**Known leaks at 0.45:** 4 of the 10 held-out off-topic questions pass the gate: "Does Amazon deliver to Riyadh?" (0.648, which also passed the old 0.60), "How much is a flight from Karachi to Lahore?" (0.583), "How long does Noon take to refund an order?" (0.455) and "Is the Dubai metro running late tonight?" (0.454). Questions naming a city, a delivery or a refund score like real customer questions, so the threshold alone cannot separate them.

The reply system prompt therefore has a general **scope rule**: answer only about RRVDXB's own store, orders and products, and decline questions about other companies, travel, transport or unrelated topics. A test checks that the rule is in the prompt on every reply path; **whether the live model follows it is not measured** until the smoke check's scope probes are run against the deployed service. Full numbers: `data/eval/README.md`.

### 3.11 After changing code

The image bakes in the code, the model and the index. After any code change, rebuild the image and push a new tag (for example `:v2`), then deploy that tag. With the current Dockerfile the model download step runs again whenever anything under `app/` changes (it took 1261 s in one local build).

---

## 4. Troubleshooting

| Symptom | Likely cause |
|:---|:---|
| Container exits at start-up with `INTERNAL_JWT_SECRET must be at least 32 characters` | The secret is missing, not mapped to the service, or too short. |
| Demo page says "Demo tokens are disabled on this server." | `DEMO_TOKENS_ENABLED` is not `true`. |
| `fallback_used: true` in every reply | `GROQ_API_KEY` missing or rejected, or `LLM_MODEL` no longer served by Groq. Check the service logs for the LLM error type and HTTP status. |
| `/health` shows `"status": "degraded"` | The embedding model or FAQ index failed to load; check the start-up log for `RAG warm-up failed`. |
| Start-up fails with a database error | `DATABASE_URL` is wrong, or the Neon project is suspended or deleted. Remove the secret mapping to fall back to SQLite in `/tmp`. |
| Many 429s for a group of people | They share one public IP: see "Shared Wi-Fi / events" in 3.6. |
