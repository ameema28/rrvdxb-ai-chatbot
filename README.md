# RRVDXB AI Shopping Chatbot

AI-powered shopping assistant for the RRVDXB premium e-commerce platform.
Serves customers in UAE, KSA, Pakistan, and UK with product discovery,
order support, and personalized recommendations.

## What This Does

- **Hybrid Intent Classification** — Regex fast-path (zero LLM cost) + LLM fallback for ambiguous queries; 5 intents routed to specialized handlers; plural forms ("discounts", "deals", "tracking"...) keep the fast-path
- **RAG-Grounded FAQ Answers** — Local sentence-transformer embeddings (`all-MiniLM-L6-v2`) + ChromaDB retrieval with a tuned 0.6 cosine-similarity threshold; no OpenAI key required
- **DB-Backed Conversation Memory** — SQLAlchemy turn history with a sliding N=5 context window; anonymous fallback to `user_id=0`
- **Hallucination Guard** — every price figure in the LLM reply is mechanically validated against the injected context (FAQ chunks, catalog, recommendations, deal, user message, history); invented figures are replaced with a safety message before persistence
- **Product Recommendations & Deal Lookup** — Stub services with keyword-matching against the live catalog; drop-in ready for the real recommendation and deals microservices
- **Centralized Error Handling** — One canonical JSON shape across every error path; stack traces and secrets never leak to the client
- **Per-User Rate Limiting** — 20 requests/minute sliding window via an in-memory store with a clean ABC seam for Redis swap-in

## Feature Specification → Delivery

This repository delivers Ameema Rashid's **complete chatbot API** within the shared
AI Shopping Chatbot feature (product recommendations, shopping advice, deals alerts,
context memory, RAG FAQ answers, and intent recognition). Status against the AI checklist:

| Checklist item | Status | Where |
|----------------|--------|-------|
| API keys in `.env` (no hardcoding) | Done | `app/core/config.py` (pydantic-settings), `.env.example`, `.env*` git-ignored |
| Chatbot response < 4 s (NFR) | Timeout guard done; load test pending | `LLM_TIMEOUT_SECONDS=3.0` + `asyncio.to_thread` / `wait_for` (Milestone 7) |
| Product recommender | Stub done — real service pending | `app/services/recommender_stub.py` |
| Deal finder | Stub done — real service pending | `app/services/deal_finder_stub.py` |
| Fallback responses | Done | LLM error / timeout / no-match graceful degradation (Milestones 6–8) |
| Context memory | Done | `app/ai/chatbot/memory.py` (SQLAlchemy, N=5 window) |
| RAG pipeline (product FAQ vector store) | Done | `app/ai/chatbot/rag/` (Milestones 5–6) |
| Intent recognition | Done | `app/ai/chatbot/intent.py` (Milestone 4) |
| Price predictor / Sentiment analysis | Out of scope for the chatbot | Owned by other AI feature streams — not part of the chatbot API |

### Intentional deviations from the AI Checklist spec

| Spec | Implementation | Rationale |
|------|----------------|-----------|
| OpenAI | Groq | Faster inference, lower cost for the < 4 s NFR (see Design Decisions) |
| LangChain | Hand-rolled orchestration | Full control of prompts/timeouts/fallbacks; zero lock-in |
| `POST /api/ai/chat` | `POST /api/v1/ai/chat` | Versioned API prefix |
| `userId` in body | `X-User-Id` header (optionally `user_id` in body) | Auth runs before LLM cost; JWT swap-in seam in `dependencies.py` |
| camelCase response | snake_case response | See the Data Contract Note below |

## Design Decisions

**Why Groq instead of OpenAI?**
Groq's inference API delivers fast response times on Llama 3.1 8B at a fraction of the cost. For a latency-sensitive chatbot with a <4s NFR, Groq's speed-to-cost ratio is the decisive factor.

**Why hand-rolled orchestration instead of LangChain?**
LangChain adds heavy abstraction for a simple pipeline (classify → retrieve → generate). Keeping orchestration explicit in `chatbot_service.py` retains full control over prompt construction, timeout boundaries, and fallback behavior — with zero LangChain lock-in.

**Why a custom rate limiter instead of slowapi?**
slowapi introduces decorator magic and extra dependencies. The `RateLimitStore` ABC + `InMemoryRateLimitStore` is ~80 lines, testable, and swaps to Redis with a one-line change in the dependency guard.

**Why local embeddings instead of a hosted model?**
`sentence-transformers` runs fully offline after the initial ~90MB model download — no external API dependency, no per-query cost, and RAG stays functional even if the LLM provider is down.

## Project Status

| Capability | Status |
|-----------|--------|
| FastAPI scaffold + Pydantic v2 validation | done |
| Groq LLM integration with graceful fallback | done |
| DB-backed conversation memory (N=5 window) | done |
| Hybrid intent classification (regex + LLM) | done |
| RAG pipeline (FAQ chunk → embed → retrieve → ground) | done |
| Product recommendation stub (real service pending) | done |
| Deal finder stub (real service pending) | done |
| Auth stub (`X-User-Id`) + per-user rate limiting | done |
| Centralized error handling | done |
| Hallucination guard (ungrounded prices blocked) | done |
| Chatbot response < 4 s — timeout guard | done |
| Streaming responses | pending |
| Real JWT authentication (`Authorization: Bearer`) | pending |
| Redis-backed rate limit store | pending |
| Docker + multi-stage deployment | pending |
| Load testing (<4s NFR validation) | pending |

> See [`docs/chatbot.md`](docs/chatbot.md) for the engineering log, architecture decision record, and verification walkthroughs.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | FastAPI (async Python) |
| ORM | SQLAlchemy 2.0 |
| Database | SQLite (local dev) → PostgreSQL (production) |
| Vector DB | ChromaDB (local, persistent) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, local, offline) |
| LLM | Groq (Chat Completions API) |
| Validation | Pydantic v2 |
| Rate Limiting | In-memory sliding window (Redis-ready via store interface) |
| Testing | pytest (63 tests, mocked LLM, no API costs in CI) |

## Project Structure

```
rrvdxb-chatbot/
├── app/
│   ├── main.py              # FastAPI app factory + exception handlers + catch-all middleware
│   ├── core/                # Config, DB engine, dependencies
│   │   ├── config.py        # Pydantic BaseSettings (.env loader)
│   │   ├── database.py      # SQLAlchemy engine, SessionLocal, Base
│   │   └── dependencies.py  # get_db, get_current_user_id (stub, JWT seam)
│   ├── middleware/          # Cross-cutting HTTP concerns
│   │   ├── __init__.py
│   │   ├── rate_limit.py    # Per-user rate limiting (in-memory store, Redis-ready)
│   │   └── error_handler.py # Centralized exceptions + canonical JSON error shape
│   ├── models/              # SQLAlchemy ORM models
│   │   └── chat_history.py
│   ├── schemas/             # Pydantic request/response schemas
│   │   └── chatbot_schema.py
│   ├── api/                 # API routers
│   │   └── v1/
│   │       ├── router.py    # Aggregates all v1 routes
│   │       └── endpoints/
│   │           └── chatbot.py  # POST /api/v1/ai/chat (Private, rate-limited)
│   ├── services/            # Business logic layer
│   │   ├── chatbot_service.py  # LLM orchestration + intent routing + hallucination guard + DB persistence
│   │   ├── recommender_stub.py # Product recommendation stub
│   │   └── deal_finder_stub.py # Deal lookup stub
│   ├── ai/chatbot/          # LLM prompts, client, memory, intent, RAG
│   │   ├── prompts.py       # Guardrailed SYSTEM_PROMPT + INTENT_CLASSIFICATION_PROMPT
│   │   ├── llm_client.py    # Groq SDK wrapper (singleton client)
│   │   ├── memory.py        # DB-backed conversation memory
│   │   ├── intent.py        # Regex + LLM intent classifier (IntentResult)
│   │   └── rag/             # vector_store.py + retriever.py
│   └── mock_data/           # Seed data (products + FAQs)
│       ├── products.json
│       └── faqs.json
├── scripts/
│   └── build_vector_store.py  # Build/persist the FAQ Chroma index (run once)
├── tests/
│   ├── test_chatbot.py      # Chat flow, memory, RAG, auth/rate-limit/errors, hallucination guard (23)
│   ├── test_intent.py       # Intent recognition: regex + LLM fallback + plural forms (21)
│   ├── test_retriever.py    # Retriever unit tests, mocked search_faqs (6)
│   ├── test_recommender_stub.py  # Recommender stub, real catalog matching (4)
│   └── test_schemas.py      # Schema validation: ChatRequest edges + ChatResponse round-trips (9)
├── docs/
│   └── chatbot.md           # Engineering log + architecture decision record
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` installs `torch`. On Windows, if pip fails with `WinError 206` (filename too long), enable the Long Paths registry setting or relocate the project to a shorter path.

### 2. Configure Environment

```bash
cp .env.example .env   # Linux/Mac
# copy .env.example .env  # Windows
```

Edit `.env`:

```bash
GROQ_API_KEY=gsk-...           # get from console.groq.com
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
LLM_TIMEOUT_SECONDS=3.0        # hard LLM call timeout (s), guards the <4s NFR
INTERNAL_JWT_SECRET=...        # min 32 chars
```

### 3. Build the FAQ Vector Store

```bash
python scripts/build_vector_store.py
```

Persists the Chroma collection to `app/ai/chatbot/rag/chroma_db/` (git-ignored). Re-running is safe — it is idempotent.

### 4. Run the Server

```bash
uvicorn app.main:app --reload
```

API docs: http://localhost:8000/docs

### 5. Test the Endpoint

```powershell
# Authenticated chat turn
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/chat" -Method POST `
  -Headers @{"X-User-Id"="1"; "Content-Type"="application/json"} `
  -Body '{"message":"I am looking for a gift"}'

# Follow-up with context (same user)
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/chat" -Method POST `
  -Headers @{"X-User-Id"="1"; "Content-Type"="application/json"} `
  -Body '{"message":"Something under 500 AED"}'
```

The response includes `intent`, `confidence`, and (for relevant intents) `recommended_products` or `deal`.

## Centralized Error Responses

Every client-facing error uses one JSON shape — no stack traces, no secrets:

```json
{"error": "<title>", "detail": "<human message>", "status_code": <int>}
```

| Situation | Status | Body |
|-----------|--------|------|
| Missing `X-User-Id` header | 401 | `{"error": "Authentication required", "detail": "X-User-Id header missing", "status_code": 401}` |
| Non-integer `X-User-Id` | 400 | `{"error": "Validation error", "detail": "X-User-Id must be an integer", "status_code": 400}` |
| Malformed request body | 400 | `{"error": "Validation error", ...}` |
| Rate limit exceeded (21st req/min/user) | 429 | `{"error": "Rate limit exceeded", "detail": "20 requests per minute allowed", "status_code": 429}` |
| Unexpected internal error | 500 | `{"error": "Internal server error", "detail": "Something went wrong", "status_code": 500}` |

## Running Tests

```bash
pytest -q
```

Expected: `63 passed`. All LLM calls are mocked — fast, deterministic, zero API cost.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | Yes | `sqlite:///./rrvdxb.db` | SQLAlchemy DB URI |
| `LLM_PROVIDER` | Yes | `groq` | Currently only groq is wired |
| `GROQ_API_KEY` | Yes | — | Groq API key |
| `OPENAI_API_KEY` | No | — | Only used if `LLM_PROVIDER=openai` |
| `LLM_MODEL` | Yes | `llama-3.1-8b-instant` | Model name for Groq |
| `INTERNAL_JWT_SECRET` | Yes | — | Min 32 chars for JWT signing |
| `DEBUG` | No | `True` | FastAPI debug mode |
| `LLM_TIMEOUT_SECONDS` | No | `3.0` | Hard LLM call timeout (s) — protects the <4s NFR |

RAG embeddings run entirely offline — no additional API key required.

## Data Contract Note

Our FastAPI response schema uses **snake_case** (e.g., `recommended_products`, `confidence`), while the project specification examples use **camelCase** (`recommendedProducts`). This is a known integration point to verify with the frontend/backend team before production deployment.

## What Changes When the Real Auth API Ships

1. `get_current_user_id` decodes the `Authorization` JWT, validates signature/expiry against `INTERNAL_JWT_SECRET`, and returns the `sub` claim.
2. Every endpoint that depends on `get_current_user_id` gets real authentication with **zero route changes** — dependencies are the only seam.
3. Rate-limit keys stay per-user; the store swaps from `InMemoryRateLimitStore` to a Redis-backed one shared across workers.
4. The `X-User-Id` header can be dropped entirely once JWT carries identity.

## Maintainer

Ameema Rashid — AI Lead, RRVDXB Chatbot
TechNexus Virtual University
