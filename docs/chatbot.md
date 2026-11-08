# RRVDXB AI Shopping Chatbot: Architecture and Engineering Notes

> **How to read this file.** Part 1 describes the code as it is now (checked
> against the code on 2026-11-08). Part 2 is the original build log
> (Milestones 1-10), kept for history. It describes earlier versions, for
> example an `X-User-Id` header instead of bearer tokens, a 0.6 retrieval
> threshold and `create_all` instead of Alembic. Where the two parts disagree,
> Part 1 is correct.
>
> Setup and Cloud Run: [`DEPLOYMENT.md`](../DEPLOYMENT.md). Retrieval
> evaluation: [`data/eval/README.md`](../data/eval/README.md). Project page:
> [`README.md`](../README.md).

---

# Part 1: Current state

## Overview

A FastAPI service with "Sara", an AI shopping assistant for the RRVDXB store
(markets: UAE, KSA, Pakistan, UK). By Ameema Rashid. For each message it:

- classifies the intent (regex fast path, then an LLM classifier);
- answers policy questions from a local FAQ vector index (RAG);
- returns products and a deal from two stub services;
- keeps the last 5 turns per user as context;
- checks every currency figure in the reply against the data the model was
  given, and replaces the reply if a figure is not grounded.

The catalog (10 products), the FAQs (12) and the deal are mock data. Groq is the
only LLM provider. Without a Groq key, or when the call fails or times out,
the reply is fixed fallback text built from the data retrieved for that
message (`fallback_used: true`).

## Request flow

```
GET  /                            demo page (one static HTML file, hash-based CSP)
POST /api/v1/auth/demo-token      anonymous token, only when DEMO_TOKENS_ENABLED=true
                                  (5 per minute per client IP)
POST /api/v1/ai/chat
  ├─ get_current_user_id          HS256 bearer token -> "sub"        401
  ├─ check_rate_limit             20/min per user, 20/min per IP     429 + Retry-After: 60
  ├─ ChatRequest validation       1-2000 characters, not blank       400
  ▼
chatbot_service.handle_chat_message      one 4.0 s budget for the bounded steps
  1. classify_intent              regex fast path, else Groq (cap 1.0 s, temperature 0)
  2. load_recent_history          last 5 turns (worker thread)
  3. route by intent
       product_faq        retrieve_faq_context (cap 1.0 s, similarity >= 0.45)
                          -> RAG prompt; no match -> "do not invent" note + catalog
       recommend_product  recommender stub (up to 3 products); none -> catalog
       deal_inquiry       deal stub -> offer block; none -> catalog
       track_order_help   tracking guidance
       general_chat       catalog context
  4. send_chat_message            Groq (cap min(3.0 s, budget left));
                                  timeout or error -> fallback_reply
  5. validate_reply_prices        price guard; on failure the reply is replaced
  6. save_turn                    chat_history with the classified intent (worker thread)
  7. ChatResponse
```

The intent, retrieval and reply steps each get `min(own cap, time left)`. The
two database calls run in worker threads, so they do not block the event loop,
but they are not bounded by the budget.

## Files

| File | Purpose |
|---|---|
| `app/main.py` | App factory, exception handlers, catch-all error middleware, lifespan (Alembic upgrade, RAG warm-up), `GET /` and `GET /health` |
| `app/demo_page.py`, `app/static/index.html` | Demo page; the CSP hashes of its inline script and style are computed at start-up |
| `app/api/v1/router.py` | Mounts `/api/v1/auth` and `/api/v1/ai` |
| `app/api/v1/endpoints/auth.py` | `POST /api/v1/auth/demo-token` |
| `app/api/v1/endpoints/chatbot.py` | `POST /api/v1/ai/chat` (thin; auth and rate-limit dependencies) |
| `app/core/config.py` | All settings (environment or `.env`); `.env.example` lists every one |
| `app/core/security.py` | HS256 signing and verification; refuses to start with a secret under 32 characters (except `APP_ENV=test`) |
| `app/core/dependencies.py` | `get_db`, `get_current_user_id` |
| `app/core/client_ip.py` | Client IP for per-IP limits (`TRUST_PROXY`, `TRUST_PROXY_HOPS`) |
| `app/core/database.py` | Engine for SQLite or PostgreSQL (psycopg 3); SQL parameters hidden |
| `app/core/logging_setup.py` | Root log handler; noisy third-party loggers kept at WARNING |
| `app/db/migrate.py`, `app/db/migrations/` | Alembic (one migration: `chat_history`) |
| `app/middleware/error_handler.py` | `AuthenticationError`, `RateLimitExceeded` and the handlers for the canonical error body |
| `app/middleware/rate_limit.py` | In-memory sliding-window limiter behind a `RateLimitStore` interface |
| `app/models/chat_history.py` | `chat_history` table |
| `app/schemas/chatbot_schema.py` | `ChatRequest`, `ChatResponse`, `RecommendedProduct` |
| `app/services/chatbot_service.py` | The request flow above |
| `app/services/fallback_reply.py` | Fallback text from retrieved data |
| `app/services/recommender_stub.py`, `app/services/deal_finder_stub.py` | Stand-ins for the recommender and deal services |
| `app/ai/chatbot/intent.py` | Regex fast path and LLM classifier (confidence threshold 0.7) |
| `app/ai/chatbot/prompts.py` | System prompt with the scope rule, classifier prompt, RAG prompt builder |
| `app/ai/chatbot/llm_client.py` | Groq client (`max_retries=0`, timeout = `LLM_TIMEOUT_SECONDS`) |
| `app/ai/chatbot/memory.py` | `save_turn`, `load_recent_history`, `format_history_for_prompt` |
| `app/ai/chatbot/price_guard.py` | Post-generation price checks (limits listed in its docstring) |
| `app/ai/chatbot/rag/vector_store.py`, `rag/retriever.py` | FAQ index (all-MiniLM-L6-v2 + ChromaDB) and the similarity gate |
| `app/mock_data/products.json`, `faqs.json` | 10 products (with aliases), 12 FAQs |
| `scripts/` | `build_vector_store.py`, `smoke_check.py`, `benchmark.py`, `eval_retrieval.py` (see DEPLOYMENT.md 3.8) |
| `data/eval/` | Retrieval dev and held-out sets |
| `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` | Image, local compose, CI |

## Implemented vs mocked

| Part | Today | Notes |
|---|---|---|
| Auth | HS256 bearer tokens; `sub`, `iat`, `exp`, `iss`, `aud` required; other algorithms rejected | No user accounts. Anonymous demo tokens (random `demo-<hex>` subject, 30 min) when enabled |
| Recommender | Keyword stub over `products.json`: `get_recommendations(user_id: str, message: str) -> list[dict]` | Matches name, brand and category words; prefix or suffix matches only for keywords of 4+ characters |
| Deals | Keyword stub `get_deals(user_id: str, message: str) -> dict \| None` | One fixed mock offer |
| FAQ retrieval | all-MiniLM-L6-v2 embeddings, ChromaDB on disk, threshold 0.45 | Model and index are built into the Docker image |
| Rate-limit store | In memory, one process | A shared store (e.g. Redis) behind `RateLimitStore` would be needed for several instances |
| Chat history | SQLite by default, PostgreSQL through `DATABASE_URL` | Alembic upgrade at start-up when `DB_AUTO_MIGRATE=true` |

## API

### `POST /api/v1/auth/demo-token`

404 unless `DEMO_TOKENS_ENABLED=true`. Limited to 5 per minute per client IP
(429). Response: `{"access_token": "...", "token_type": "bearer", "expires_in": 1800}`.
Each token has a new random subject, so each visitor has their own history and
their own per-user limit.

### `POST /api/v1/ai/chat`

Header `Authorization: Bearer <token>`. Body:

```json
{"message": "Can you recommend a perfume?"}
```

`message`: 1-2000 characters and not only whitespace. An optional integer
`user_id` is accepted and ignored (the token identifies the user).

Response (an offline run without a Groq key, so `fallback_used` is true):

```json
{
  "reply": "Here are some products from our catalog that match your request:",
  "recommended_products": [
    {"id": 4, "name": "Chanel Coco Mademoiselle Body Spray", "price": 189.0, "currency": "AED",
     "category": "Perfumes", "brand": "Chanel", "reason": "Matches your request for perfume"},
    {"id": 7, "name": "Harak Perfume Oud Edition", "price": 450.0, "currency": "AED",
     "category": "Perfumes", "brand": "Harak", "reason": "Matches your request for perfume"}
  ],
  "deal": null,
  "intent": "recommend_product",
  "confidence": 1.0,
  "response_time_ms": 17,
  "fallback_used": true
}
```

| Field | Meaning |
|---|---|
| `reply` | Reply text (after the price guard) |
| `recommended_products` | Products from the recommender, or `null` |
| `deal` | Offer text, or `null` |
| `intent` | One of `recommend_product`, `track_order_help`, `deal_inquiry`, `product_faq`, `general_chat` |
| `confidence` | 1.0 for a regex match; the classifier's value otherwise. Below 0.7 the intent is `general_chat`; 0.0 means the classifier was unavailable or failed |
| `response_time_ms` | Server time to handle the chat message |
| `fallback_used` | `true` when the reply is fallback text because the LLM call timed out or failed. A price-guard replacement does not set it |

The schema marks `intent`, `confidence`, `response_time_ms` and `fallback_used`
as optional for backward compatibility; the service always fills them.

**Errors** all use one body, `{"error": "...", "detail": "...", "status_code": <int>}`:

| Status | When |
|---|---|
| 400 | Invalid body (`"Request body is not valid JSON"` for malformed JSON) |
| 401 | `"Bearer token missing"` or `"Invalid or expired token"` |
| 404 / 405 | Unknown path / wrong method (405 keeps the `Allow` header) |
| 429 | Per-user, per-IP or demo-token limit; `Retry-After: 60` |
| 500 | `"Something went wrong"`; the traceback goes to the server log only |

LLM failures are not errors: the endpoint answers 200 with fallback text.

### `GET /health`

Always 200: `{"status": "ok" | "degraded", "service": "...", "rag_ready": true | false}`.
`degraded` means the start-up warm-up of the embedding model and index failed.

## Guards and fallbacks

- **Fallback text** (`fallback_reply.py`): the matched FAQ answer, a short
  intro for the returned products or deal, the tracking guidance, or a help
  message for `general_chat`. When nothing was found, the text says so and
  offers human support. On a timeout, "I'm taking longer than usual. Please try
  again in a moment." comes first.
- **Price guard** (`price_guard.py`): every currency figure in the reply must
  appear in what the model saw (prompt, context, the customer's message,
  history), and a figure attached to a named product must be that product's
  catalog price. Currencies: AED, USD, SAR, GBP, PKR, INR, Rs, the $, £, ₨ and ₹
  signs, and dirham(s). Bare numbers, percentages, stock and delivery times are
  not checked. On failure the whole reply becomes "I can't confirm that price
  right now, so I won't quote it. Please ask about a specific product from our
  catalog." The guard runs before the turn is saved.
- **Prompt rules**: a scope rule (answer only about RRVDXB); plain text without
  Markdown; prices exactly as given, no currency conversion; when products or a
  deal are returned as structured fields, the reply only introduces them.
  These are instructions; whether the live model follows them is not measured.
- **Retrieval gate**: FAQ chunks ground an answer only at cosine similarity
  >= 0.45. Measured on dev and held-out sets, including 4 known off-topic
  leaks: see `data/eval/README.md`.

## Logging and privacy

- One root handler; `LOG_LEVEL` (default INFO).
- App logs contain no message text, tokens or keys (tests capture the logs on
  every reply path). They do contain character counts, intents, step timings
  (one line per request) and, on a per-IP 429, the client IP.
- `sqlalchemy.engine`, `httpx`, `httpcore` and `sentence_transformers` stay at
  WARNING. Embedding calls run without progress bars. SQL parameters are hidden
  (`hide_parameters=True`).

## Testing

- `pytest` with every LLM call mocked and no API keys. Marker `db` tests also
  run against PostgreSQL 16 in CI. Marker `chroma` tests build and query a real
  index with the local model.
- CI (`.github/workflows/ci.yml`): ruff, the full suite with a 92% coverage
  gate on SQLite, then the `db` tests on PostgreSQL.
- Not covered by tests: live Groq latency and answer quality (run
  `scripts/benchmark.py` after deploying), whether the model follows the prompt
  rules, Cloud Run behaviour, and how the demo page renders in a browser.

## Architecture decisions

| Decision | Why | Trade-off |
|---|---|---|
| Groq, called through a thin wrapper | Fast inference for the 4 s budget | One provider; a retired model means fallback replies until `LLM_MODEL` is changed |
| Hand-rolled orchestration, no LangChain | Full control of prompts, timeouts and fallbacks | More code to maintain |
| Regex fast path before the LLM classifier | No LLM cost or latency for common phrasings | Patterns need upkeep |
| Local embeddings + ChromaDB | Offline, no per-query cost | Small model; threshold leaks documented |
| Post-generation price guard | Prompt rules are not enforcement | Covers currency figures only |
| Fallback text from retrieved data | Useful answers without the LLM | Less natural than a model reply |
| HS256 bearer tokens, opt-in anonymous demo tokens | Identity from a verified token, not a header | No user accounts |
| Custom in-memory rate limiter | No extra dependency; per-user and per-IP keys | One instance only (Cloud Run max instances 1) |
| Alembic migrations at start-up | One schema history for SQLite and PostgreSQL | Turn off for several instances |
| `asyncio.to_thread` + `wait_for` around blocking calls | Bounds the synchronous Groq SDK and retrieval | An abandoned call keeps its thread until the client timeout |
| Exact dependency pins | CI builds what was tested | Manual upgrades (see DEPLOYMENT.md, Known limits) |
| Model and index built into the image | No download at start-up; works offline | 2.64 GB image (measured locally) |

## Gotchas

- **WinError 206 (path too long) on Windows** when installing torch: enable
  long paths or use a shorter project path.
- **Em-dash shown as `â` in a Windows terminal**: a console code-page issue;
  the JSON is correct.
- **Chroma telemetry noise**: off through `anonymized_telemetry=False` and the
  `posthog==3.5.0` pin.
- **Rate-limit state between tests**: an autouse fixture resets the in-memory
  store before and after every test.
- **Fresh clone without `chroma_db/`**: the index is built by the start-up
  warm-up (`RAG_WARMUP=true`, needs the model, which downloads once) or by
  `python scripts/build_vector_store.py`.
- **Price guard replaced a correct reply**: check that the figure's currency is
  one the guard recognises and that the product name or alias in the reply
  matches `products.json`.
- **Ports**: the image listens on `$PORT` (default 8080); compose maps 8000.

## Open items

- Streaming replies.
- A shared rate-limit store and more than one instance.
- A Docker build in CI.
- Live latency and answer quality, measured after a deployment.
- Guarding percentages, stock and delivery promises, and flagging a
  price-guard replacement in the response.
- Upgrading FastAPI, Starlette and Uvicorn (DEPLOYMENT.md, Known limits).

---

# Part 2: Build log (historical)

> Written during the original 10-day build and kept unchanged. It describes
> earlier versions of the code (for example the `X-User-Id` header, the 0.6
> threshold, `create_all`, a generic fallback message and the test counts of
> that time). For current behaviour, see Part 1.

## Milestone 1: Project Scaffold

- FastAPI scaffold with clean layered architecture (`api/`, `services/`, `ai/`, `core/`)
- Pydantic v2 `Settings` with `.env` support (`pydantic-settings`)
- SQLAlchemy 2.0 + SQLite with `SessionLocal` dependency injection
- `chat_history` ORM model matching the required schema
- Request/response Pydantic schemas with strict validation
- `POST /api/v1/ai/chat` placeholder endpoint
- Service-layer stub with DB persistence on every turn
- Guardrailed system prompt (persona + guardrails against price/policy invention)
- Mock products (10 items) and FAQs (seed data)
- Basic pytest suite; `.env.example` and `README.md`

---

## Milestone 2: Groq LLM Integration

- Created `app/ai/chatbot/llm_client.py` — singleton Groq client (connection reuse) with error handling
- Replaced the placeholder echo with real LLM calls via `send_chat_message()`
- Product catalog from `products.json` injected into the system prompt as runtime context
- Graceful fallback when Groq fails (rate limit, timeout, auth error) — canned persona message, never crashes the endpoint
- LLM mocked in tests via `unittest.mock.patch` — fast, free, deterministic; no real API calls in CI

**New/updated files:** `llm_client.py`, `chatbot_service.py`, `test_chatbot.py`, `prompts.py`, `README.md`, `chatbot.md`

---

## Milestone 3: DB-Backed Conversation Memory

- Created `app/ai/chatbot/memory.py` with three primitives:
  - `save_turn(user_id, message, response, intent)` — persists to `chat_history`
  - `load_recent_history(user_id, limit=5)` — loads the last N turns
  - `format_history_for_prompt(turns)` — formats as `User: ...\nAssistant: ...` for prompt injection
- Configurable context window (N=5) — loads recent turns only, keeping token count bounded
- `chatbot_service.py` loads history before each LLM call and saves the turn after the response
- `llm_client.py` accepts a `history` parameter and injects prior turns
- Anonymous-user fallback: `user_id` → `0` when `None`, so persistence always succeeds
- No LangChain — implemented with SQLAlchemy for full control

**Testing:** `test_conversation_memory_persists_across_turns` (in-memory SQLite + `StaticPool`) asserts the second call's prompt contains the first turn's context.

---

## Milestone 4: Intent Recognition & Routing

- Created `app/ai/chatbot/intent.py` — hybrid classifier with two stages:
  1. **Regex fast-path** — zero LLM cost, `confidence=1.0`; covers `recommend_product`, `track_order_help`, `deal_inquiry`, `product_faq`
  2. **LLM fallback** — reuses the Groq client; no second API client
- `IntentResult` Pydantic model: `intent`, `confidence`, `entities`
- Defense-in-depth JSON parsing: strips markdown fences, `json.loads` in try/except, enum allow-list, confidence clamped to `[0.0, 1.0]`
- Confidence threshold `0.7` — low-confidence labels override to `general_chat` (original confidence kept for observability)
- Any API/parse failure degrades to `general_chat` with `confidence=0.0`
- `INTENT_CLASSIFICATION_PROMPT` with strict JSON contract + few-shot examples
- Routing in `chatbot_service.py` via `_build_system_prompt_for_intent()`:
  - `recommend_product` → product catalog context
  - `product_faq` → RAG pipeline (resolved in Milestone 6)
  - `deal_inquiry` → Deal Finder (resolved in Milestone 7)
  - `track_order_help` → order-tracking guidance
  - `general_chat` → standard flow
- Intent persisted to `chat_history.intent`; `ChatResponse` exposes `intent` and `confidence`

**Post-review hardening:** word boundaries on all regexes (`suggestion` ≠ `suggest`); `gift` restricted to shopping phrases; `\bbuy\b` added; low-confidence instruction in the prompt; catalog header only injected when context is non-empty; regression tests for each.

**Testing:** `pytest -q` → 20 passed

---

## Milestone 5: RAG Fundamentals & Vector Store

- Created `app/ai/chatbot/rag/vector_store.py`:
  - `load_faqs()` — safe reads of `faqs.json` (missing file / malformed JSON / invalid entries handled)
  - `build_vector_store()` — one Q&A pair per chunk, embedded locally (`all-MiniLM-L6-v2`, 384-dim), persisted to ChromaDB; **idempotent** (upsert + stale-delete)
  - `get_vector_store()` — lazy build only when `chroma.sqlite3` is not on disk
  - `search_faqs(query, top_k)` — cosine similarity search with ranking + score
- Embedded entirely locally — no OpenAI key; persists to git-ignored `chroma_db/`
- Expanded `faqs.json` to 11 coverage entries (Shipping UAE/KSA/Pak/UK, Returns, Payments, Free-shipping AED 489, Warranty, Tracking, Authenticity, Discounts)
- `scripts/build_vector_store.py` — standalone build script, optional `--query` demo
- First-run model download handled gracefully with offline-cache fast path (`local_files_only`)
- Chroma telemetry suppressed (`anonymized_telemetry=False` + pinned `posthog==3.5.0`)
- Added `chromadb==0.5.5`, `sentence-transformers==3.0.1`, `httpx==0.27.2` to requirements

**Manual verification:** `python scripts/build_vector_store.py --query "do you ship to Pakistan?"` → Pakistan FAQ top hit (~0.79 similarity)

**Testing:** `pytest -q` → 20 passed (RAG not yet wired to chat)

---

## Milestone 6: RAG Pipeline Integration

- Created `app/ai/chatbot/rag/retriever.py`:
  - `retrieve_faq_context(query, k=3, similarity_threshold=0.6)` reuses `search_faqs()`/`get_vector_store()` (never rebuilds the index)
  - Returns only chunks above the relevance threshold; clamp `k` (1–10) and threshold (0–1)
  - Returns `[]` gracefully on blank query, missing/empty store, or search failure
  - Output contract: `[{"question", "answer", "similarity"}]`, most-similar first
- `build_rag_system_prompt()` in `prompts.py`: Day-1 guardrails preserved verbatim + flagged `FAQ CONTEXT:` block + cite-only / never-invent instructions
- `send_chat_message()` extended with optional `system_prompt_override` (replaces base prompt, skips catalog injection; history still appended) — client stays a thin Groq wrapper
- Wired into the `product_faq` path: chunks found → RAG prompt; no chunks → polite no-match fallback with catalog re-injected
- Intent regex extended for shipping-destination / delivery-options / stock questions; `track_order_help` precedence preserved

**Tuning note — why 0.6:** the exact "return policy" FAQ match scores ~0.66, so 0.7 would wrongly drop it. 0.6 keeps strong hits (shipping ~0.79, returns ~0.66) while excluding junk (0.23–0.45) — measured empirically, not guessed.

**Manual verifications:** "do you ship to Pakistan?" → `product_faq` grounded in the Pakistan chunk; "what is your return policy?" → grounded (14-day policy); "warranty policy for a plumbus?" → graceful no-match + human-support hand-off

**Testing:** `pytest -q` → 32 passed

---

## Milestone 7: Recommendations, Deals, Fallback & Timeout

### NFR protection architecture

The Groq SDK is synchronous; an unbounded blocking call on the async loop can consume the <4s budget. The fix:

```python
await asyncio.wait_for(
    asyncio.to_thread(functools.partial(send_chat_message, ...)),
    timeout=settings.llm_timeout_seconds,
)
```

`to_thread` moves the blocking call off the event loop (interruptible); `wait_for` bounds it. With ~1s of pre-LLM overhead, a 3.0s timeout protects the NFR.

### Recommendation stub

`app/services/recommender_stub.py` — `get_recommendations(user_id, message)`: safe catalog loader (candidate-path search, never raises), keyword matching via word prefix/suffix equality (`phone` → `iPhone`, not `headphones`), plural normalization, top-3 results with `reason`.

### Deal finder stub

`app/services/deal_finder_stub.py` — `get_deals(user_id, message)`: keyword-gated Summer Sale deal (`SUMMER20`, 20% off); `None` otherwise (never invents an offer).

### Service rewire

- `recommend_product` → stub → `RecommendedProduct` list + `RECOMMENDED PRODUCTS:` context; no matches → catalog fallback
- `deal_inquiry` → stub → `ACTIVE OFFER:` context + human-readable `deal`; no deal → general flow
- Timeout (`asyncio.TimeoutError`) → `"I'm taking longer than usual. Please try again in a moment."` (slow, not broken)
- Provider error (`RuntimeError`) → existing "human support" message
- Fallback turns persist as `general_chat` (`_FALLBACK_PERSIST_INTENT`) so DB constraints hold; original intent still returned in `ChatResponse`
- API-only confidence floor: sub-`0.7` surfaces as 0.9 in the response; intent layer keeps the raw value (DB stores the true number)

**Manual verifications:** "recommend me a phone" → `recommended_products` populated (Sony-headphones false positive fixed); "any discounts today?" → deal = "Summer Sale — 20% off (code: SUMMER20)"; empty `GROQ_API_KEY` restart → graceful 200; `LLM_TIMEOUT_SECONDS=0.1` → timeout reply within budget

**Testing:** `pytest -q` → 41 passed

---

## Milestone 8: Auth Stubs, Rate Limiting & Centralized Error Handling

### Auth rationale

Every chat request may trigger a paid LLM inference, so authentication must run **before** any LLM work to prevent cost abuse. `X-User-Id` is a stub; when the Auth API ships, `get_current_user_id` decodes a JWT with zero route changes (dependency injection is the seam).

### Rate limiting architecture

- **Per-user vs global:** a global budget lets one noisy user starve everyone; per-user gives fair independent budgets + a per-customer abuse signal.
- **In-memory vs Redis:** in-memory is fine for one dev process; with N uvicorn workers each keeps its own counter (real limit becomes N × configured) → shared Redis required in production.
- **Custom vs slowapi:** chose a custom `RateLimitStore` ABC + `InMemoryRateLimitStore`. Zero new dependencies, no decorator magic, store isolated behind an interface; Redis swap is a one-line change in the guard.

### Implementation

- `app/middleware/error_handler.py`:
  - `AppError` base (class-level `error`/`status_code`, instance `detail`, `to_dict()`)
  - `AuthenticationError` (401), `ValidationError` (400), `RateLimitExceeded` (429), `AIServiceError` (502, for future non-recoverable AI-service failures)
  - Canonical body `{"error", "detail", "status_code"}` on every error path
  - Handlers: `app_error_handler`, `validation_error_handler` (422 → 400), `http_exception_handler` (404/405 → same shape), `unhandled_exception_handler` (generic 500, logs server-side, never exposes stacks/secrets)
- `app/middleware/rate_limit.py`: `RateLimitStore` ABC (`allow_request(key, limit, window_seconds)`); `InMemoryRateLimitStore` sliding-window deque + lock; `check_rate_limit` dependency keyed by resolved `X-User-Id`; 20 requests/minute/user; 21st → 429
- `app/core/dependencies.py`: `get_current_user_id` reads `X-User-Id` (string) → int; missing → 401; non-integer → 400; `# TODO: Replace with JWT validation when Auth API is ready`
- `app/api/v1/endpoints/chatbot.py`: auth + rate-limit guard wired via `Depends`; all errors flow to centralized handlers — no catch-and-silence
- `app/main.py`: registers exception handlers + `UnhandledErrorMiddleware` (guarantees a clean JSON 500 even in debug mode — Starlette's `ServerErrorMiddleware` would otherwise emit a raw traceback)

### Security audit

- `GROQ_API_KEY` / `INTERNAL_JWT_SECRET` read only via `app/core/config.py` — nothing hardcoded
- `.env.example` lists every required variable (this milestone adds none)
- `.env*` is git-ignored; `INTERNAL_JWT_SECRET` must be rotated before production

**Manual verifications:**
- No `X-User-Id` → 401 `{"error": "Authentication required", "detail": "X-User-Id header missing", "status_code": 401}`
- `X-User-Id: abc` → 400 `{"error": "Validation error", "detail": "X-User-Id must be an integer", "status_code": 400}`
- 21 rapid requests from one user → 429 `{"error": "Rate limit exceeded", "detail": "20 requests per minute allowed", "status_code": 429}`
- Simulated internal error → 500 `{"error": "Internal server error", "detail": "Something went wrong", "status_code": 500}`, no traceback leaked

**Notes:**
- `AIServiceError` is defined but deliberately not raised today — the LLM path still degrades gracefully (200 + canned reply), which is better chatbot UX; it is the lever for truly non-recoverable upstream failure.
- The rate-limit store is a module-level singleton; tests reset it per test via an autouse fixture so quota never leaks across tests.

**Testing:** `pytest -q` → 44 passed

---

## Milestone 9: Test Suite Completion + Onboarding Docs (Day 9)

### Test gaps closed

- **`tests/test_schemas.py` (new, 9 tests)** — direct Pydantic validation, no HTTP layer:
  - `ChatRequest`: empty message, 2001-char message, missing message, wrong-type `user_id` all raise `ValidationError`; valid payload and optional `user_id` pass.
  - `ChatResponse`: full round-trip (dict → model → dict) with `recommended_products`/`deal`/`intent`/`confidence`; optional fields default to `None`; out-of-range confidence rejected.
- **`tests/test_chatbot.py` (+6 tests)**:
  - `test_memory_does_not_leak_between_users` — cross-pollution guard: user B's prompt never contains user A's turns, while A still sees their own.
  - `test_track_order_help_uses_tracking_guidance_and_keeps_history` — asserts the `ORDER TRACKING GUIDANCE` block (incl. "NEVER invent a tracking number") is in the prompt and history is still appended on the next turn.
  - `test_general_chat_uses_standard_prompt_with_catalog` — explicit: base persona prompt, no RAG override, no tracking block, catalog context injected.
  - `test_unknown_route_returns_canonical_404_body` / `test_wrong_method_returns_canonical_405_body` — `http_exception_handler` (previously untested) returns the same `{"error","detail","status_code"}` shape as our typed errors.
  - `test_malformed_request_body_returns_canonical_400` — `validation_error_handler` (previously untested) maps FastAPI's 422 to our canonical 400.

### Review pass

- The only genuine TODO in the codebase is the JWT swap in `app/core/dependencies.py` — kept. No stale placeholders remain.
- `app/middleware/rate_limit.py`: removed unused imports (`Session`, `get_db`); added missing Google-style docstrings to `RateLimitResult` and the `InMemoryRateLimitStore` methods.
- Docstrings standardized on Google style throughout; type hints present on every public function.
- `requirements.txt`: no duplicates. `colorama>=0.4.6` is not imported anywhere (transitive Windows console helper — safe to keep or drop). `openai==1.10.0` is not imported by code but backs the `LLM_PROVIDER=openai` config option — kept deliberately.

### Day 9 hardening (post-live-verification fixes)

**1. Plural forms restored to the regex fast-path** (`app/ai/chatbot/intent.py`)

Live testing exposed that `"any discounts today?"` was NOT caught by the regex layer — `\bdiscount\b` (singular-only) missed `discounts`, so every plural phrasing burned a paid LLM classification call. Fixed patterns:
- `track_order_help`: `\btrack\b` → `\btrack(?:ing|s)?\b` (also `tracks`, `tracking`)
- `deal_inquiry`: `\bdeals?\b|\bdiscounts?\b|\bsales?\b|\boffers?\b|\bcoupons?\b|\bpromos?\b`

Regression tests (`test_regex_deal_plural_forms_hit_fast_path`, `test_regex_track_inflected_forms_hit_fast_path`) assert `confidence == 1.0` AND that the mocked LLM is never called across six deal phrasings and three track phrasings.

**2. Post-generation hallucination guard** (`app/services/chatbot_service.py`)

Live testing caught Sara quoting **"AED 6,499"** for the iPhone 14 Pro Max — the catalog says **4,699 AED**. The FAQ chunk that was retrieved said nothing about prices, but the model invented a figure anyway: prompt guardrails ("NEVER invent prices") are instructions, not enforcement. The fix adds a **mechanical** check between the LLM call and persistence:

- `_PRICE_PATTERN` matches currency-tagged figures in BOTH spellings — `4,699 AED` and `AED 6,499` — tolerating commas, decimals, and separators.
- `_extract_price_figures()` normalizes (commas and decimal tails stripped) so `4,699.00 AED` (catalog format) equals `4,699 AED` (reply format).
- `_validate_prices_against_grounding()` builds the allowlist from everything the model actually SAW this turn: the system prompt (including any RAG `FAQ CONTEXT:` block), the injected catalog/recommendation/deal context, the customer's own message (they may quote a budget), and the history.
- Any reply figure missing from the allowlist → the WHOLE reply is replaced with a safety message (`"Let me double-check that exact figure for you — I don't want to quote anything that isn't accurate. One moment, please."`) and the incident is logged at `WARNING` with the invented figures.
- The replacement runs **before** `save_turn`, so hallucinated text never reaches the DB — and therefore can never poison the next turn's memory.

Guard tests: `test_hallucinated_price_in_reply_is_replaced` (currency-first `AED 6,499` with no grounding → replaced, and the DB row stores only the safe text) and `test_grounded_price_from_faq_context_is_kept` (489 AED, which IS in the chunk, passes untouched — no false positives).

**3. Test data made catalog-truthful** — `test_conversation_memory_persists_across_turns`'s mock reply used to quote a "Lacoste L.12.12 (299 AED)" that does not exist in `products.json` (the real catalog has no 299 AED item; the Ultraboost is 749 AED). Under the guard that reply would correctly be flagged ungrounded, so the mock now quotes the real prices (Men's Classic Polo Shirt 399 AED, Harak Perfume Oud Edition 450 AED) — the test still proves memory, and the figures are now consistent with the grounding the guard enforces.

### Documentation

- Added to this doc: Overview, architecture diagram, file inventory, implemented-vs-mocked table, API reference, and NFR sections — a new developer can onboard from this document alone.

**Testing:** `pytest -q` → 63 passed, no skips

---

## Milestone 10: Deployment Readiness + CI (Day 10)

### Container image

- `Dockerfile` — single-stage `python:3.11-slim` (the same Python version as local dev AND the CI runner, so behavior matches everywhere):
  - Dependencies install first (layer caching). Every dependency ships manylinux wheels, so **no compiler toolchain is needed**; if a future dependency lacks a wheel, add `build-essential gcc g++` to the image.
  - `RUN python scripts/build_vector_store.py` bakes the FAQ index into the image at **build time** — the ~90MB embedding model downloads once during `docker build` and containers start RAG-grounded with zero internet at runtime.
  - Runtime command is shell-form so the port expands: `CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]`. Exec-form (`["uvicorn", ..., "${PORT:-8000}"]`) would pass the literal string — a classic container bug. `$PORT` is the Render/platform convention; `8000` is the local fallback.
  - `HF_HOME` is set inside the image so the baked model cache lives in the `/app` layer — deterministic image contents regardless of runner OS.
- `.dockerignore` — prunes the build context to what the image needs: never ships `.env`/secrets, venv, tests, docs, or the local `chroma_db/` (regenerated by the build step).

### Render / `$PORT` notes

- **Ephemeral disk**: platforms like Render wipe the instance filesystem on every deploy. SQLite (`rrvdxb.db`) and runtime Chroma writes reset — but boot is **self-healing**: `Base.metadata.create_all` recreates the tables at startup (`app/main.py`), and the RAG index is baked into the image, so answers stay grounded.
- **Not durable**: conversation history does not survive a redeploy on such platforms. Durable memory requires PostgreSQL (set via `DATABASE_URL`) + a shared Redis/vector store — see "What's Next".

### Config knob: RATE_LIMIT_PER_MINUTE

- `app/core/config.py` gains `rate_limit_per_minute: int = 20` (overridable via env/`.env`) — the per-user budget was a hardcoded module constant since Day 8; ops can now tune it per environment with zero code changes.
- `app/middleware/rate_limit.py` reads it into the **same module-level names** (`RATE_LIMIT_LIMIT`) so the tests and the 429 detail message keep one source of truth. The default stays 20, so no test file changes — expected 63 passed.
- Values load once at import; changing them requires a restart (standard env-var behavior).

### Reproducible dependencies

- `groq>=0.4.0` → `groq==1.6.0`: the pin now matches exactly what was installed and tested. Loose upper bounds are the classic "green locally, red on GitHub" cause — a new groq release could break the build with zero repo changes.
- Stale comment fixed: requirements no longer claims "LangChain added later" (LangChain was never used; orchestration is hand-rolled per the ADR).

### GitHub Actions CI

- `.github/workflows/ci.yml`: checkout → Python 3.11 → `pip install -r requirements.txt` → `pytest -q`, on every push to `main` and every pull request.
- **Zero environment required**: the suite is repo-self-contained — mocked LLM/retriever, config defaults for every field, in-memory SQLite, no vector store needed by any test. No `.env`, no API keys, no network.

**Testing:** clean-env verification — running `pytest -q` from a directory with NO `.env` (simulating the CI runner) → 63 passed; live boot from the same clean environment with `RATE_LIMIT_PER_MINUTE=3` → 3×200 then 429 on the 4th request from the same user (the knob takes effect end-to-end).
