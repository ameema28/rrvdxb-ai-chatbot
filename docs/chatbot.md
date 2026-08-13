# RRVDXB AI Shopping Chatbot — Engineering Log & Architecture Decisions

This document is the canonical engineering log and architecture decision record
(ADR) for the RRVDXB chatbot. It covers the full build history from scaffold
through auth/rate-limiting, with rationale for major choices, tuning notes, and
manual verification results.

> For the project storefront (quick start, features, tech stack), see
> [`README.md`](../README.md).

---

## Overview

The RRVDXB AI Shopping Chatbot is a FastAPI service that lets customers of the
RRVDXB premium e-commerce platform (markets: UAE, KSA, Pakistan, UK) hold a
natural-language conversation with "Sara", an AI shopping assistant. It
recognizes what the customer wants (product recommendations, order help, deals,
product/FAQ questions, or small talk), grounds answers in a local FAQ vector
store (RAG) or the product catalog, remembers recent context per user, and
returns structured replies with optional product recommendations and deal
alerts — all under a strict <4s latency budget with graceful fallbacks when the
LLM is slow or down. Built by Ameema Rashid (AI Lead, TechNexus Virtual
University), this API is the complete deliverable for the "AI Shopping
Chatbot" feature: intent recognition, RAG FAQ pipeline, context memory,
recommendation/deal integration seams, per-user rate limiting, and a
post-generation hallucination guard that mechanically blocks invented prices.

## Architecture

```
                         ┌───────────────────────────────────────────┐
                         │  FastAPI app (app/main.py)                 │
                         │  exception handlers + UnhandledErrorMW     │
                         └─────────────────────┬─────────────────────┘
                                               │ POST /api/v1/ai/chat
                                               │ (X-User-Id header)
                          ┌────────────────────▼─────────────────────┐
                          │  Dependency guards (run before handler)  │
                          │  get_current_user_id  → 401 missing      │
                          │                         → 400 non-int    │
                          │  check_rate_limit     → 429 (20/min/user)│
                          └────────────────────┬─────────────────────┘
                                               │
                          ┌────────────────────▼─────────────────────┐
                          │  chatbot_service.handle_chat_message()    │
                          └────────────────────┬─────────────────────┘
                                               │ classify_intent()
                          ┌────────────────────▼─────────────────────┐
                          │  intent.py (regex fast-path → LLM)       │
                          └────────────────────┬─────────────────────┘
                                               │ route by intent
          ┌──────────────┬──────────────┬──────┴───────┬───────────────┬──────────────┐
          ▼              ▼              ▼              ▼               ▼              ▼
┌────────────────┐┌──────────────┐┌──────────────┐┌──────────────┐┌──────────────┐
│ product_faq    ││recommend_    ││ deal_inquiry ││track_order   ││general_chat  │
│ RAG retriever  ││product       ││ deal finder  ││help          ││ + catalog    │
│ rag/retriever  ││recommender   ││ stub         ││ tracking     ││ context      │
│ → ChromaDB     ││stub          │└──────┬───────┘│ guidance     │└──────┬───────┘
│ (local embeds) │└──────┬───────┘       │         └──────┬───────┘       │
└──────┬─────────┘       │               │                │               │
       └─────────────────┴───────┬───────┴────────────────┴───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │  LLM client (Groq)           │
                  │  to_thread + wait_for,       │
                  │  timeout = LLM_TIMEOUT_SECS  │
                  └──────────────┬───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │  hallucination guard (Day 9) │
                  │  reply figures vs injected   │
                  │  context; replace if invented│
                  └──────────────┬───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │  memory.save_turn → SQLite   │
                  │  chat_history (intent kept)  │
                  └──────────────┬───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │  ChatResponse (snake_case)   │
                  └──────────────────────────────┘
```

Every arrow is one function call; every box is one module (see the file
inventory below). Errors at any layer propagate to the centralized handlers and
come back as the canonical `{"error", "detail", "status_code"}` body.

## File Inventory

| File | Purpose |
|------|---------|
| `app/main.py` | App factory, exception-handler registration, catch-all error middleware, `/health` |
| `app/api/v1/router.py` | Aggregates all v1 routers under `/api/v1` |
| `app/api/v1/endpoints/chatbot.py` | `POST /api/v1/ai/chat` — thin handler; auth + rate-limit deps; delegates to the service |
| `app/core/config.py` | Pydantic v2 settings (.env) — single source of config truth |
| `app/core/database.py` | SQLAlchemy engine, `SessionLocal` factory, declarative `Base` |
| `app/core/dependencies.py` | `get_db` session lifecycle; `get_current_user_id` (JWT swap seam) |
| `app/middleware/error_handler.py` | `AppError` hierarchy + all handlers producing the canonical error body |
| `app/middleware/rate_limit.py` | Per-user sliding-window limiter (`RateLimitStore` ABC + in-memory impl) |
| `app/models/chat_history.py` | `chat_history` ORM model (id, user_id, message, ai_response, intent, created_at) |
| `app/schemas/chatbot_schema.py` | `ChatRequest`, `ChatResponse`, `RecommendedProduct` |
| `app/services/chatbot_service.py` | Intent routing, RAG/recommender/deal orchestration, LLM timeout, hallucination guard, persistence |
| `app/services/recommender_stub.py` | Recommender interface stub (teammates' real service drops in behind it) |
| `app/services/deal_finder_stub.py` | Deal-finder interface stub |
| `app/ai/chatbot/prompts.py` | `SYSTEM_PROMPT`, `INTENT_CLASSIFICATION_PROMPT`, `build_rag_system_prompt()` |
| `app/ai/chatbot/llm_client.py` | Groq client singleton + `send_chat_message()` |
| `app/ai/chatbot/memory.py` | `save_turn` / `load_recent_history` / `format_history_for_prompt` |
| `app/ai/chatbot/intent.py` | Hybrid intent classifier (regex fast-path → LLM fallback; plural forms covered) |
| `app/ai/chatbot/rag/vector_store.py` | `faqs.json` → local embeddings → ChromaDB (build, lazy load, semantic search) |
| `app/ai/chatbot/rag/retriever.py` | `retrieve_faq_context()` with the 0.6 similarity relevance gate |
| `app/mock_data/products.json` | 10-product seed catalog |
| `app/mock_data/faqs.json` | 12 FAQ seed entries (shipping, returns, payments, warranty, tracking, stock…) |
| `scripts/build_vector_store.py` | One-shot Chroma index build, optional `--query` demo |
| `tests/test_chatbot.py` | Endpoint flow, memory (+user isolation), RAG, auth/rate-limit/errors, hallucination guard — 23 tests |
| `tests/test_intent.py` | Intent recognition: regex, LLM fallback, parsing robustness, plural forms — 21 tests |
| `tests/test_retriever.py` | Retriever relevance gate and degradation — 6 tests |
| `tests/test_recommender_stub.py` | Recommender stub catalog matching — 4 tests |
| `tests/test_schemas.py` | Schema validation: ChatRequest edges + ChatResponse round-trips — 9 tests |

## Implemented vs Mocked (integration seams)

This repo ships two placeholder services that stand in for teammates' work.
They are deliberate, contract-defining stubs — NOT missing features:

| Capability | Today | Swap path |
|-----------|-------|-----------|
| Product recommender | Keyword stub in `recommender_stub.py` | Teammates' real service must match `get_recommendations(user_id: int, message: str) -> list[dict]` (dicts shaped like `RecommendedProduct`). Swap is import-level in `chatbot_service.py` — zero route/service changes. |
| Deal finder | Keyword stub in `deal_finder_stub.py` | Real service must match `get_deals(user_id: int, message: str) -> dict | None`. Swap is import-level. |
| RAG embeddings | Local, offline `all-MiniLM-L6-v2` | Hosted model = change `_MODEL_NAME` in `vector_store.py`, then rebuild via `scripts/build_vector_store.py`. |
| Auth | `X-User-Id` header stub | Real JWT swap in `get_current_user_id` (`dependencies.py` is the only seam). |
| Rate-limit store | In-memory (one process) | Redis behind the `RateLimitStore` ABC — one-line change in `check_rate_limit`. |

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

## Testing Strategy

| Test file | Coverage | Count |
|-----------|----------|-------|
| `tests/test_chatbot.py` | Full chat flow (mocked LLM), memory + user isolation, intent persistence, RAG, auth/rate-limit/error-handling, 404/405 shape, validation handler, hallucination guard | 23 |
| `tests/test_intent.py` | Regex fast-path (incl. plural forms), LLM fallback, malformed JSON, unknown intent, API failure, confidence override, regressions | 21 |
| `tests/test_retriever.py` | Relevance gate, threshold clamping, blank query, missing store, search failure | 6 |
| `tests/test_recommender_stub.py` | Real catalog matching, plural normalization, substring-trap avoidance, empty-catalog fallback | 4 |
| `tests/test_schemas.py` | ChatRequest validation edges, ChatResponse round-trip/defaults/range | 9 |
| **Total** | | **63** |

All LLM calls are mocked — tests run in ~30s with zero API cost. The in-memory DB tests use real SQLite with `StaticPool`; the recommender-stub tests exercise the real catalog matching code.

### What the suite does NOT catch

- Real Groq latency and the end-to-end <4s NFR (mocked LLM returns instantly) — needs load testing (locust/k6) against a real key.
- Real embedding quality — `test_retriever.py` mocks `search_faqs`; a live query sanity check is `python scripts/build_vector_store.py --query "..."`.
- DB connection pooling under concurrency (in-memory `StaticPool` hides it) — production test with PostgreSQL.
- Chroma concurrency (tests are single-threaded).
- The recommender/deal stubs vs teammates' real services (contract tests must be written when those land).
- The hallucination guard only validates **currency-tagged** figures — an invented discount *percentage* ("50% off") or quantity ("in stock in 3 colors") is not yet mechanically checked; it relies on the prompt guardrails. Extending the allowlist to percentages/quantities is natural follow-up work.
- The guard's allowlist includes the conversation history, so a figure hallucinated by a PRE-guard turn could be repeated and pass. Guarded turns never write such text, so history stays clean going forward.

## Architecture Decision Record

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| Groq over OpenAI | Fast inference, lower cost, fits the <4s NFR | Vendor lock-in (mitigated by thin wrapper) |
| Hand-rolled over LangChain | Full control of prompts/timeouts/fallbacks, no dependency bloat | More code to maintain (acceptable for this pipeline) |
| Custom rate limiter over slowapi | Zero new deps, testable, Redis-swap seam | Re-invents a solved problem (acceptable at this scope) |
| Local embeddings over hosted | Offline operation, zero per-query cost | Model quality ceiling (fine for FAQ retrieval) |
| SQLite over PostgreSQL (dev) | Zero setup, pytest-friendly | Not for multi-worker production |
| In-memory over Redis (dev) | Zero infrastructure | Limit becomes N×configured with N workers |
| `to_thread` + `wait_for` over raw SDK call | Bounds the blocking Groq call, protects NFR | One thread per concurrent call (acceptable given rate limits) |
| Regex fast-path over pure LLM intent | Zero cost + instant latency for common queries | Regex maintenance as intents evolve |
| Post-generation price guard over prompt-only guardrails | Instructions aren't enforcement — the guard mechanically blocks invented currency figures before they reach the customer or the DB | Covers only currency-tagged figures today; percentages/quantities remain prompt-governed |
| `/api/v1` prefix, `X-User-Id` header, snake_case | Versioned API, auth before LLM cost, Pydantic-native serialization | Deliberate deviations from the feature-spec examples (documented in README) |

## API Reference

### `POST /api/v1/ai/chat`

Versioned endpoint — note the `/api/v1` prefix (intentional deviation from the spec's `/api/ai/chat`, see ADR).

**Headers**

| Header | Required | Behavior |
|--------|----------|----------|
| `X-User-Id` | Yes | Integer user id. Missing → 401 `Authentication required`; non-integer → 400 `Validation error`. (Stub for JWT — see "What's Next".) |

**Request body** (`ChatRequest`, snake_case)

```json
{
  "message": "recommend me a perfume under 500 AED",
  "user_id": 42
}
```

- `message`: string, 1–2000 chars (required).
- `user_id`: optional int — currently informational; the `X-User-Id` header is the identity used for memory and rate limiting. Malformed bodies → 400 canonical body.

**200 response** (`ChatResponse`, snake_case)

```json
{
  "reply": "For a 500 AED gift, the Lacoste L.12.12 (299 AED) ...",
  "recommended_products": [
    {"id": 1, "name": "...", "price": 299.0, "currency": "AED",
     "category": "...", "brand": "...", "reason": "..."}
  ],
  "deal": "Summer Sale — 20% off (code: SUMMER20)",
  "intent": "recommend_product",
  "confidence": 0.95
}
```

- `recommended_products` / `deal` / `intent` / `confidence` are `null` when not applicable.
- The `reply` passes the hallucination guard: any currency figure not present in the injected context is replaced with the safety message.

**Rate limiting**

20 requests per minute per user (sliding window) → 21st request returns 429 with the canonical body. Budgets are per-user and independent.

**Errors** — every failure path returns the same shape:

```json
{"error": "<title>", "detail": "<human message>", "status_code": <int>}
```

See README's error table for the full matrix (401/400/429/500, and now 404/405).

## API Walkthroughs

### Authentication & rate limiting

```powershell
# Missing header → 401
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/chat" -Method POST `
  -Headers @{"Content-Type"="application/json"} `
  -Body '{"message":"no auth header"}'

# Invalid header → 400
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/chat" -Method POST `
  -Headers @{"X-User-Id"="abc"; "Content-Type"="application/json"} `
  -Body '{"message":"bad user id"}'

# Rate limit → 429 after 20 requests/min from the same X-User-Id
```

### Intent routing

```powershell
# Regex fast-path → intent: recommend_product, confidence: 1.0
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/chat" -Method POST `
  -Headers @{"X-User-Id"="1"; "Content-Type"="application/json"} `
  -Body '{"message":"Can you recommend a gift?"}'

# Regex fast-path → intent: track_order_help, confidence: 1.0
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/chat" -Method POST `
  -Headers @{"X-User-Id"="1"; "Content-Type"="application/json"} `
  -Body '{"message":"track my order"}'

# Regex fast-path → intent: deal_inquiry, confidence: 1.0 (plural forms too)
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/chat" -Method POST `
  -Headers @{"X-User-Id"="1"; "Content-Type"="application/json"} `
  -Body '{"message":"any discounts today?"}'

# LLM path (no regex keyword) → intent varies, confidence from classifier
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/chat" -Method POST `
  -Headers @{"X-User-Id"="1"; "Content-Type"="application/json"} `
  -Body '{"message":"what do you think I should get my dad"}'
```

### Recommendations, deals & RAG

```powershell
# recommend_product → recommended_products populated
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/chat" -Method POST `
  -Headers @{"X-User-Id"="1"; "Content-Type"="application/json"} `
  -Body '{"message":"recommend me a phone"}'

# deal_inquiry → deal populated ("Summer Sale — 20% off (code: SUMMER20)")
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/chat" -Method POST `
  -Headers @{"X-User-Id"="1"; "Content-Type"="application/json"} `
  -Body '{"message":"any discounts today?"}'

# product_faq → RAG-grounded answer
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/chat" -Method POST `
  -Headers @{"X-User-Id"="1"; "Content-Type"="application/json"} `
  -Body '{"message":"do you ship to Pakistan?"}'

# product_faq, no FAQ match → graceful no-match + human-support hand-off
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/chat" -Method POST `
  -Headers @{"X-User-Id"="1"; "Content-Type"="application/json"} `
  -Body '{"message":"warranty policy for a plumbus?"}'
```

### Timeout fallback

```powershell
# Set LLM_TIMEOUT_SECONDS=0.1 in .env, restart, then:
# → "I'm taking longer than usual. Please try again in a moment."
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/chat" -Method POST `
  -Headers @{"X-User-Id"="1"; "Content-Type"="application/json"} `
  -Body '{"message":"this will timeout"}'
```

## Non-Functional Requirements (NFR)

| NFR | Status | Mechanism |
|-----|--------|-----------|
| Chatbot response < 4 s | Guard in place; full validation pending | `LLM_TIMEOUT_SECONDS=3.0` + `asyncio.to_thread`/`wait_for` around the blocking Groq call; intent regex fast-path avoids LLM for common queries |
| Load testing | Pending | locust/k6 script required to prove p95 < 4s under concurrency |
| Streaming | Not implemented | Endpoint returns a single JSON body; SSE/WebSocket is future work |
| Performance telemetry | Stub present | `step_times` collected per request in `chatbot_service` (intent/catalog/route/llm) — needs a metrics export |

Budget math: intent (~ms, regex) + history load (~ms, SQLite) + catalog (~ms) + RAG retrieval (~50ms local) leaves ~3.5s of the 4s budget for the LLM; the 3.0s timeout keeps a ~0.5s cushion for response serialization and network. The hallucination guard is two regex scans over prompt-size strings — negligible on this budget.

## Gotchas & Troubleshooting

**WinError 206 (path too long) on Windows** — `sentence-transformers` installs `torch`, which extracts deeply nested files. Enable the registry Long Paths setting (`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1`) or relocate the project to a shorter path (e.g., `C:\dev\rrvdxb-chatbot`).

**Terminal mojibake on em-dashes** — the `deal` field contains an em-dash (—). Some Windows terminals render it as `â` due to console code-page detection. The JSON/API response contains the correct character; the artifact is display-only.

**Chroma telemetry noise** — suppressed via `Settings(anonymized_telemetry=False)` in `vector_store.py` and pinned `posthog==3.5.0`. If noise persists, check no other dependency pulled a newer `posthog`.

**Rate-limit quota leaking across tests** — the `InMemoryRateLimitStore` is a module-level singleton; an autouse pytest fixture resets it before every test (`rate_limit_store.reset()`) so quota never leaks.

**Hallucination guard tripping on valid replies** — the allowlist is everything the model saw (FAQ chunks, catalog, recommendations, deal, user message, history). If a legitimately grounded figure is ever replaced, the first thing to check is formatting: the pattern only recognizes `AED|USD|SAR|GBP` adjacent to the number — e.g. `US$4,699` or `4,699 AED inclusive` still matches, but a currency abbreviation outside those four does not. Extend `_PRICE_PATTERN` as new markets/currencies arrive.

**First-run model download** — `all-MiniLM-L6-v2` (~90MB) downloads from HuggingFace on first build. To pre-cache for an offline/CI environment:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

Then copy `~/.cache/torch/sentence_transformers/` into the deployment image.

**Missing `chroma.sqlite3` on a fresh clone** — the vector store is git-ignored; run `python scripts/build_vector_store.py` after cloning. The app won't crash without it (RAG degrades to `[]`), but FAQ answers won't be grounded until the index exists.

## What's Next

- [ ] Streaming response support (the endpoint returns a single JSON body today)
- [ ] Real JWT authentication — swap the `X-User-Id` stub for `Authorization: Bearer` when the Users/Auth API is ready (`dependencies.py` is the only seam)
- [ ] Redis-backed rate-limit store (shared counter) for multi-worker production
- [ ] Dockerize (multi-stage image; workers share Redis + PostgreSQL)
- [ ] Load testing for the <4s latency NFR (locust / k6)
- [ ] Performance budget telemetry (metrics on `step_times` in `chatbot_service`)
- [ ] Extend the hallucination guard beyond currency figures (discount percentages, quantities, delivery-day promises) and consider an alerts/observability hook (e.g., counter metric) on every guard trip

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | `sqlite:///./rrvdxb.db` for local dev |
| `GROQ_API_KEY` | Yes | From console.groq.com |
| `OPENAI_API_KEY` | No | Only used if `LLM_PROVIDER=openai` (from platform.openai.com) |
| `LLM_PROVIDER` | Yes | `groq` (default) |
| `LLM_MODEL` | Yes | `llama-3.1-8b-instant` (default) |
| `INTERNAL_JWT_SECRET` | Yes | Min 32 chars, service-to-service JWT signing |
| `DEBUG` | No | `True` or `False` (default: `True`) |
| `LLM_TIMEOUT_SECONDS` | No | `3.0` — hard cap (s) on each LLM call, protects the <4s NFR |

Query routing and RAG use local, offline embeddings — no additional API key required.
