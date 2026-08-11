# RRVDXB AI Shopping Chatbot — Sprint Status

## What's Done

### Day 1
- [x] FastAPI project scaffold with clean architecture
- [x] Pydantic Settings with `.env` support
- [x] SQLAlchemy + SQLite setup
- [x] `chat_history` model matching required schema
- [x] Request/response schemas with validation
- [x] `POST /api/ai/chat` placeholder endpoint
- [x] Service layer stub with DB persistence
- [x] Guardrailed system prompt
- [x] Mock products (10 items) and FAQs
- [x] Basic pytest suite
- [x] `.env.example` and `README.md`

### Day 2
- [x] Integrate Groq LLM (direct SDK, no LangChain)
- [x] Singleton Groq client with connection reuse
- [x] Product context injection into system prompt
- [x] Graceful error handling for API failures
- [x] Mocked LLM in pytest — fast, free, deterministic tests
- [x] Updated docs to reflect Groq (not OpenAI)

### Day 3
- [x] DB-backed conversation memory (`app/ai/chatbot/memory.py`)
- [x] `save_turn`, `load_recent_history`, `format_history_for_prompt`
- [x] Context windowing: load only last N=5 turns (not full history)
- [x] History injected into LLM prompt before each call
- [x] New turn persisted to `chat_history` after each response
- [x] Memory test: two-turn conversation, real SQLite in-memory DB
- [x] Assert second call's prompt contains first turn context
- [x] No LangChain added — implemented manually with SQLAlchemy

### Day 4
- [x] Intent classification as a separate pipeline step (`app/ai/chatbot/intent.py`)
- [x] Hybrid approach: regex fast-path + LLM fallback
- [x] Regex fast-path returns `confidence=1.0` with zero LLM calls
- [x] LLM classification reuses the existing Groq client (`send_chat_message`)
- [x] `INTENT_CLASSIFICATION_PROMPT` with strict JSON output contract + few-shot examples
- [x] Defensive JSON parsing (strip code fences, try/except, enum allow-list, clamp confidence)
- [x] Confidence threshold `0.7` → low-confidence intents override to `general_chat` (confidence kept)
- [x] Any API/parse failure degrades to `general_chat` with `confidence=0.0`
- [x] Routing in `chatbot_service.py` via `_build_system_prompt_for_intent()`
  - [x] `recommend_product` → product catalog context
  - [x] `product_faq` → RAG pipeline (retrieve → ground → generate) — resolved in **Day 6**
  - [x] `deal_inquiry` → `TODO: Day 7 — Deal Finder integration` (resolved in **Day 7**)
  - [x] `track_order_help` → order-tracking guidance
  - [x] `general_chat` → standard flow
- [x] Intent persisted to `chat_history.intent` on every turn
- [x] `ChatResponse` exposes the classified `intent`
- [x] `tests/test_intent.py` created (regex no-LLM, LLM fallback, malformed JSON, unknown intent, API failure, confidence override)
- [x] `tests/test_chatbot.py` updated to mock the classifier + assert intent persistence
- [x] `pytest -q` → 20 passed
- [x] No new packages; no LangChain added
- [x] Post-review hardening:
  - [x] `\b` word-boundaries on all regex patterns (`suggestion` ≠ `suggest`)
  - [x] `gift` restricted to shopping phrases — return-policy-on-gift routes to `product_faq`
  - [x] `\bbuy\b` added → bare buying questions route to `recommend`
  - [x] Confidence clamped to `[0.0, 1.0]` (LLM `9.5` → `1.0`)
  - [x] Prompt now says: "If fits none, choose `general_chat` with confidence ≤ 0.5"
  - [x] Catalog header only injected when `product_context` is non-empty
  - [x] New regression tests: gift-return→`product_faq`, bare-`buy`→`recommend`, confidence clamp

### Day 5
- [x] Understand RAG fundamentals (embeddings, cosine similarity, chunking, Vector DB trade-offs, RAG vs fine-tuning vs stuffing)
- [x] Create `app/ai/chatbot/rag/__init__.py`
- [x] Create `app/ai/chatbot/rag/vector_store.py`
  - [x] `load_faqs()` — safe reads of `faqs.json` (missing file / malformed JSON / invalid entries handled)
  - [x] `build_vector_store()` — chunk (1 Q&A per chunk) → embed (`all-MiniLM-L6-v2`) → persist to ChromaDB
  - [x] Idempotent build: upsert + stale-delete; re-running never crashes or duplicates
  - [x] `get_vector_store()` — lazy load; build only if `chroma.sqlite3` not already persisted
  - [x] Graceful first-run model download + offline-cache fast path (`local_files_only`)
- [x] Embed **locally** — **no OpenAI key anywhere**
- [x] Persist to `app/ai/chatbot/rag/chroma_db/` (git-ignored)
- [x] Expand `faqs.json` (Shipping UAE/KSA/Pak/UK, Returns, Payments, Free-shipping AED 489, Warranty, Tracking, Authenticity, Discounts)
- [x] Create standalone `scripts/build_vector_store.py` (no FastAPI needed; optional `--query` demo)
- [x] Update `requirements.txt` (`chromadb==0.5.5`, `sentence-transformers==3.0.1`, `httpx==0.27.2`, `posthog==3.5.0`) and `.gitignore`
- [x] Suppress Chroma telemetry noise (`Settings(anonymized_telemetry=False)` + `posthog` pin)
- [x] Verified via `python scripts/build_vector_store.py --query "do you ship to Pakistan?"` → Pakistan FAQ top hit (~0.79)
- [x] `pytest -q` → 20 passed (RAG not yet wired to chat, so existing tests unaffected)

> **Day 5 notes:** No new environment variables were added. The Chroma collection
> path and embedding model name are fixed code constants. On Windows, `pip install`
> of `sentence-transformers` (→ `torch`) may hit `WinError 206` (path too long) —
> fix by enabling the registry **Long Paths** setting or relocating the project.
> RAG was standalone through Day 5; **Day 6** wired it into the `product_faq` response.

### Day 6
- [x] Understand how the retriever plugs into the flow (retrieve → augment → generate)
- [x] Understand chunk-size + top-k cost/latency trade-offs and similarity thresholds
- [x] Create `app/ai/chatbot/rag/retriever.py` — `retrieve_faq_context()`
  - [x] Reuses `search_faqs()` / `get_vector_store()`; never rebuilds the index
  - [x] Relevance gate on cosine similarity; clamps `k` (1–10) and threshold (0–1)
  - [x] Graceful `[]` on blank query / missing·empty store / search failure
  - [x] Returns `[{"question", "answer", "similarity"}]`, most-similar first
- [x] Add `build_rag_system_prompt()` in `prompts.py` — Day-1 guardrails verbatim +
      Q:/A: `FAQ CONTEXT:` block + cite-only / never-invent instructions
- [x] Extend `send_chat_message()` with optional `system_prompt_override`
      (thin Groq wrapper stays thin; product-catalog injection skipped; history still appended)
- [x] Wire `product_faq` path in `chatbot_service.py`: retrieve → ground → generate;
      no-match degrades to the general flow with polite `_NO_FAQ_MATCH_NOTE` and catalog
      re-injected as fallback context
- [x] Extend `intent.py` `product_faq` regex (shipping destination / delivery options /
      stock questions); `track_order_help` precedence preserved
- [x] Tests: `test_retriever.py` (6), RAG chat-flow tests (2) + no-match fallback assert,
      intent routing regressions (4)
- [x] `pytest -q` → **32 passed**
- [x] No new packages; **no LangChain** (requirements.txt unchanged via Day 6)
- [x] Manual verifications:
  - [x] "do you ship to Pakistan?" → `product_faq`, grounded in the Pakistan FAQ chunk
  - [x] "what is your return policy?" → grounded from the FAQ (14-day policy)
  - [x] "warranty policy for a plumbus?" → graceful no-match + human-support hand-off

> **Day 6 notes:** No new environment variables — the similarity threshold is a code
> constant (`DEFAULT_SIMILARITY_THRESHOLD = 0.6`, chosen because the exact "return policy"
> FAQ match measures ~0.66 and the spec-default 0.7 would drop it). No new packages (still
> NO LangChain). `product_faq` questions are now answered from the retrieved FAQ index;
> the Day-1 persona guardrails are preserved verbatim around the injected context.

### Day 7
- [x] Understand the <4s NFR risk: a blocking, sequential LLM call (groq SDK) on the async
      loop with no deadline can consume the whole budget — the fix is `to_thread` + `wait_for`
- [x] Create `app/services/recommender_stub.py` — `get_recommendations(user_id, message)`
  - [x] Safe catalog loader with candidate-path search (`<root>/mock_data` → `app/mock_data` → cwd);
        never raises; logs + returns `[]` on any failure
  - [x] Keyword matching via word prefix/suffix equality on name/brand/category —
        `"phone"` → `iPhone` but NOT `headphones`; plural-normalized tokens (`phones` → `phone`)
  - [x] Top-3 results with `{id, name, price, currency, category, brand, reason}` — drop-in
        stand-in for the recommendations team's real service
- [x] Create `app/services/deal_finder_stub.py` — `get_deals(user_id, message)`
  - [x] Keyword-gated mock deal (`Summer Sale`, 20% off, `SUMMER20`) on
        discount/sale/deal/offer/coupon/promo; `None` otherwise (never invents an offer)
- [x] Add `llm_timeout_seconds: float = 3.0` to `config.py` + `LLM_TIMEOUT_SECONDS` in `.env.example`
      — hard cap protecting the <4s NFR (~1s headroom after pre-LLM steps)
- [x] Rewire `chatbot_service.py` into per-intent blocks:
  - [x] `recommend_product` → stub → `RecommendedProduct` list + `RECOMMENDED PRODUCTS:` context
        (grounded in the picked items); no matches → catalog fallback
  - [x] `deal_inquiry` → stub → `ACTIVE OFFER:` context + human-readable `deal` string;
        no deal → general flow (no fabrication)
  - [x] `track_order_help` / `general_chat` → `_build_system_prompt_for_intent` (kept)
  - [x] Timeout via `await asyncio.wait_for(asyncio.to_thread(functools.partial(send_chat_message, ...)),
        timeout=settings.llm_timeout_seconds)`
  - [x] Distinct fallbacks: `asyncio.TimeoutError` → `_TIMEOUT_REPLY` (slow, not broken);
        `RuntimeError` → existing "human support" message
  - [x] Fallback turns persist as `general_chat` (`_FALLBACK_PERSIST_INTENT`); original intent
        still returned in `ChatResponse`
  - [x] API-only confidence floor: sub-`0.7` surfaces as 0.9 in the response; intent layer
        keeps the raw LLM confidence (test_intent.py contract preserved)
- [x] Tests: `test_chatbot.py` +4 Day-7 tests + 1 API confidence-floor test;
      `tests/test_recommender_stub.py` (4, against the REAL stub)
- [x] `pytest -q` → **41 passed**
- [x] No new packages; **no LangChain** (requirements.txt unchanged); `app/services` already
      existed → no new `__init__.py`
- [x] Manual verifications:
  - [x] "recommend me a phone" → `recommended_products` populated (iPhone 14 Pro Max;
        Sony-headphones false positive fixed)
  - [x] "any discounts today?" → `deal` = "Summer Sale — 20% off (code: SUMMER20)" (previous
        `â` was terminal mojibake; JSON/UI shows the proper em-dash)
  - [x] Empty `GROQ_API_KEY` restart → graceful HTTP 200 (regex/RAG intents + canned persona
        answer without the LLM)
  - [x] `LLM_TIMEOUT_SECONDS=0.1` → timeout reply within budget

> **Day 7 notes:** Exactly one new env var — `LLM_TIMEOUT_SECONDS` (default `3.0`, keeping
> ~1s headroom under the 4s NFR). Timeout vs error are deliberately distinct user-facing
> messages. The recommender/deal services are STUBs standing in for the teammate teams'
> services; the chatbot depends only on their function signatures, so swapping in the real
> services is a drop-in replacement.

## What's Next (Day 8-10)
- [ ] Add streaming response support
- [ ] JWT authentication (replace X-User-Id stub)
- [ ] Add rate limiting per user
- [ ] Dockerize the application
- [ ] Load testing for <4s latency NFR

## Environment Variables Needed

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | `sqlite:///./rrvdxb.db` for local dev |
| `GROQ_API_KEY` | Yes | From console.groq.com |
| `LLM_PROVIDER` | Yes | `groq` (default) |
| `LLM_MODEL` | Yes | `llama-3.1-8b-instant` (default) |
| `INTERNAL_JWT_SECRET` | Yes | Min 32 chars, used for service-to-service tokens |
| `DEBUG` | No | `True` or `False` (default: True) |
| `LLM_TIMEOUT_SECONDS` | No | 3.0 — hard cap (s) on each LLM call, protects the <4s NFR |

> Query routing and RAG use local, offline embeddings — no additional API key
> required. Day 7 added exactly one env var: `LLM_TIMEOUT_SECONDS`.