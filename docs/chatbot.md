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
- [x] Mock products (10 items) and FAQs (8 items)
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
  - [x] `product_faq` → `TODO: Day 6 — RAG pipeline`
  - [x] `deal_inquiry` → `TODO: Day 7 — Deal Finder integration`
  - [x] `track_order_help` → order-tracking guidance
  - [x] `general_chat` → standard flow
- [x] Intent persisted to `chat_history.intent` on every turn
- [x] `ChatResponse` exposes the classified `intent`
- [x] `tests/test_intent.py` created (regex no-LLM, LLM fallback, malformed JSON, unknown intent, API failure, confidence override)
- [x] `tests/test_chatbot.py` updated to mock the classifier + assert intent persistence
- [x] `pytest -q` → 17 passed
- [x] No new packages; no LangChain added
- [x] Post-review hardening:
  - [x] `\b` word-boundaries on all regex patterns (`suggestion` ≠ `suggest`)
  - [x] `gift` restricted to shopping phrases — return-policy-on-gift routes to `product_faq`
  - [x] `\bbuy\b` added → bare buying questions route to `recommend_product`
  - [x] Confidence clamped to `[0.0, 1.0]` (LLM `9.5` → `1.0`)
  - [x] Prompt now says: "If fits none, choose `general_chat` with confidence ≤ 0.5"
  - [x] Catalog header only injected when `product_context` is non-empty
  - [x] New regression tests: gift-return→`product_faq`, bare-`buy`→`recommend_product`, confidence clamp
  
## What's Next (Day 5-10)
- [ ] Wire vector DB / RAG for product search (product_faq path, Day 6)
- [ ] Deal Finder integration (deal_inquiry path, Day 7)
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

> No new environment variables introduced on Day 4.