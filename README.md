# RRVDXB AI Shopping Chatbot

AI-powered shopping assistant for RRVDXB premium e-commerce platform.
Serves customers in UAE, KSA, Pakistan, and UK with product discovery,
order support, and personalized recommendations.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | FastAPI (async Python) |
| ORM | SQLAlchemy 2.0 |
| Database | SQLite (local dev) → PostgreSQL (production) |
| LLM | Groq (Chat Completions API) |
| Validation | Pydantic v2 |
| Testing | pytest |

## Project Structure

```
rrvdxb-chatbot/
├── app/
│   ├── main.py              # FastAPI app factory + startup
│   ├── core/                # Config, DB engine, dependencies
│   │   ├── config.py        # Pydantic BaseSettings (.env loader)
│   │   ├── database.py      # SQLAlchemy engine, SessionLocal, Base
│   │   └── dependencies.py  # get_db, get_current_user_id (stub)
│   ├── models/              # SQLAlchemy ORM models
│   │   └── chat_history.py  # chat_history table
│   ├── schemas/             # Pydantic request/response schemas
│   │   └── chatbot_schema.py
│   ├── api/                 # API routers
│   │   └── v1/
│   │       ├── router.py    # Aggregates all v1 routes
│   │       └── endpoints/
│   │           └── chatbot.py  # POST /api/v1/ai/chat
│   ├── services/            # Business logic layer
│   │   └── chatbot_service.py  # LLM orchestration + intent routing + DB persistence
│   ├── ai/                  # LLM prompts, clients, memory, and intent
│   │   ├── __init__.py
│   │   └── chatbot/
│   │       ├── __init__.py
│   │       ├── prompts.py   # Guardrailed SYSTEM_PROMPT + INTENT_CLASSIFICATION_PROMPT
│   │       ├── llm_client.py  # Groq SDK wrapper (singleton client)
│   │       ├── memory.py    # DB-backed conversation memory
│   │       └── intent.py    # Day 4: regex + LLM intent classifier (IntentResult)
│   └── mock_data/           # Seed data for Day 1-2
│       ├── products.json
│       └── faqs.json
├── tests/
│   ├── test_chatbot.py      # pytest suite (mocked LLM + memory + intent persistence)
│   └── test_intent.py       # Day 4: intent recognition tests (regex + LLM fallback)
├── docs/
│   └── chatbot.md           # Sprint status tracker
├── requirements.txt
├── .env.example             # Environment variable template
├── .gitignore
└── README.md
```

## Quick Start

### 1. Clone & Enter Project

```bash
cd rrvdxb-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

Edit `.env` and set:

```
GROQ_API_KEY=gsk-... (get from console.groq.com)
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
```

### 5. Run the Server

```bash
uvicorn app.main:app --reload
```

### 6. Open API Docs

Navigate to: http://localhost:8000/docs

### 7. Test the Endpoint (PowerShell)

**Turn 1 — Start a conversation:**

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/chat" -Method POST -Headers @{"X-User-Id"="1"; "Content-Type"="application/json"} -Body '{"message":"I am looking for a gift"}'
```

**Turn 2 — Follow-up with context (same user):**

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/chat" -Method POST -Headers @{"X-User-Id"="1"; "Content-Type"="application/json"} -Body '{"message":"Something under 500 AED"}'
```

### 8. Test Intent Routing (Day 4)

Each response includes an `"intent"` (and `"confidence"`) field showing which path handled the request.

```powershell
# Regex fast-path → intent: recommend_product, confidence: 1.0
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/chat" -Method POST -Headers @{"X-User-Id"="1"; "Content-Type"="application/json"} -Body '{"message":"Can you recommend a gift?"}'

# Regex fast-path → intent: track_order_help, confidence: 1.0
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/chat" -Method POST -Headers @{"X-User-Id"="1"; "Content-Type"="application/json"} -Body '{"message":"track my order"}'

# Regex fast-path → intent: deal_inquiry, confidence: 1.0
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/chat" -Method POST -Headers @{"X-User-Id"="1"; "Content-Type"="application/json"} -Body '{"message":"any sale going on?"}'

# LLM path (no regex keyword) → intent: recommend_product, confidence: ~0.9
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/chat" -Method POST -Headers @{"X-User-Id"="1"; "Content-Type"="application/json"} -Body '{"message":"what do you think I should get my dad"}'
```

## Running Tests

```bash
pytest -q
```

Expected: `20 passed`

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| DATABASE_URL | Yes | sqlite:///./rrvdxb.db | SQLAlchemy DB URI |
| LLM_PROVIDER | Yes | groq | Currently only groq is wired |
| GROQ_API_KEY | Yes | — | Groq API key (from console.groq.com) |
| LLM_MODEL | Yes | llama-3.1-8b-instant | Model name for Groq |
| INTERNAL_JWT_SECRET | Yes | — | Min 32 chars for JWT signing |
| DEBUG | No | True | FastAPI debug mode |

> No new environment variables were added in Day 4.

## Progress

### Day 1: Project Scaffold
- FastAPI project scaffold with clean architecture
- Pydantic Settings with `.env` support
- SQLAlchemy + SQLite setup
- `chat_history` model, request/response schemas
- `POST /api/v1/ai/chat` placeholder endpoint
- Guardrailed system prompt
- Mock products (10 items) and FAQs
- Basic pytest suite

### Day 2: Groq LLM Integration
- Created `app/ai/chatbot/llm_client.py` — singleton Groq client with error handling
- Replaced placeholder echo in `chatbot_service.py` with real LLM calls
- Product catalog from `products.json` injected into system prompt as context
- Added graceful fallback when Groq API fails (rate limit, timeout, auth)
- Mocked LLM in tests using `unittest.mock.patch` — no real API calls in CI
- Updated all docs to reflect Groq (not OpenAI)

**New/updated files:** `llm_client.py`, `chatbot_service.py`, `test_chatbot.py`, `prompts.py`, `README.md`, `chatbot.md`

### Day 3: DB-Backed Conversation Memory
- Created `app/ai/chatbot/memory.py` — `save_turn`, `load_recent_history`, `format_history_for_prompt`
- DB-backed conversation memory with configurable window (N=5)
- Updated `chatbot_service.py` to load prior history before LLM call, save turn after response
- Updated `llm_client.py` to accept `history` parameter
- Added `test_conversation_memory_persists_across_turns` with SQLite in-memory + `StaticPool`
- Anonymous user fallback (`user_id` → 0 when `None`)
- No LangChain added

### Day 4: Intent Recognition + Routing
- Created `app/ai/chatbot/intent.py` — hybrid intent classifier
  - `IntentResult` Pydantic model with `intent`, `confidence`, `entities`
  - Step 1 regex fast-path (`recommend_product`, `track_order_help`, `deal_inquiry`, `product_faq`) — zero LLM cost, `confidence=1.0`
  - Step 2 LLM fallback using the existing Groq client (`send_chat_message`) — no second API client
  - Step 3 hardened JSON parsing: strips markdown code fences, `json.loads` in try/except, enum allow-list, confidence clamped to `[0, 1]`
  - Confidence threshold `0.7` — low-confidence labels override to `general_chat` (original confidence kept)
  - Any API/parse failure degrades to `general_chat` with `confidence=0.0`
- Added `INTENT_CLASSIFICATION_PROMPT` in `prompts.py` (strict JSON output contract + few-shot examples)
- Updated `chatbot_service.py`:
  - `classify_intent()` runs as the FIRST step in `handle_chat_message()`
  - `_build_system_prompt_for_intent()` routes to per-intent code paths
    - `recommend_product` → product catalog context (existing behavior)
    - `product_faq` → `# TODO: Day 6 — RAG pipeline`
    - `deal_inquiry` → `# TODO: Day 7 — Deal Finder integration`
    - `track_order_help` → order-tracking guidance injected
    - `general_chat` → standard flow
  - Classified intent saved to `chat_history.intent` on every turn
- Updated `chatbot_schema.py` — `ChatResponse` now exposes `intent` and `confidence`
- Created `tests/test_intent.py` (regex fast-path without LLM, LLM fallback, malformed JSON, unknown intent, Groq API failure, confidence override)
- Updated `tests/test_chatbot.py` — mocked classifier in chat-flow tests, intent persisted to DB asserted
- Manually verified all 5 intents route correctly via the API (regex fast-path → `confidence=1.0`; LLM path → `confidence=0.92`)
- Post-review hardening (applied):
  - All regex patterns use `\b` word boundaries (so `suggestion` no longer matches `suggest`)
  - `gift` only matches in shopping phrases (`gift for`, `gift idea(s)`) — a "return policy for a gift" query correctly routes to `product_faq`
  - `\bbuy\b` added so bare buying questions route to `recommend_product`
  - Confidence output is clamped to `[0.0, 1.0]` (LLM returning `9.5` becomes `1.0`)
  - `INTENT_CLASSIFICATION_PROMPT` now instructs: "If the message fits none of these intents, choose `general_chat` with confidence ≤ 0.5"
  - `CURRENT PRODUCT CATALOG:` header is only injected when `product_context` is non-empty
  - Regression tests added (`test_gift_return_query_routes_to_product_faq_not_recommend`, `test_bare_buy_matches_recommend`, `test_confidence_is_clamped_to_1_0`, extended `test_regex_fast_path_never_calls_llm`)
- No new packages added (stdlib `re`/`json` only) — **no LangChain**

**New files:** `app/ai/chatbot/intent.py`, `tests/test_intent.py`
**Updated files:** `prompts.py`, `chatbot_service.py`, `chatbot_schema.py`, `test_chatbot.py`, `README.md`, `chatbot.md`

## Maintainer

Ameema Rashid — AI Lead, RRVDXB Chatbot Sprint  
TechNexus Virtual University
