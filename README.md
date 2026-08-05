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
│   │   └── chatbot_service.py  # LLM orchestration + DB persistence
│   ├── ai/                  # LLM prompts, clients, and memory
│   │   ├── __init__.py
│   │   └── chatbot/
│   │       ├── __init__.py
│   │       ├── prompts.py   # Guardrailed SYSTEM_PROMPT
│   │       ├── llm_client.py  # Groq SDK wrapper (singleton client)
│   │       └── memory.py    # DB-backed conversation memory
│   └── mock_data/           # Seed data for Day 1-2
│       ├── products.json
│       └── faqs.json
├── tests/
│   └── test_chatbot.py      # pytest suite (mocked LLM + memory tests)
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

## Running Tests

```bash
pytest -q
```

Expected: `4 passed`

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| DATABASE_URL | Yes | sqlite:///./rrvdxb.db | SQLAlchemy DB URI |
| LLM_PROVIDER | Yes | groq | Currently only groq is wired |
| GROQ_API_KEY | Yes | — | Groq API key (from console.groq.com) |
| LLM_MODEL | Yes | llama-3.1-8b-instant | Model name for Groq |
| INTERNAL_JWT_SECRET | Yes | — | Min 32 chars for JWT signing |
| DEBUG | No | True | FastAPI debug mode |

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

## Maintainer

Ameema Rashid — AI Lead, RRVDXB Chatbot Sprint  
TechNexus Virtual University
