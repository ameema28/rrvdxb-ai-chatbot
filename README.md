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
| LLM | Groq / OpenAI (Chat Completions API) |
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
│   │           └── chatbot.py  # POST /api/ai/chat
│   ├── services/            # Business logic layer
│   │   └── chatbot_service.py
│   ├── ai/                  # LLM prompts and clients
│   │   └── chatbot/
│   │       └── prompts.py   # Guardrailed SYSTEM_PROMPT
│   └── mock_data/           # Seed data for Day 1 placeholder
│       ├── products.json
│       └── faqs.json
├── tests/
│   └── test_chatbot.py      # pytest suite
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
LLM_MODEL=llama3-8b-8192
```

### 5. Run the Server

```bash
uvicorn app.main:app --reload
```

### 6. Open API Docs

Navigate to: http://localhost:8000/docs

### 7. Test the Endpoint

```bash
curl -X POST "http://localhost:8000/api/v1/ai/chat" \
  -H "X-User-Id: 1" \
  -H "Content-Type: application/json" \
  -d '{"message":"Do you have iPhones?"}'
```

## Running Tests

```bash
pytest -q
```

Expected: `2 passed`

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| DATABASE_URL | Yes | sqlite:///./rrvdxb.db | SQLAlchemy DB URI |
| LLM_PROVIDER | Yes | groq | openai or groq |
| OPENAI_API_KEY | If provider=openai | — | OpenAI API key |
| GROQ_API_KEY | If provider=groq | — | Groq API key |
| LLM_MODEL | Yes | llama3-8b-8192 | Model name for chosen provider |
| INTERNAL_JWT_SECRET | Yes | — | Min 32 chars for JWT signing |
| DEBUG | No | True | FastAPI debug mode |

## Maintainer

Ameema Rashid — AI Lead, RRVDXB Chatbot Sprint  
TechNexus Virtual University
