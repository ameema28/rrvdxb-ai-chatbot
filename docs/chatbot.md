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

## What's Next (Day 4-10)
- [ ] Add intent classification
- [ ] Wire vector DB / RAG for product search
- [ ] Add streaming response support
- [ ] JWT authentication (replace X-User-Id stub)
- [ ] Add rate limiting per user
- [ ] Dockerize the application
- [ ] Load testing for &lt;4s latency NFR

## Environment Variables Needed

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | `sqlite:///./rrvdxb.db` for local dev |
| `GROQ_API_KEY` | Yes | From console.groq.com |
| `LLM_PROVIDER` | Yes | `groq` (default) |
| `LLM_MODEL` | Yes | `llama-3.1-8b-instant` (default) |
| `INTERNAL_JWT_SECRET` | Yes | Min 32 chars, used for service-to-service tokens |
| `DEBUG` | No | `True` or `False` (default: True) |