# RRVDXB AI Shopping Chatbot — Sprint Status

## What's Done (Day 1)
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

## What's Next (Day 2-10)
- [ ] Integrate OpenAI Chat Completions API
- [ ] Implement conversation history retrieval
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
| `OPENAI_API_KEY` | Yes | From OpenAI dashboard |
| `INTERNAL_JWT_SECRET` | Yes | Min 32 chars, used for service-to-service tokens |
| `DEBUG` | No | `True` or `False` (default: True) |