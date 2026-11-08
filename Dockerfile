# RRVDXB AI Shopping Chatbot: container image for Google Cloud Run (and local Docker).
#
# - CPU-only torch, installed in its own layer from the PyTorch CPU index.
# - Runtime dependencies only: requirements-dev.txt is never installed.
# - Runs as a non-root user (uid 1000) and listens on $PORT (default 8080).
# - The embedding model and the FAQ Chroma index are built into the image, and
#   Hugging Face is offline at runtime (no download at start-up).
# - INTERNAL_JWT_SECRET is NOT in the image: the app refuses to start without
#   one (at least 32 characters). On Cloud Run it comes from Secret Manager.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

WORKDIR /app

# Layer 1: CPU-only torch from the PyTorch CPU index only. The torch==2.15.0
# pin in requirements.txt is then already satisfied, so pip keeps this build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0

# Layer 2: the other runtime dependencies.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Non-root runtime user. /home/user/data is the mount point for the
# docker-compose volume (SQLite chat history in local Docker).
RUN useradd --create-home --uid 1000 user \
 && mkdir -p /home/user/data \
 && chown user:user /home/user/data

USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface

# Layer 3: the embedding model, downloaded before any app code is copied, so
# code changes do not repeat the download.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

USER root

# Application code is owned by root, so the runtime user can only read it.
COPY --chown=root:root app ./app
COPY --chown=root:root scripts/build_vector_store.py ./scripts/build_vector_store.py

# The Chroma index directory is the one path under /app the runtime user owns:
# Chroma opens its SQLite file read-write.
RUN mkdir -p /app/app/ai/chatbot/rag/chroma_db \
 && chown user:user /app/app/ai/chatbot/rag/chroma_db

USER user

# Build the FAQ index (12 chunks) offline, from the cached model.
RUN python scripts/build_vector_store.py

# Runtime defaults. Without DATABASE_URL, chat history is SQLite in /tmp, which
# is reset whenever the container restarts. Demo tokens stay off unless enabled.
ENV PORT=8080 \
    APP_ENV=production \
    DEBUG=false \
    DATABASE_URL=sqlite:////tmp/rrvdxb.db \
    DB_AUTO_MIGRATE=true \
    RAG_WARMUP=true \
    DEMO_TOKENS_ENABLED=false

EXPOSE 8080

# One worker: the rate limiter is in memory, per process. --no-proxy-headers
# keeps request.client as the socket peer; TRUST_PROXY decides whether
# X-Forwarded-For is used (app/core/client_ip.py).
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --no-proxy-headers"]
