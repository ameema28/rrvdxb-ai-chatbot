# RRVDXB AI Shopping Chatbot — container image (Day 10)
#
# Single-stage build, tuned for Render or any PORT-based PaaS:
#   - python:3.11-slim matches the local dev version AND the CI runner, so
#     behavior is identical everywhere.
#   - Every dependency ships manylinux wheels, so no compiler toolchain is
#     needed. If a future dependency lacks a wheel, add:
#       RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc g++
#   - The FAQ vector store is baked INTO the image at `docker build` time:
#       * the ~90MB all-MiniLM-L6-v2 model downloads ONCE during the build and
#         is baked in -> zero download when the container starts;
#       * chroma_db/ is baked in, so RAG answers are grounded from second zero.
#   - Runtime reads $PORT (Render convention) with a 8000 fallback so plain
#     `docker run` needs zero configuration.

# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Dependencies first: this layer is cached unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Application code + the vector-store build script.
COPY app ./app
COPY scripts ./scripts

# Build the FAQ vector store into the image (idempotent, see scripts/).
RUN python scripts/build_vector_store.py

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]