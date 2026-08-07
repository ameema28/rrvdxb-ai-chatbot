"""
Vector store for RAG — FAQ retrieval via local embeddings (Day 5).

This module owns EVERYTHING that talks to ChromaDB and the embedding model,
so the service layer (and future Day-6 code) never touches them directly.

Key ideas:
  1. We embed text with sentence-transformers (all-MiniLM-L6-v2) — local,
     free, no OpenAI key.
  2. Chunks = one Q&A pair each (our FAQs are short; splitting would hurt).
  3. ChromaDB persists to disk so build once, load many.
  4. All build/load/query is idempotent and error-tolerant.

Author: Ameema Rashid — RRVDXB AI Sprint, Day 5.

"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (derived, never hard-coded strings)
# ---------------------------------------------------------------------------
# vector_store.py lives at: app/ai/chatbot/rag/vector_store.py
#   parents[0] = rag, [1] = chatbot, [2] = ai, [3] = app
_APP_DIR: Path = Path(__file__).resolve().parents[3]
_FAQS_PATH: Path = _APP_DIR / "mock_data" / "faqs.json"
_PERSIST_DIR: Path = Path(__file__).resolve().parent / "chroma_db"

_COLLECTION_NAME = "faqs"
_MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Module-level caches: load each expensive resource at most once per process
# ---------------------------------------------------------------------------
_embedding_model: Optional[SentenceTransformer] = None
_collection: Optional[Collection] = None


# ===========================================================================
# 1. Load FAQs safely
# ===========================================================================
def load_faqs() -> List[Dict[str, Any]]:
    """
    Read app/mock_data/faqs.json and return a normalized list of FAQ entries.

    Tolerances (production-grade, never raise on file problems):
      - file missing                         -> log + return []
      - malformed JSON / not a list          -> log + return []
      - entry without 'question'/'answer'    -> skipped
      - missing 'id'                         -> auto-assigned (1-based)

    Returns:
        List[dict] each shaped {"id": int, "question": str, "answer": str}.
    """
    if not _FAQS_PATH.is_file():
        logger.warning("FAQs file not found: %s — returning empty list", _FAQS_PATH)
        return []

    try:
        with _FAQS_PATH.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read/parse FAQs: %s", exc)
        return []

    if not isinstance(raw, list):
        logger.error("FAQs root is not a JSON list — returning empty list")
        return []

    faqs: List[Dict[str, Any]] = []
    for index, entry in enumerate(raw, start=1):
        if not isinstance(entry, dict):
            continue
        question = entry.get("question")
        answer = entry.get("answer")
        # A valid chunk needs a non-empty question AND answer, both strings.
        if not isinstance(question, str) or not question.strip():
            continue
        if not isinstance(answer, str) or not answer.strip():
            continue
        faqs.append(
            {
                "id": int(entry.get("id", index)),  # stable fallback id
                "question": question.strip(),
                "answer": answer.strip(),
            }
        )
    return faqs


# ===========================================================================
# 2. Embedding model (cached singleton)
# ===========================================================================
def _get_embedding_model() -> SentenceTransformer:
    """
    Return the shared SentenceTransformer, loading it at most once.

    Fast path: the model is already cached from a previous run, so load it
    with local_files_only=True — this SKIPS the HuggingFace HEAD check that
    otherwise hits huggingface.co on every start (and times out on slow or
    blocked networks).

    Fallback: if the model isn't cached (a real first run), load normally,
    which downloads all-MiniLM-L6-v2 (~90MB) once and caches it.

    Raises:
        RuntimeError: if the model cannot be loaded even online.
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    try:
        _embedding_model = SentenceTransformer(_MODEL_NAME, local_files_only=True)
    except Exception:
        logger.info(
            "Model not in local cache — downloading '%s' from Hugging Face",
            _MODEL_NAME,
        )
        try:
            _embedding_model = SentenceTransformer(_MODEL_NAME)
        except Exception as exc:
            raise RuntimeError(
                "Embedding model could not be loaded. Check your internet "
                f"connection and try again (model '{_MODEL_NAME}' "
                "downloads once to ~/.cache/huggingface)."
            ) from exc
    return _embedding_model


# ===========================================================================
# 3. Chroma collection (always ours; we pass embeddings explicitly)
# ===========================================================================
def _get_collection() -> Collection:
    """
    Return (creating if needed) the persistent collection.

    embedding_function=None tells Chroma NOT to load its own embedding model;
    we always pass our own embeddings on add/upsert/query.
    anonymized_telemetry=False stops Chroma calling posthog (removes the noisy
    "Failed to send telemetry event" ERROR logs proven above and in production).
    """
    client = chromadb.PersistentClient(
        path=str(_PERSIST_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},  # distance metric = cosine
    )

# ===========================================================================
# 4. Build the index (chunk -> embed -> persist). Idempotent.
# ===========================================================================
def build_vector_store() -> int:
    """
    Build the FAQ vector index from faqs.json. Safe to re-run at any time.

    Idempotency strategy: stable chunk ids (faq-<id>) + upsert + stale-delete
    so a re-run UPDATES changed entries, ADDS new ones, and REMOVES deleted
    ones — it never crashes and never duplicates.

    Returns:
        Number of chunks embedded & stored (0 if there is nothing to index).
    """
    faqs: List[Dict[str, Any]] = load_faqs()
    if not faqs:
        logger.warning("build_vector_store: no FAQs to embed — nothing built")
        return 0

    model = _get_embedding_model()

    # One chunk per Q&A pair (short docs; splitting would destroy meaning).
    chunk_ids = [f"faq-{faq['id']}" for faq in faqs]
    chunk_text = [f"{faq['question']}\n{faq['answer']}" for faq in faqs]
    chunk_metas = [
        {"id": faq["id"], "question": faq["question"], "answer": faq["answer"]}
        for faq in faqs
    ]

    logger.info("Embedding %d chunk(s) with %s ...", len(chunk_text), _MODEL_NAME)
    # normalize_embeddings=True: unit vectors => cosine = dot product.
    embeddings = model.encode(chunk_text, normalize_embeddings=True)

    collection = _get_collection()

    # Re-sync: delete entries whose id is no longer in faqs.json.
    existing_ids = set(collection.get()["ids"])
    stale = [cid for cid in existing_ids if cid not in set(chunk_ids)]
    if stale:
        collection.delete(ids=stale)
        logger.info("Removed %d stale chunk(s)", len(stale))

    # upsert = insert-or-update in one atomic call (idempotent).
    collection.upsert(
        ids=chunk_ids,
        documents=chunk_text,
        metadatas=chunk_metas,
        embeddings=embeddings.tolist(),
    )

    final_count = collection.count()
    logger.info(
        "Vector store ready: %d chunk(s) in collection '%s'.",
        final_count,
        _COLLECTION_NAME,
    )
    return final_count


# ===========================================================================
# 5. Lazy loader — build only if the index does not already exist on disk
# ===========================================================================
def get_vector_store() -> Collection:
    """
    Return the persisted collection, building it on the first call only.

    We look for the persistent Chroma sqlite file on disk; if absent we build
    first (this triggers the one-time model download). The result is cached so
    subsequent calls are instant and never re-embed.

    Raises:
        RuntimeError: if an embedding model download is required and fails.
    """
    global _collection
    if _collection is not None:
        return _collection

    sqlite_exists = (_PERSIST_DIR / "chroma.sqlite3").exists()
    if not sqlite_exists:
        logger.info("No persisted index found — building from faqs.json")
        build_vector_store()

    _collection = _get_collection()
    logger.info("Using persisted index at %s", _PERSIST_DIR)
    return _collection


# ===========================================================================
# 6. Semantic search
# ===========================================================================
def search_faqs(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Semantic search over FAQ chunks; returns the top_k closest matches.

    Args:
        query: natural-language user question (e.g. "do you ship to Pakistan?").
        top_k: number of results to return.

    Returns:
        List[dict] with keys: id, question, answer, similarity (0..1).
        Empty list if the store is empty.
    """
    collection = get_vector_store()
    if collection.count() == 0:
        logger.info("search_faqs: store is empty — no results")
        return []

    model = _get_embedding_model()
    q_emb = model.encode([query], normalize_embeddings=True)

    res = collection.query(
        query_embeddings=q_emb.tolist(),
        n_results=max(1, min(top_k, collection.count())),
        include=["documents", "metadatas", "distances"],
    )

    docs = res.get("documents", [[]])[0] or []
    metas = res.get("metadatas", [[]])[0] or []
    distances = res.get("distances", [[]])[0] or []

    out: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, distances):
        # Chroma cosine distance = 1 - cosine_similarity, so similarity is 1-d.
        out.append(
            {
                "id": meta.get("id"),
                "question": meta.get("question", ""),
                "answer": meta.get("answer", ""),
                "similarity": round(1.0 - dist, 4),
            }
        )
    return out