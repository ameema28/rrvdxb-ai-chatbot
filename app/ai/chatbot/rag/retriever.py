"""
RAG retriever — the retrieval + relevance-gate layer of the RAG pipeline (Day 6).

Lives between the chat service and the vector store. Its single job: answer
"given this customer question, which FAQ facts are trustworthy enough to ground
the answer?" It applies a similarity threshold on top of the Day-5 semantic
search so weak matches never reach the LLM prompt.

Guardrails of this module:
  - Reuses search_faqs() (which routes through get_vector_store()) so the
    index is never rebuilt here — Day-5 lazy-builder is the only thing that
    ever builds.
  - Never raises on infrastructure problems: it logs and returns [].
  - An empty result is a FINE outcome — the service layer treats it as
    "no match found" and degrades gracefully.

Author: Ameema Rashid — RRVDXB AI Sprint, Day 6.
"""

import logging
from typing import Any, Dict, List

from app.ai.chatbot.rag.vector_store import search_faqs

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 3                  # how many candidate chunks to fetch
DEFAULT_SIMILARITY_THRESHOLD = 0.6  # min cosine similarity (exact return-policy = 0.66)
MAX_TOP_K = 10                     # hard cap: never blow up the prompt with noise


def retrieve_faq_context(
    query: str,
    k: int = DEFAULT_TOP_K,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Retrieve FAQ chunks relevant enough to ground a product_faq answer.

    Args:
        query: The customer's raw question (e.g. "what is your return policy?").
        k: How many candidate chunks to pull from the store (capped at 10).
        similarity_threshold: Minimum cosine similarity (0..1) a chunk must
            meet to be considered trustworthy context.

    Returns:
        List of dicts, most-similar first, each shaped:
            {"question": str, "answer": str, "similarity": float}
        Empty list when nothing qualifies — the service layer treats this
        as "no match found" and degrades gracefully.
    """
    # Guard 1: never query a blank string — nothing useful comes back.
    if not query or not query.strip():
        logger.debug("retrieve_faq_context: empty query — returning no context")
        return []

    # Guard 2: clamp k into the sane 1..10 range. k=0/negative would break
    # Chroma's n_results; a huge k would bloat the prompt with tokens + noise.
    k = max(1, min(k, MAX_TOP_K))
    # Guard 3: clamp the threshold into [0, 1] so callers can't pass nonsense.
    similarity_threshold = max(0.0, min(similarity_threshold, 1.0))

    # The vector store may not be built yet, the embedding model may be
    # missing, or Chroma may fail (e.g. a first-run without network). All of
    # that is caught here so the chat flow never crashes on retrieval.
    try:
        candidates = search_faqs(query, top_k=k)
    except Exception as exc:
        logger.warning(
            "retrieve_faq_context: search failed (%s: %s) — returning no context",
            type(exc).__name__,
            exc,
        )
        return []

    # Relevance gate: keep only chunks whose similarity clears the threshold,
    # and reshape them to exactly the contract build_rag_system_prompt() wants.
    filtered = [
        {
            "question": chunk["question"],
            "answer": chunk["answer"],
            "similarity": chunk["similarity"],
        }
        for chunk in candidates
        if chunk["similarity"] >= similarity_threshold
    ]

    if not filtered:
        logger.info(
            "retrieve_faq_context: no chunks met threshold %.2f for query=%r",
            similarity_threshold,
            query[:80],
        )
    return filtered