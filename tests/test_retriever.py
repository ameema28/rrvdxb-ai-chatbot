"""
pytest suite for the RAG retriever (Day 6).

Covers the relevance gate and graceful degradation without touching the real
vector store — search_faqs is mocked so tests are instant and deterministic.
"""

from unittest.mock import patch

from app.ai.chatbot.rag.retriever import retrieve_faq_context


def _fake_candidates():
    """Mixed-relevance candidates as search_faqs() would return them."""
    return [
        {"id": 1, "question": "What is your return policy?",
         "answer": "Returns within 14 days.", "similarity": 0.92},
        {"id": 2, "question": "Do you offer free shipping?",
         "answer": "Free shipping over 489 AED.", "similarity": 0.85},
        {"id": 3, "question": "Do you sell watches?",
         "answer": "No.", "similarity": 0.45},
        {"id": 4, "question": "How old are you?",
         "answer": "N/A", "similarity": 0.30},
    ]


def test_retriever_keeps_only_above_threshold_chunks():
    with patch(
        "app.ai.chatbot.rag.retriever.search_faqs",
        return_value=_fake_candidates(),
    ):
        result = retrieve_faq_context("what is your return policy?")

    # Relevance gate with the default threshold 0.6: 0.92 and 0.85 survive.
    assert len(result) == 2
    assert all(chunk["similarity"] >= 0.6 for chunk in result)
    assert result[0]["question"] == "What is your return policy?"
    # Output contract is exactly question / answer / similarity.
    assert set(result[0].keys()) == {"question", "answer", "similarity"}
    assert "id" not in result[0]


def test_retriever_returns_empty_when_nothing_passes_threshold():
    below = [
        {"id": 1, "question": "Q1", "answer": "A1", "similarity": 0.55},
        {"id": 2, "question": "Q2", "answer": "A2", "similarity": 0.40},
    ]
    with patch(
        "app.ai.chatbot.rag.retriever.search_faqs",
        return_value=below,
    ):
        result = retrieve_faq_context("gibberish query zzz")
    assert result == []


def test_retriever_returns_empty_when_search_fails():
    # Vector store not built / embedding model missing -> search_faqs raises.
    with patch(
        "app.ai.chatbot.rag.retriever.search_faqs",
        side_effect=RuntimeError("vector store not built"),
    ):
        result = retrieve_faq_context("what is your return policy?")
    assert result == []


def test_retriever_empty_query_returns_empty():
    # Empty/blank queries short-circuit before touching the store.
    assert retrieve_faq_context("") == []
    assert retrieve_faq_context("   ") == []


def test_retriever_forward_k_to_search():
    with patch(
        "app.ai.chatbot.rag.retriever.search_faqs",
        return_value=_fake_candidates(),
    ) as mock_search:
        retrieve_faq_context("free shipping?", k=2, similarity_threshold=0.0)
    mock_search.assert_called_once_with("free shipping?", top_k=2)


def test_retriever_clamps_k_and_threshold():
    # k=25 clamps to MAX_TOP_K=10; threshold 1.5 clamps to 1.0.
    with patch(
        "app.ai.chatbot.rag.retriever.search_faqs",
        return_value=_fake_candidates(),
    ) as mock_search:
        retrieve_faq_context("q", k=25, similarity_threshold=1.5)
    mock_search.assert_called_once_with("q", top_k=10)