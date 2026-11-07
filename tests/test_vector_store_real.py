"""
Unmocked RAG retrieval: build a real ChromaDB index from faqs.json in a temp
directory with the local all-MiniLM-L6-v2 embedding model, then query it.

Needs the embedding model in the Hugging Face cache (~90 MB download on the
very first run; CI caches ~/.cache/huggingface). No LLM and no API key.
"""

import pytest

from app.ai.chatbot.rag import retriever, vector_store

pytestmark = pytest.mark.chroma


@pytest.fixture(scope="module")
def real_index(tmp_path_factory):
    persist_dir = tmp_path_factory.mktemp("chroma_db")
    mp = pytest.MonkeyPatch()
    mp.setattr(vector_store, "_PERSIST_DIR", persist_dir)
    mp.setattr(vector_store, "_collection", None)
    try:
        count = vector_store.build_vector_store()
        yield count
    finally:
        mp.undo()


def test_build_indexes_every_faq(real_index):
    assert real_index == 12


@pytest.mark.parametrize(
    "query, expected_id",
    [
        ("what is your return policy?", 5),
        ("do you deliver to Karachi in Pakistan?", 3),
        ("is there a minimum order for free shipping?", 7),
    ],
)
def test_top_hit_is_the_right_faq(real_index, query, expected_id):
    hits = vector_store.search_faqs(query, top_k=3)
    assert hits, "no results from the real index"
    assert hits[0]["id"] == expected_id


def test_retriever_gate_passes_the_real_return_policy_hit(real_index):
    chunks = retriever.retrieve_faq_context("what is your return policy?")
    assert chunks
    assert chunks[0]["question"] == "What is your return policy?"
    assert chunks[0]["similarity"] >= retriever.DEFAULT_SIMILARITY_THRESHOLD
