"""
Startup warm-up of the embedding model and FAQ index, and /health readiness.
"""

import time

import pytest
from fastapi.testclient import TestClient

from app.ai.chatbot.rag import retriever, vector_store
from app.core.config import settings


@pytest.fixture
def fresh_state(monkeypatch):
    monkeypatch.setattr(vector_store, "_ready", False)
    monkeypatch.setattr(settings, "db_auto_migrate", False)
    return monkeypatch


def _app():
    from app.main import create_application

    return create_application()


def test_lifespan_runs_warm_up_when_enabled(fresh_state):
    calls = []

    def fake_warm_up():
        calls.append(1)
        vector_store._ready = True
        return True

    fresh_state.setattr(settings, "rag_warmup", True)
    fresh_state.setattr(vector_store, "warm_up", fake_warm_up)
    with TestClient(_app()) as c:
        body = c.get("/health").json()
    assert calls == [1]
    assert body["status"] == "ok"
    assert body["rag_ready"] is True


def test_lifespan_skips_warm_up_when_disabled(fresh_state):
    calls = []
    fresh_state.setattr(settings, "rag_warmup", False)
    fresh_state.setattr(vector_store, "warm_up", lambda: calls.append(1))
    with TestClient(_app()) as c:
        response = c.get("/health")
    assert calls == []
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["rag_ready"] is False


def test_failed_warm_up_still_starts_and_reports_degraded(fresh_state, caplog):
    def broken_model():
        raise RuntimeError("model missing")

    fresh_state.setattr(settings, "rag_warmup", True)
    fresh_state.setattr(vector_store, "_embedding_model", None)
    fresh_state.setattr(vector_store, "_get_embedding_model", broken_model)
    with TestClient(_app()) as c:
        response = c.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "service": settings.app_name,
        "rag_ready": False,
    }
    assert "RAG warm-up failed (RuntimeError)" in caplog.text
    assert "model missing" not in caplog.text


def test_empty_index_is_not_ready(fresh_state):
    class EmptyCollection:
        def count(self):
            return 0

    fresh_state.setattr(vector_store, "_get_embedding_model", lambda: object())
    fresh_state.setattr(vector_store, "get_vector_store", lambda: EmptyCollection())
    assert vector_store.warm_up() is False
    assert vector_store.is_ready() is False


def test_search_faqs_turns_off_the_progress_bar(monkeypatch):
    """With INFO logging, sentence-transformers would print a tqdm bar per query."""
    seen = []

    class FakeModel:
        def encode(self, texts, **kwargs):
            seen.append(kwargs)

            class Vectors:
                def tolist(self):
                    return [[0.0]]

            return Vectors()

    class FakeCollection:
        def count(self):
            return 1

        def query(self, **kwargs):
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    monkeypatch.setattr(vector_store, "get_vector_store", lambda: FakeCollection())
    monkeypatch.setattr(vector_store, "_get_embedding_model", lambda: FakeModel())
    assert vector_store.search_faqs("return policy") == []
    assert seen and seen[0]["show_progress_bar"] is False


@pytest.mark.chroma
def test_real_warm_up_makes_the_first_query_fast(tmp_path, monkeypatch):
    """After warm-up the first FAQ query fits well inside the 1.0 s retrieval cap."""
    monkeypatch.setattr(vector_store, "_PERSIST_DIR", tmp_path / "chroma_db")
    monkeypatch.setattr(vector_store, "_collection", None)
    monkeypatch.setattr(vector_store, "_ready", False)

    assert vector_store.warm_up() is True
    assert vector_store.is_ready() is True

    started = time.perf_counter()
    chunks = retriever.retrieve_faq_context("what is your return policy?")
    elapsed = time.perf_counter() - started
    assert chunks
    assert elapsed < settings.retrieval_timeout_seconds
