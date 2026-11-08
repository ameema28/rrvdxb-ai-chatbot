"""
Conversation-memory queries (SQLAlchemy 2.0 style), on SQLite and PostgreSQL.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.ai.chatbot import memory
from app.ai.chatbot.intent import IntentResult
from app.ai.chatbot.memory import (
    format_history_for_prompt,
    load_recent_history,
    save_turn,
)
from app.main import app
from app.models.chat_history import ChatHistory
from tests.support import IS_SQLITE, TestingSessionLocal, auth_headers

pytestmark = pytest.mark.db


@pytest.fixture
def db():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_save_then_load_in_chronological_order(db):
    for i in range(3):
        save_turn(db, "mem-user", f"question {i}", f"answer {i}", "general_chat")
    turns = load_recent_history(db, "mem-user", limit=5)
    assert [t["content"] for t in turns] == [
        "question 0", "answer 0", "question 1", "answer 1", "question 2", "answer 2",
    ]
    assert format_history_for_prompt(turns).splitlines()[0] == "User: question 0"


def test_limit_keeps_the_most_recent_turns(db):
    for i in range(7):
        save_turn(db, "mem-limit", f"q{i}", f"a{i}")
    turns = load_recent_history(db, "mem-limit", limit=5)
    assert [t["content"] for t in turns if t["role"] == "user"] == ["q2", "q3", "q4", "q5", "q6"]


def test_identical_timestamps_are_ordered_by_id(db):
    stamp = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    for i in range(3):
        db.add(ChatHistory(user_id="mem-tie", message=f"t{i}", ai_response="r", created_at=stamp))
    db.commit()
    turns = load_recent_history(db, "mem-tie", limit=2)
    assert [t["content"] for t in turns if t["role"] == "user"] == ["t1", "t2"]


def test_history_is_isolated_per_user(db):
    save_turn(db, "mem-a", "a's private note", "ok")
    save_turn(db, "mem-b", "b's note", "ok")
    contents = [t["content"] for t in load_recent_history(db, "mem-b")]
    assert "a's private note" not in contents
    assert "b's note" in contents


def test_created_at_round_trip(db):
    row = save_turn(db, "mem-time", "hi", "hello")
    loaded = db.get(ChatHistory, row.id)
    assert loaded.created_at is not None
    if not IS_SQLITE:
        # PostgreSQL keeps the timezone; SQLite stores naive datetimes.
        assert loaded.created_at.tzinfo is not None


def test_user_id_is_stored_as_string_subject(db):
    sub = "demo-" + "f" * 32
    save_turn(db, sub, "hi", "hello")
    assert db.query(ChatHistory).filter(ChatHistory.user_id == sub).count() == 1


def test_chat_runs_history_load_and_save_off_the_event_loop():
    """Blocking DB calls run in a worker thread (no running event loop there)."""
    on_loop = {}

    def _record(name, func):
        def wrapper(*args, **kwargs):
            try:
                asyncio.get_running_loop()
                on_loop[name] = True
            except RuntimeError:
                on_loop[name] = False
            return func(*args, **kwargs)

        return wrapper

    with patch(
        "app.services.chatbot_service.load_recent_history",
        _record("load", memory.load_recent_history),
    ), patch(
        "app.services.chatbot_service.save_turn", _record("save", memory.save_turn)
    ), patch(
        "app.services.chatbot_service.send_chat_message", return_value="Hello!"
    ), patch(
        "app.services.chatbot_service.classify_intent",
        return_value=IntentResult(intent="general_chat", confidence=1.0),
    ):
        response = TestClient(app).post(
            "/api/v1/ai/chat", json={"message": "hi"}, headers=auth_headers("thread-user")
        )
    assert response.status_code == 200
    assert on_loop == {"load": False, "save": False}
