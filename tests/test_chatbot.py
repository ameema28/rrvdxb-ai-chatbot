"""
pytest suite for the chatbot endpoint and conversation memory.

Ensures:
  - The endpoint responds HTTP 200 with a valid LLM reply.
  - The response conforms to the ChatResponse schema.
  - The LLM client is mocked so tests run fast, free, and deterministically.
  - Conversation memory persists across multiple turns in the DB.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.database import Base
from app.core.dependencies import get_db

# CRITICAL: Import the model so SQLAlchemy registers the chat_history table
# in Base.metadata BEFORE we call create_all(). Without this import,
# Base.metadata is empty and create_all() creates zero tables.
from app.models.chat_history import ChatHistory

client = TestClient(app)

# --------------------------------------------------------------------------
# In-memory SQLite DB for memory tests (isolated, fast, no file I/O)
# --------------------------------------------------------------------------
TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,  # <-- CRITICAL: reuses one connection for :memory:
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Create tables once for the test session — ChatHistory import above is required
Base.metadata.create_all(bind=test_engine)


def override_get_db():
    """Yield a fresh in-memory DB session per test."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Override the production get_db dependency with our test DB
app.dependency_overrides[get_db] = override_get_db


def _extract_history_arg(mock_call_args):
    """
    Safely extract the 'history' argument from a mocked send_chat_message call.
    Handles both positional and keyword invocation.
    """
    if "history" in mock_call_args.kwargs:
        return mock_call_args.kwargs["history"]
    if len(mock_call_args.args) > 3:
        return mock_call_args.args[3]
    return ""


def test_chat_endpoint_responds_200():
    """
    Verify POST /api/v1/ai/chat returns 200 with a real AI-generated reply.

    The LLM client is mocked to avoid real API calls during tests.
    """
    payload = {
        "message": "Hello, do you have iPhones?",
        "user_id": 1,
    }

    mock_reply = (
        "Hi there! Yes, we have the iPhone 14 Pro Max in stock — "
        "256GB in Deep Purple for 4,699 AED. Would you like more details?"
    )

    with patch(
        "app.services.chatbot_service.send_chat_message",
        return_value=mock_reply,
    ):
        response = client.post(
            "/api/v1/ai/chat",
            json=payload,
            headers={"X-User-Id": "1"},
        )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert "reply" in data
    assert isinstance(data["reply"], str)
    assert len(data["reply"]) > 0

    # Day 2 guardrail: the reply must NOT be the old placeholder echo text
    assert "You asked:" not in data["reply"], "Response still contains Day 1 placeholder text"
    assert "currently learning" not in data["reply"], "Response still contains Day 1 placeholder text"

    # The reply should contain our mock content
    assert "iPhone 14 Pro Max" in data["reply"], "Mock reply content not found in response"


def test_chat_endpoint_requires_user_id_header():
    """
    Verify the endpoint rejects requests without X-User-Id.
    This tests the authentication stub behavior.
    """
    payload = {"message": "Test without auth"}

    response = client.post(
        "/api/v1/ai/chat",
        json=payload,
        # No X-User-Id header
    )

    assert response.status_code == 401


def test_chat_endpoint_falls_back_on_llm_error():
    """
    Verify the endpoint returns a friendly fallback when the LLM call fails.
    """
    payload = {
        "message": "This will trigger an LLM error",
        "user_id": 1,
    }

    with patch(
        "app.services.chatbot_service.send_chat_message",
        side_effect=RuntimeError("Simulated Groq failure"),
    ):
        response = client.post(
            "/api/v1/ai/chat",
            json=payload,
            headers={"X-User-Id": "1"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "human support" in data["reply"] or "trouble" in data["reply"].lower()


def test_conversation_memory_persists_across_turns():
    """
    Day 3 — Verify that two sequential messages from the same user:
      1. Are both saved to chat_history (2 rows total).
      2. The second LLM call receives the first turn in its history parameter.

    We use the real in-memory SQLite DB (not mocked) to prove persistence works.
    The Groq client is mocked to avoid real API calls.
    """
    user_id = 42

    # ------------------------------------------------------------------
    # Turn 1: "I am looking for a gift"
    # ------------------------------------------------------------------
    first_mock_reply = (
        "That sounds lovely! What occasion is the gift for, "
        "and do you have a budget in mind?"
    )

    with patch(
        "app.services.chatbot_service.send_chat_message",
        return_value=first_mock_reply,
    ) as mock_llm:
        response1 = client.post(
            "/api/v1/ai/chat",
            json={"message": "I am looking for a gift", "user_id": user_id},
            headers={"X-User-Id": str(user_id)},
        )
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["reply"] == first_mock_reply

        # Capture the call arguments for Turn 1
        history_arg_1 = _extract_history_arg(mock_llm.call_args)
        assert history_arg_1 == "", "First turn should not have prior history"

    # ------------------------------------------------------------------
    # Turn 2: "Something under 500 AED" (follow-up, relies on memory)
    # ------------------------------------------------------------------
    second_mock_reply = (
        "Great! Here are some gift ideas under 500 AED: "
        "Lacoste L.12.12 Pour Lui (299 AED) and Adidas Ultraboost (450 AED)."
    )

    with patch(
        "app.services.chatbot_service.send_chat_message",
        return_value=second_mock_reply,
    ) as mock_llm:
        response2 = client.post(
            "/api/v1/ai/chat",
            json={"message": "Something under 500 AED", "user_id": user_id},
            headers={"X-User-Id": str(user_id)},
        )
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["reply"] == second_mock_reply

        # Capture the call arguments for Turn 2
        history_arg_2 = _extract_history_arg(mock_llm.call_args)

        # The history passed to the LLM for Turn 2 MUST contain Turn 1
        assert "I am looking for a gift" in history_arg_2, (
            "Second turn's prompt did not include first user's message"
        )
        assert first_mock_reply in history_arg_2, (
            "Second turn's prompt did not include first AI response"
        )

    # ------------------------------------------------------------------
    # Verify DB persistence: exactly 2 rows for this user
    # ------------------------------------------------------------------
    db = TestingSessionLocal()
    rows = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).all()
    db.close()

    assert len(rows) == 2, f"Expected 2 chat_history rows, got {len(rows)}"
    assert rows[0].message == "I am looking for a gift"
    assert rows[0].ai_response == first_mock_reply
    assert rows[1].message == "Something under 500 AED"
    assert rows[1].ai_response == second_mock_reply