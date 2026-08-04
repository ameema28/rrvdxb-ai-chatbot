"""
pytest suite for the chatbot endpoint.

Ensures:
  - The endpoint responds HTTP 200 with a valid LLM reply.
  - The response conforms to the ChatResponse schema.
  - The LLM client is mocked so tests run fast, free, and deterministically.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


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