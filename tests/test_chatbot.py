"""
Basic pytest suite for the chatbot endpoint.

Ensures:
  - The placeholder endpoint responds HTTP 200.
  - The response conforms to the ChatResponse schema.
"""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_chat_endpoint_responds_200():
    """
    Verify POST /api/v1/ai/chat returns 200 with valid payload.

    We must include the X-User-Id header because get_current_user_id
    requires it. This also implicitly tests the dependency stub.
    """
    payload = {
        "message": "Hello, do you have iPhones?",
        "user_id": 1,
    }

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