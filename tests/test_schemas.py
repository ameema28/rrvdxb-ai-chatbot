"""
pytest suite for the chatbot API schemas (Day 9).

Unit-level validation of ChatRequest / ChatResponse — independent of the
HTTP layer, instant, and deterministic.

Note on status codes: at the schema level, violations raise pydantic
ValidationError. Over HTTP, FastAPI turns those into RequestValidationError
(422) and our validation_error_handler maps them to the canonical
{"error", "detail", "status_code": 400} body — see
test_malformed_request_body_returns_canonical_400 in test_chatbot.py.
"""

import pytest
from pydantic import ValidationError

from app.schemas.chatbot_schema import ChatRequest, ChatResponse


# --------------------------------------------------------------------------
# ChatRequest — input validation edges
# --------------------------------------------------------------------------

def test_chat_request_rejects_empty_message():
    """Empty message violates min_length=1."""
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_chat_request_rejects_message_over_2000_chars():
    """Message over max_length=2000 is rejected (prompt-inflation guard)."""
    with pytest.raises(ValidationError):
        ChatRequest(message="a" * 2001)


def test_chat_request_rejects_missing_message():
    """message is a required field — omitting it must fail validation."""
    with pytest.raises(ValidationError):
        ChatRequest(user_id=1)


def test_chat_request_rejects_wrong_type_user_id():
    """user_id must be an int — 'not-an-int' must fail validation."""
    with pytest.raises(ValidationError):
        ChatRequest(message="hello", user_id="not-an-int")


def test_chat_request_accepts_valid_payload():
    """Happy path: message + integer user_id validate cleanly."""
    request = ChatRequest(message="recommend a perfume", user_id=42)
    assert request.message == "recommend a perfume"
    assert request.user_id == 42


def test_chat_request_user_id_is_optional():
    """user_id defaults to None (the X-User-Id header is the real identity)."""
    request = ChatRequest(message="hello")
    assert request.user_id is None


# --------------------------------------------------------------------------
# ChatResponse — serialization round-trip and defaults
# --------------------------------------------------------------------------

def test_chat_response_round_trips_recommendations_deal_intent():
    """A fully-populated response round-trips dict -> model -> dict losslessly."""
    payload = {
        "reply": "I'd recommend the iPhone 14 Pro Max at 4,699 AED.",
        "recommended_products": [
            {
                "id": 1,
                "name": "iPhone 14 Pro Max",
                "price": 4699.0,
                "currency": "AED",
                "category": "Electronics",
                "brand": "Apple",
                "reason": "Flagship pick",
            }
        ],
        "deal": "Summer Sale - 20% off (code: SUMMER20)",
        "intent": "recommend_product",
        "confidence": 0.95,
    }
    response = ChatResponse.model_validate(payload)
    assert response.model_dump() == payload
    assert response.recommended_products[0].name == "iPhone 14 Pro Max"
    assert response.recommended_products[0].currency == "AED"


def test_chat_response_optional_fields_default_to_none():
    """recommended_products / deal / intent / confidence are all optional."""
    response = ChatResponse(reply="hi")
    assert response.recommended_products is None
    assert response.deal is None
    assert response.intent is None
    assert response.confidence is None


def test_chat_response_rejects_out_of_range_confidence():
    """confidence is constrained to [0.0, 1.0] — 1.5 must fail validation."""
    with pytest.raises(ValidationError):
        ChatResponse(reply="hi", confidence=1.5)
