"""
pytest suite for the chatbot endpoint and conversation memory.

Ensures:
  - The endpoint responds HTTP 200 with a valid LLM reply.
  - The response conforms to the ChatResponse schema.
  - The LLM client is mocked so tests run fast, free, and deterministically.
  - Conversation memory persists across multiple turns in the DB.
  - The classified intent is persisted with each turn.
  - product_faq intents are grounded in retrieved FAQ context (Day 6 RAG).
  - recommend_product / deal_inquiry populate recommended_products and deal (Day 7).
  - LLM errors and timeouts degrade gracefully (Day 7), turn still saved.
"""

from unittest.mock import patch

import asyncio

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.ai.chatbot.intent import IntentResult
from app.core.database import Base
from app.core.dependencies import get_db

from app.models.chat_history import ChatHistory

client = TestClient(app)

# --------------------------------------------------------------------------
# In-memory SQLite DB for memory tests (isolated, fast, no file I/O)
# --------------------------------------------------------------------------
TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

Base.metadata.create_all(bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


def _mock_intent(intent: str = "general_chat", confidence: float = 1.0):
    return patch(
        "app.services.chatbot_service.classify_intent",
        return_value=IntentResult(intent=intent, confidence=confidence, entities={}),
    )


def _extract_history_arg(mock_call_args):
    if "history" in mock_call_args.kwargs:
        return mock_call_args.kwargs["history"]
    if len(mock_call_args.args) > 3:
        return mock_call_args.args[3]
    return ""


# ==========================================================================
# Existing tests (unchanged behavior)
# ==========================================================================

def test_chat_endpoint_responds_200():
    payload = {"message": "Hello, do you have iPhones?", "user_id": 1}
    mock_reply = (
        "Hi there! Yes, we have the iPhone 14 Pro Max in stock — "
        "256GB in Deep Purple for 4,699 AED. Would you like more details?"
    )
    with patch(
        "app.services.chatbot_service.send_chat_message",
        return_value=mock_reply,
    ), _mock_intent("recommend_product"):
        response = client.post(
            "/api/v1/ai/chat",
            json=payload,
            headers={"X-User-Id": "1"},
        )

    assert response.status_code == 200, f"Got {response.status_code}: {response.text}"
    data = response.json()
    assert "reply" in data and len(data["reply"]) > 0
    assert "You asked:" not in data["reply"]
    assert "currently learning" not in data["reply"]
    assert "iPhone 14 Pro Max" in data["reply"]
    assert data["intent"] == "recommend_product"


def test_chat_endpoint_requires_user_id_header():
    payload = {"message": "Test without auth"}
    response = client.post("/api/v1/ai/chat", json=payload)
    assert response.status_code == 401


def test_chat_endpoint_falls_back_on_llm_error():
    payload = {"message": "This will trigger an LLM error", "user_id": 1}
    with patch(
        "app.services.chatbot_service.send_chat_message",
        side_effect=RuntimeError("Simulated Groq failure"),
    ), _mock_intent("general_chat"):
        response = client.post(
            "/api/v1/ai/chat",
            json=payload,
            headers={"X-User-Id": "1"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "human support" in data["reply"] or "trouble" in data["reply"].lower()
    assert data["intent"] == "general_chat"


def test_conversation_memory_persists_across_turns():
    user_id = 42
    first_mock_reply = (
        "That sounds lovely! What occasion is the gift for, "
        "and do you have a budget in mind?"
    )
    with patch(
        "app.services.chatbot_service.send_chat_message",
        return_value=first_mock_reply,
    ) as mock_llm, _mock_intent("recommend_product"):
        response1 = client.post(
            "/api/v1/ai/chat",
            json={"message": "I am looking for a gift", "user_id": user_id},
            headers={"X-User-Id": str(user_id)},
        )
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["reply"] == first_mock_reply
        assert data1["intent"] == "recommend_product"
        history_arg_1 = _extract_history_arg(mock_llm.call_args)
        assert history_arg_1 == ""

    second_mock_reply = (
        "Great! Here are some gift ideas under 500 AED: "
        "Lacoste L.12.12 Pour Lui (299 AED) and Adidas Ultraboost (450 AED)."
    )
    with patch(
        "app.services.chatbot_service.send_chat_message",
        return_value=second_mock_reply,
    ) as mock_llm, _mock_intent("recommend_product"):
        response2 = client.post(
            "/api/v1/ai/chat",
            json={"message": "Something under 500 AED", "user_id": user_id},
            headers={"X-User-Id": str(user_id)},
        )
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["reply"] == second_mock_reply
        history_arg_2 = _extract_history_arg(mock_llm.call_args)
        assert "I am looking for a gift" in history_arg_2
        assert first_mock_reply in history_arg_2

    db = TestingSessionLocal()
    rows = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).all()
    db.close()
    assert len(rows) == 2
    assert rows[0].message == "I am looking for a gift"
    assert rows[0].ai_response == first_mock_reply
    assert rows[0].intent == "recommend_product"
    assert rows[1].message == "Something under 500 AED"
    assert rows[1].ai_response == second_mock_reply
    assert rows[1].intent == "recommend_product"


def test_intent_is_saved_to_chat_history():
    user_id = 99
    with patch(
        "app.services.chatbot_service.send_chat_message",
        return_value="Here are today's top offers!",
    ), _mock_intent("deal_inquiry"):
        response = client.post(
            "/api/v1/ai/chat",
            json={"message": "Show me today's deals", "user_id": user_id},
            headers={"X-User-Id": str(user_id)},
        )
    assert response.status_code == 200
    assert response.json()["intent"] == "deal_inquiry"
    db = TestingSessionLocal()
    row = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).one()
    db.close()
    assert row.intent == "deal_inquiry"


# ==========================================================================
# Day 6 — RAG pipeline tests
# ==========================================================================

_FAKE_FAQ_CHUNKS = [
    {
        "question": "What is your return policy?",
        "answer": "You can return any unworn item within 30 days for a full refund.",
        "similarity": 0.92,
    },
    {
        "question": "Do you offer free shipping?",
        "answer": "Yes, shipping is free on orders over 489 AED.",
        "similarity": 0.85,
    },
]


def test_product_faq_uses_retrieved_faq_context():
    payload = {"message": "what is your return policy?", "user_id": 7}
    mock_reply = "You can return any unworn item within 30 days for a full refund."
    with patch(
        "app.services.chatbot_service.retrieve_faq_context",
        return_value=_FAKE_FAQ_CHUNKS,
    ) as mock_retriever, patch(
        "app.services.chatbot_service.send_chat_message",
        return_value=mock_reply,
    ) as mock_llm, _mock_intent("product_faq"):
        response = client.post(
            "/api/v1/ai/chat",
            json=payload,
            headers={"X-User-Id": "7"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == mock_reply
    assert data["intent"] == "product_faq"
    mock_retriever.assert_called_once_with("what is your return policy?")
    kwargs = mock_llm.call_args.kwargs
    assert kwargs["user_message"] == "what is your return policy?"
    assert "history" in kwargs
    rag_prompt = kwargs.get("system_prompt_override") or ""
    assert "FAQ CONTEXT:" in rag_prompt
    assert "Q: What is your return policy?" in rag_prompt
    assert "A: You can return any unworn item within 30 days" in rag_prompt
    assert "STRICT GUARDRAILS" in rag_prompt
    assert kwargs.get("product_context") == ""
    assert "30 days" in data["reply"]


def test_product_faq_no_rag_match_falls_back_gracefully():
    payload = {"message": "what is your return policy?", "user_id": 8}
    mock_reply = (
        "I don't have that specific policy on file right now — let me "
        "connect you with our support team for an accurate answer."
    )
    with patch(
        "app.services.chatbot_service.retrieve_faq_context",
        return_value=[],
    ) as mock_retriever, patch(
        "app.services.chatbot_service.send_chat_message",
        return_value=mock_reply,
    ) as mock_llm, _mock_intent("product_faq"):
        response = client.post(
            "/api/v1/ai/chat",
            json=payload,
            headers={"X-User-Id": "8"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == mock_reply
    assert data["intent"] == "product_faq"
    mock_retriever.assert_called_once_with("what is your return policy?")
    kwargs = mock_llm.call_args.kwargs
    assert kwargs.get("system_prompt_override") is None
    assert "FAQ CONTEXT:" not in kwargs.get("system_prompt", "")
    assert "no matching FAQ" in kwargs.get("system_prompt", "")
    assert "Do NOT invent" in kwargs.get("system_prompt", "")
    assert kwargs["product_context"] != ""


# ==========================================================================
# Day 7 — recommendations + deals + timeout/fallback
# ==========================================================================

_VALID_PRODUCT_IDS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}


def test_recommend_product_populates_recommended_products():
    payload = {"message": "recommend me a phone", "user_id": 11}
    mock_reply = (
        "I'd recommend the iPhone 14 Pro Max — a 256GB flagship with "
        "Dynamic Island, now at 4,699 AED. Would you like to see more?"
    )
    stub_recommendation = {
        "id": 1,
        "name": "iPhone 14 Pro Max",
        "price": 4699.0,
        "currency": "AED",
        "category": "Electronics",
        "brand": "Apple",
        "reason": "Matches your request for phone",
    }
    with patch(
        "app.services.chatbot_service.send_chat_message",
        return_value=mock_reply,
    ), patch(
        "app.services.chatbot_service.get_recommendations",
        return_value=[stub_recommendation],
    ), _mock_intent("recommend_product"):
        response = client.post(
            "/api/v1/ai/chat",
            json=payload,
            headers={"X-User-Id": "11"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "recommend_product"
    assert data["reply"] == mock_reply
    # (a) recommended_products is populated
    assert data["recommended_products"] is not None and len(data["recommended_products"]) > 0
    # (b) ids are ints within the products.json range
    ids = {rec["id"] for rec in data["recommended_products"]}
    assert ids.issubset(_VALID_PRODUCT_IDS)
    # (c) the stub reason surfaces
    assert data["recommended_products"][0]["reason"] == "Matches your request for phone"


def test_deal_inquiry_populates_deal():
    payload = {"message": "any discounts today?", "user_id": 12}
    mock_reply = "Yes! We're running our Summer Sale — 20% off site-wide with code SUMMER20."
    with patch(
        "app.services.chatbot_service.send_chat_message",
        return_value=mock_reply,
    ), patch(
        "app.services.chatbot_service.get_deals",
        return_value={"title": "Summer Sale", "discount": "20% off", "code": "SUMMER20"},
    ), _mock_intent("deal_inquiry"):
        response = client.post(
            "/api/v1/ai/chat",
            json=payload,
            headers={"X-User-Id": "12"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "deal_inquiry"
    assert data["reply"] == mock_reply
    # deal is populated with a human-readable string
    assert data["deal"] is not None
    assert "Summer Sale" in data["deal"]
    assert "SUMMER20" in data["deal"]


def test_deal_inquiry_no_deal_falls_back_to_general():
    payload = {"message": "please help", "user_id": 13}
    mock_reply = "Certainly — how can I help with your RRVDXB shopping today?"
    with patch(
        "app.services.chatbot_service.send_chat_message",
        return_value=mock_reply,
    ), patch(
        "app.services.chatbot_service.get_deals",
        return_value=None,
    ), _mock_intent("deal_inquiry"):
        response = client.post(
            "/api/v1/ai/chat",
            json=payload,
            headers={"X-User-Id": "13"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "deal_inquiry"
    assert data["deal"] is None


def test_llm_timeout_returns_timeout_message_and_saves_general_chat():
    user_id = 14
    payload = {"message": "recommend me a laptop", "user_id": user_id}

    with patch(
        "app.services.chatbot_service.send_chat_message",
        side_effect=asyncio.TimeoutError("simulated timeout"),
    ), _mock_intent("recommend_product"):
        response = client.post(
            "/api/v1/ai/chat",
            json=payload,
            headers={"X-User-Id": str(user_id)},
        )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["reply"] == "I'm taking longer than usual. Please try again in a moment."
    # observability: original intent preserved in the API payload
    assert data["intent"] == "recommend_product"

    db = TestingSessionLocal()
    row = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).one()
    db.close()
    # fallback turns are persisted with a VALID intent (general_chat)
    assert row.intent == "general_chat"


def test_api_surfaces_floor_confidence_for_offtopic_chat():
    """
    Off-topic query -> LLM reports ~0.1; the API response should surface the
    fallback floor (0.9) without changing intent-level confidence behavior.
    """
    payload = {"message": "what is the meaning of life?", "user_id": 15}
    mock_reply = (
        "I'm Sara, your RRVDXB shopping assistant — how can I help you find "
        "something today?"
    )
    with patch(
        "app.services.chatbot_service.send_chat_message",
        return_value=mock_reply,
    ), _mock_intent("general_chat", confidence=0.1):
        response = client.post(
            "/api/v1/ai/chat",
            json=payload,
            headers={"X-User-Id": "15"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "general_chat"
    assert data["confidence"] >= 0.7