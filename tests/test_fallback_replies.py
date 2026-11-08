"""
Deterministic fallback replies (fallback_used=true): built from content the
request already retrieved for the intent. LLM and FAQ retrieval are mocked.
"""

import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.ai.chatbot.intent import IntentResult
from app.ai.chatbot.price_guard import GUARD_REPLY
from app.core.config import settings
from app.main import app
from app.models.chat_history import ChatHistory
from app.services import fallback_reply
from tests.support import TestingSessionLocal, auth_headers

pytestmark = pytest.mark.db

client = TestClient(app)
CHAT = "/api/v1/ai/chat"

FAQ = [
    {
        "question": "What is your return policy?",
        "answer": "You can return unused items within 14 days of delivery.",
        "similarity": 0.8,
    }
]


def _chat(intent, message, faq=None, sub="fb-user", side_effect=None):
    with patch(
        "app.services.chatbot_service.send_chat_message",
        side_effect=side_effect or RuntimeError("Simulated Groq failure"),
    ), patch(
        "app.services.chatbot_service.classify_intent",
        return_value=IntentResult(intent=intent, confidence=1.0),
    ), patch(
        "app.services.chatbot_service.retrieve_faq_context",
        return_value=faq if faq is not None else [],
    ):
        response = client.post(CHAT, json={"message": message}, headers=auth_headers(sub))
    assert response.status_code == 200
    return response.json()


def test_product_faq_uses_the_matched_faq_answer():
    data = _chat("product_faq", "What is your return policy?", faq=FAQ)
    assert data["reply"] == FAQ[0]["answer"]
    assert data["fallback_used"] is True
    assert data["intent"] == "product_faq"


def test_product_faq_without_a_match_says_so_and_offers_support():
    data = _chat("product_faq", "What is the meaning of life?", faq=[])
    assert data["reply"] == fallback_reply.NO_FAQ_MATCH
    assert "human support" in data["reply"]
    assert data["fallback_used"] is True


def test_recommend_product_reply_is_a_lead_in_and_products_are_structured():
    data = _chat("recommend_product", "Can you recommend a perfume?")
    assert data["fallback_used"] is True
    assert data["reply"] == fallback_reply.PRODUCTS_LEAD_IN
    # products and catalog prices are only in the structured field (no duplicate)
    assert "Chanel" not in data["reply"] and "AED" not in data["reply"]
    assert [p["name"] for p in data["recommended_products"]] == [
        "Chanel Coco Mademoiselle Body Spray",
        "Harak Perfume Oud Edition",
    ]
    assert [p["price"] for p in data["recommended_products"]] == [189.0, 450.0]


def test_recommend_product_without_a_match_offers_support():
    data = _chat("recommend_product", "recommend me a unicorn saddle")
    assert data["reply"] == fallback_reply.NO_RECOMMENDATION
    assert data["fallback_used"] is True
    assert data["recommended_products"] is None


def test_deal_inquiry_uses_the_deal_text():
    data = _chat("deal_inquiry", "Are there any deals right now?")
    assert data["fallback_used"] is True
    assert data["reply"] == fallback_reply.DEAL_LEAD_IN
    assert "SUMMER20" in data["deal"] and "20% off" in data["deal"]
    assert "SUMMER20" not in data["reply"]  # the deal is shown from the structured field


def test_deal_inquiry_without_a_deal_offers_support():
    data = _chat("deal_inquiry", "tell me something")
    assert data["reply"] == fallback_reply.NO_DEAL
    assert data["fallback_used"] is True


def test_track_order_help_uses_the_tracking_guidance():
    data = _chat("track_order_help", "How do I track my order?")
    assert data["fallback_used"] is True
    for fact in fallback_reply.TRACKING_FACTS:
        assert fact in data["reply"]
    assert "My Orders" in data["reply"]


def test_general_chat_lists_what_sara_can_do():
    data = _chat("general_chat", "hello")
    assert data["reply"] == fallback_reply.HELP_MESSAGE
    assert data["fallback_used"] is True
    for topic in ("recommend", "returns", "deals", "track"):
        assert topic in data["reply"]


def test_fallback_turn_is_saved_with_the_real_intent():
    _chat("recommend_product", "Can you recommend a perfume?", sub="fb-save")
    with TestingSessionLocal() as db:
        row = db.query(ChatHistory).filter(ChatHistory.user_id == "fb-save").one()
    assert row.intent == "recommend_product"


def test_price_guard_still_runs_on_the_fallback_text():
    with patch(
        "app.services.chatbot_service.build_fallback_reply",
        return_value=("- Ladies Leather Handbag (Lacoste): AED 5.00", True),
    ):
        data = _chat("recommend_product", "recommend me a handbag", sub="fb-guard")
    assert data["reply"] == GUARD_REPLY
    assert data["fallback_used"] is True


def test_timeout_with_content_keeps_the_timeout_message_first(monkeypatch):
    monkeypatch.setattr(settings, "llm_timeout_seconds", 0.05)

    def slow(**_kwargs):
        time.sleep(0.3)
        return "too late"

    data = _chat("deal_inquiry", "Any discount today?", sub="fb-slow", side_effect=slow)
    assert data["reply"].startswith("I'm taking longer than usual.")
    assert data["reply"].endswith(fallback_reply.DEAL_LEAD_IN)
    assert "SUMMER20" in data["deal"]
    assert data["fallback_used"] is True
