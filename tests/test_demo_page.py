"""
Demo chat page at GET / and the reply-details fields it shows
(response_time_ms and fallback_used in ChatResponse).
"""

import re
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.ai.chatbot import intent as intent_module
from app.ai.chatbot.intent import IntentResult, classify_intent
from app.ai.chatbot.price_guard import GUARD_REPLY
from app.core.config import settings
from app.demo_page import DEMO_PAGE_PATH, csp_hash, load_demo_page
from app.main import app
from app.models.chat_history import ChatHistory
from tests.support import TestingSessionLocal, auth_headers

pytestmark = pytest.mark.db

client = TestClient(app)
CHAT = "/api/v1/ai/chat"
DEMO = "/api/v1/auth/demo-token"

EXAMPLES = {
    "Can you recommend a perfume?": "recommend_product",
    "What is your return policy?": "product_faq",
    "Are there any deals right now?": "deal_inquiry",
    "How do I track my order?": "track_order_help",
}


def _page() -> str:
    return client.get("/").text


def _script(html: str) -> str:
    return re.search(r"<script>(.*?)</script>", html, re.DOTALL).group(1)


def _style(html: str) -> str:
    return re.search(r"<style>(.*?)</style>", html, re.DOTALL).group(1)


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------

def test_root_serves_the_demo_page_with_strict_headers():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<meta name="rrvdxb-demo" content="chat">' in response.text
    csp = response.headers["content-security-policy"]
    assert "unsafe-inline" not in csp
    assert "default-src 'none'" in csp
    assert "connect-src 'self'" in csp
    assert f"script-src {csp_hash(_script(response.text))}" in csp
    assert f"style-src {csp_hash(_style(response.text))}" in csp
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_page_is_not_in_the_openapi_schema():
    assert "/" not in client.get("/openapi.json").json()["paths"]


def test_page_is_self_contained_and_same_origin():
    html = _page()
    assert "http://" not in html and "https://" not in html
    assert "<link" not in html
    assert "src=" not in html
    urls = set(re.findall(r'"(/[^"]*)"', _script(html)))
    assert urls == {"/api/v1/auth/demo-token", "/api/v1/ai/chat"}


def test_token_is_kept_in_memory_only_and_replies_use_text_only():
    script = _script(_page())
    for forbidden in ("localStorage", "sessionStorage", "document.cookie", "innerHTML", "outerHTML", "eval("):
        assert forbidden not in script
    assert "textContent" in script


def test_page_renews_the_token_on_401_and_reports_404_and_429():
    script = _script(_page())
    assert "response.status === 401" in script
    assert "Demo tokens are disabled on this server." in script
    assert "Too many requests" in script


def test_details_show_only_api_fields():
    script = _script(_page())
    labels = re.findall(r'\["(\w+)", data\.(\w+)\]', script)
    assert labels == [
        ("intent", "intent"),
        ("confidence", "confidence"),
        ("response_time_ms", "response_time_ms"),
        ("fallback_used", "fallback_used"),
    ]
    from app.schemas.chatbot_schema import ChatResponse

    for _, field in labels:
        assert field in ChatResponse.model_fields


def test_example_buttons_match_the_regex_fast_path():
    html = _page()
    buttons = re.findall(r'<button type="button">([^<]+)</button>', html)
    assert buttons == list(EXAMPLES)

    def no_llm(*_args, **_kwargs):
        raise AssertionError("the LLM must not be called for an example question")

    with patch.object(intent_module, "send_chat_message", side_effect=no_llm):
        for text, expected in EXAMPLES.items():
            result = classify_intent(text)
            assert result.intent == expected
            assert result.confidence == 1.0


def test_each_example_button_returns_useful_content_without_the_llm():
    """Offline mode: the LLM fails, yet every button shows real content."""
    faq = [{"question": "Return policy", "answer": "Unused items can be returned within 14 days.", "similarity": 0.9}]
    expected_text = {
        "Can you recommend a perfume?": "AED 189.00",
        "What is your return policy?": faq[0]["answer"],
        "Are there any deals right now?": "SUMMER20",
        "How do I track my order?": "My Orders",
    }
    assert set(expected_text) == set(EXAMPLES)
    for i, (text, intent) in enumerate(EXAMPLES.items()):
        with patch(
            "app.services.chatbot_service.send_chat_message", side_effect=RuntimeError("offline")
        ), patch("app.ai.chatbot.intent.send_chat_message", side_effect=RuntimeError("offline")), patch(
            "app.services.chatbot_service.retrieve_faq_context", return_value=faq
        ):
            data = client.post(CHAT, json={"message": text}, headers=auth_headers(f"btn-{i}")).json()
        assert data["intent"] == intent, text
        assert data["fallback_used"] is True
        assert expected_text[text] in data["reply"], text
        assert data["reply"] != GUARD_REPLY
        if intent == "recommend_product":
            assert data["recommended_products"], "the product button must return products"
            assert all(p["price"] > 0 for p in data["recommended_products"])
        if intent == "deal_inquiry":
            assert data["deal"]


def test_page_load_rejects_a_page_without_one_script(tmp_path):
    bad = tmp_path / "index.html"
    bad.write_text("<html><style>a{}</style></html>", encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_demo_page(bad)
    assert load_demo_page(DEMO_PAGE_PATH)[0] == _page()


def test_root_and_health_are_never_rate_limited(monkeypatch):
    monkeypatch.setattr(settings, "demo_tokens_enabled", True)
    monkeypatch.setattr(settings, "chat_ip_rate_per_minute", 1)
    for _ in range(settings.demo_token_rate_per_minute):
        client.post(DEMO)
    assert client.post(DEMO).status_code == 429
    with patch("app.services.chatbot_service.send_chat_message", return_value="ok"), patch(
        "app.services.chatbot_service.classify_intent",
        return_value=IntentResult(intent="general_chat", confidence=1.0),
    ):
        client.post(CHAT, json={"message": "hi"}, headers=auth_headers("limit-a"))
        assert client.post(CHAT, json={"message": "hi"}, headers=auth_headers("limit-b")).status_code == 429
    for _ in range(30):
        assert client.get("/").status_code == 200
        assert client.get("/health").status_code == 200


# --------------------------------------------------------------------------
# response_time_ms and fallback_used
# --------------------------------------------------------------------------

def _chat(reply=None, side_effect=None, intent="general_chat", sub="details-user"):
    kwargs = {"side_effect": side_effect} if side_effect else {"return_value": reply}
    with patch("app.services.chatbot_service.send_chat_message", **kwargs), patch(
        "app.services.chatbot_service.classify_intent",
        return_value=IntentResult(intent=intent, confidence=1.0),
    ):
        return client.post(CHAT, json={"message": "hello there"}, headers=auth_headers(sub))


def _saved_intent(sub: str) -> str:
    with TestingSessionLocal() as db:
        return db.query(ChatHistory).filter(ChatHistory.user_id == sub).one().intent


def test_llm_success_is_not_fallback():
    data = _chat(reply="Happy to help!").json()
    assert data["fallback_used"] is False
    assert isinstance(data["response_time_ms"], int)
    assert data["response_time_ms"] >= 0


def test_llm_error_is_fallback():
    data = _chat(side_effect=RuntimeError("Simulated Groq failure"), sub="err-user").json()
    assert data["fallback_used"] is True
    assert _saved_intent("err-user") == "general_chat"


def test_llm_timeout_is_fallback(monkeypatch):
    monkeypatch.setattr(settings, "llm_timeout_seconds", 0.05)

    def slow(**_kwargs):
        time.sleep(0.3)
        return "too late"

    data = _chat(side_effect=slow, sub="slow-user").json()
    assert data["fallback_used"] is True
    assert data["response_time_ms"] >= 50


def test_price_guard_replacement_is_not_counted_as_fallback():
    data = _chat(reply="The iPhone 14 Pro Max is only AED 6,499 today!", sub="guard-user").json()
    assert data["reply"] == GUARD_REPLY
    assert data["fallback_used"] is False


def test_llm_reply_mentioning_human_support_keeps_its_intent():
    reply = "Sure! Our human support team can also help with order tracking."
    data = _chat(reply=reply, intent="track_order_help", sub="support-user").json()
    assert data["fallback_used"] is False
    assert _saved_intent("support-user") == "track_order_help"
