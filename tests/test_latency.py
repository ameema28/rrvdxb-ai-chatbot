"""
Request time budget: slow fakes for the intent LLM, retrieval and reply.

Timings are measured around handle_chat_message itself (not the HTTP
client), because abandoned worker threads keep sleeping in the background
and the event loop waits for them on shutdown.
"""

import asyncio
import time
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from app.ai.chatbot import llm_client
from app.ai.chatbot.intent import IntentResult
from app.core.config import settings
from app.services import chatbot_service
from tests.support import TestingSessionLocal

pytestmark = pytest.mark.db

TIMEOUT_REPLY = "I'm taking longer than usual. Please try again in a moment."
NO_REGEX_MESSAGE = "hmm something nice maybe"  # misses every regex fast-path


def _slow(seconds, value):
    def _fake(*args, **kwargs):
        time.sleep(seconds)
        return value
    return _fake


def _run(message: str):
    async def _timed():
        db = TestingSessionLocal()
        try:
            started = time.perf_counter()
            response = await chatbot_service.handle_chat_message(
                user_id="latency-user", message=message, db=db
            )
            return response, time.perf_counter() - started
        finally:
            db.close()
    return asyncio.run(_timed())


@pytest.fixture
def fast_budget(monkeypatch):
    monkeypatch.setattr(settings, "request_budget_seconds", 1.0)
    monkeypatch.setattr(settings, "intent_timeout_seconds", 0.1)
    monkeypatch.setattr(settings, "retrieval_timeout_seconds", 0.1)
    monkeypatch.setattr(settings, "llm_timeout_seconds", 0.3)


def test_slow_intent_llm_falls_back_to_general_chat(fast_budget):
    with patch(
        "app.ai.chatbot.intent.send_chat_message",
        side_effect=_slow(0.6, '{"intent": "deal_inquiry", "confidence": 0.95}'),
    ), patch(
        "app.services.chatbot_service.send_chat_message", return_value="Here to help!"
    ):
        response, elapsed = _run(NO_REGEX_MESSAGE)
    assert response.intent == "general_chat"
    assert response.confidence == 0.0
    assert response.reply == "Here to help!"
    assert elapsed < 0.5  # bounded by the 0.1 s intent cap, not the 0.6 s fake


def test_slow_retrieval_uses_no_match_path(fast_budget):
    chunks = [{"question": "Q", "answer": "A", "similarity": 0.9}]
    with patch(
        "app.services.chatbot_service.classify_intent",
        return_value=IntentResult(intent="product_faq", confidence=1.0),
    ), patch(
        "app.services.chatbot_service.retrieve_faq_context",
        side_effect=_slow(0.6, chunks),
    ), patch(
        "app.services.chatbot_service.send_chat_message", return_value="Let me check."
    ) as llm:
        response, elapsed = _run("what is your return policy?")
    kwargs = llm.call_args.kwargs
    assert kwargs.get("system_prompt_override") is None
    assert "no matching FAQ" in kwargs["system_prompt"]
    assert response.intent == "product_faq"
    assert elapsed < 0.5


def test_slow_reply_gets_timeout_message(fast_budget):
    with patch(
        "app.services.chatbot_service.classify_intent",
        return_value=IntentResult(intent="general_chat", confidence=1.0),
    ), patch(
        "app.services.chatbot_service.send_chat_message",
        side_effect=_slow(1.0, "too late"),
    ):
        response, elapsed = _run("hello")
    assert response.reply == TIMEOUT_REPLY
    assert elapsed < 0.6  # 0.3 s reply cap + overhead


def test_reply_only_gets_what_is_left_of_the_total_budget(monkeypatch):
    """Intent uses most of the budget; the reply is cut off by the TOTAL
    budget even though it is under its own 3 s cap."""
    monkeypatch.setattr(settings, "request_budget_seconds", 0.4)
    monkeypatch.setattr(settings, "intent_timeout_seconds", 0.35)
    monkeypatch.setattr(settings, "llm_timeout_seconds", 3.0)
    with patch(
        "app.ai.chatbot.intent.send_chat_message",
        side_effect=_slow(0.25, '{"intent": "general_chat", "confidence": 0.9}'),
    ), patch(
        "app.services.chatbot_service.send_chat_message",
        side_effect=_slow(0.5, "too late"),
    ):
        response, elapsed = _run(NO_REGEX_MESSAGE)
    assert response.confidence == 0.9  # intent finished inside its cap
    assert response.reply == TIMEOUT_REPLY
    assert elapsed < 0.4 + 0.2


def test_spent_budget_skips_the_reply_call(monkeypatch):
    monkeypatch.setattr(settings, "request_budget_seconds", 0.1)
    monkeypatch.setattr(settings, "intent_timeout_seconds", 1.0)
    with patch(
        "app.ai.chatbot.intent.send_chat_message",
        side_effect=_slow(0.5, '{"intent": "general_chat", "confidence": 0.9}'),
    ), patch(
        "app.services.chatbot_service.send_chat_message",
        side_effect=_slow(0.5, "too late"),
    ):
        response, elapsed = _run(NO_REGEX_MESSAGE)
    assert response.intent == "general_chat"
    assert response.confidence == 0.0
    assert response.reply == TIMEOUT_REPLY
    assert elapsed < 0.1 + 0.2


def test_groq_client_has_timeout_and_no_retries(monkeypatch):
    monkeypatch.setattr(llm_client, "_groq_client", None)
    monkeypatch.setattr(settings, "groq_api_key", SecretStr("fake-test-key"))
    with patch("app.ai.chatbot.llm_client.Groq") as groq_cls:
        llm_client._get_groq_client()
    kwargs = groq_cls.call_args.kwargs
    assert kwargs["timeout"] == settings.llm_timeout_seconds
    assert kwargs["max_retries"] == 0
    monkeypatch.setattr(llm_client, "_groq_client", None)
