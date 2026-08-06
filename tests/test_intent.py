"""
pytest suite for intent recognition (Day 4).

Covers:
  - Regex fast-path classification (zero LLM calls).
  - LLM fallback classification with mocked send_chat_message.
  - JSON parsing robustness (code fences, malformed JSON, bad intents).
  - Confidence threshold override and clamping.
  - Graceful degradation when the Groq API fails.

All LLM calls are mocked — no real API, no network, deterministic.
"""

from unittest.mock import patch

import pytest

from app.ai.chatbot.intent import IntentResult, classify_intent

# A message that matches NO regex pattern, so it always reaches the LLM path.
_AMBIGUOUS = "I bought headphones last week and I have a concern about them"


# --------------------------------------------------------------------------
# Regex fast-path
# --------------------------------------------------------------------------

def test_regex_recommend_product_no_llm():
    result = classify_intent("recommend me a gift for my mom")
    assert isinstance(result, IntentResult)
    assert result.intent == "recommend_product"
    assert result.confidence == 1.0


def test_regex_track_order_help():
    result = classify_intent("Can you track my order please?")
    assert result.intent == "track_order_help"
    assert result.confidence == 1.0


def test_regex_deal_inquiry():
    result = classify_intent("Do you have any discount codes?")
    assert result.intent == "deal_inquiry"


def test_regex_product_faq():
    result = classify_intent("What's your return policy?")
    assert result.intent == "product_faq"


def test_regex_fast_path_never_calls_llm():
    """
    The whole point of the regex layer: obvious patterns must NOT burn an
    LLM call. We assert the mocked LLM function is never invoked across
    multiple obvious queries.
    """
    with patch("app.ai.chatbot.intent.send_chat_message") as mock_llm:
        classify_intent("recommend me a gift")
        classify_intent("track my order")
        classify_intent("do you have any discount codes")
        mock_llm.assert_not_called()


def test_gift_return_query_routes_to_product_faq_not_recommend():
    """
    Regression: 'gift' must not hijack FAQ queries. The word gift appears
    in a return-policy question, so this must NOT route to recommend_product.
    """
    result = classify_intent("i want to know the return policy for a gift i received")
    assert result.intent == "product_faq"
    assert result.confidence == 1.0


def test_bare_buy_matches_recommend():
    """Regression: a bare buying question routes to recommend_product."""
    result = classify_intent("what should I buy for Eid?")
    assert result.intent == "recommend_product"
    assert result.confidence == 1.0


# --------------------------------------------------------------------------
# LLM fallback (mocked)
# --------------------------------------------------------------------------

def test_nonsense_message_maps_to_general_chat():
    """'blah blah nonsense' -> LLM says general_chat, low confidence kept."""
    with patch(
        "app.ai.chatbot.intent.send_chat_message",
        return_value='{"intent": "general_chat", "confidence": 0.35, "entities": {}}',
    ):
        result = classify_intent("blah blah nonsense")
    assert result.intent == "general_chat"
    assert result.confidence == pytest.approx(0.35)


def test_low_confidence_intent_overridden_to_general_chat():
    """
    LLM labels a specific intent but is unsure (< 0.7):
    we override to general_chat while KEEPING the original confidence.
    """
    with patch(
        "app.ai.chatbot.intent.send_chat_message",
        return_value='{"intent": "deal_inquiry", "confidence": 0.5, "entities": {}}',
    ):
        result = classify_intent(_AMBIGUOUS)
    assert result.intent == "general_chat"
    assert result.confidence == pytest.approx(0.5)


def test_high_confidence_intent_is_respected():
    with patch(
        "app.ai.chatbot.intent.send_chat_message",
        return_value='{"intent": "product_faq", "confidence": 0.9, "entities": {}}',
    ):
        result = classify_intent(_AMBIGUOUS)
    assert result.intent == "product_faq"
    assert result.confidence == pytest.approx(0.9)


def test_confidence_is_clamped_to_1_0():
    """A model returning confidence above 1.0 (e.g. 9.5) is clamped to 1.0."""
    with patch(
        "app.ai.chatbot.intent.send_chat_message",
        return_value='{"intent": "deal_inquiry", "confidence": 9.5, "entities": {}}',
    ):
        result = classify_intent(_AMBIGUOUS)
    assert result.confidence == 1.0
    assert result.intent == "deal_inquiry"


def test_markdown_code_fences_are_stripped():
    fenced = (
        '```json\n{"intent": "recommend_product", '
        '"confidence": 0.9, "entities": {}}\n```'
    )
    with patch(
        "app.ai.chatbot.intent.send_chat_message",
        return_value=fenced,
    ):
        result = classify_intent(_AMBIGUOUS)
    assert result.intent == "recommend_product"
    assert result.confidence == pytest.approx(0.9)


def test_malformed_json_falls_back_gracefully():
    with patch(
        "app.ai.chatbot.intent.send_chat_message",
        return_value="I am not sure what that means",
    ):
        result = classify_intent(_AMBIGUOUS)
    assert result.intent == "general_chat"
    assert result.confidence == 0.0


def test_unknown_intent_falls_back():
    with patch(
        "app.ai.chatbot.intent.send_chat_message",
        return_value='{"intent": "buy_stuff", "confidence": 0.9, "entities": {}}',
    ):
        result = classify_intent(_AMBIGUOUS)
    assert result.intent == "general_chat"
    assert result.confidence == 0.0


def test_groq_api_failure_falls_back_to_general_chat():
    """Simulate a Groq API failure (timeout / rate-limit / auth)."""
    with patch(
        "app.ai.chatbot.intent.send_chat_message",
        side_effect=RuntimeError("Simulated Groq outage"),
    ):
        result = classify_intent(_AMBIGUOUS)
    assert result.intent == "general_chat"
    assert result.confidence == 0.0