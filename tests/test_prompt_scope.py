"""
The general scope rule (answer only about RRVDXB; decline other companies,
travel, transport and unrelated topics) reaches the LLM on every reply path.

This checks that the rule is in the prompt. Whether the live model follows it
is not measured here (the LLM is mocked).
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.ai.chatbot.intent import IntentResult
from app.ai.chatbot.price_guard import GUARD_REPLY, extract_price_figures
from app.ai.chatbot.prompts import (
    INTENT_CLASSIFICATION_PROMPT,
    SCOPE_RULE,
    SHOWN_DEAL_NOTE,
    SHOWN_PRODUCTS_NOTE,
    SYSTEM_PROMPT,
    build_rag_system_prompt,
)
from app.main import app
from tests.support import auth_headers

pytestmark = pytest.mark.db

client = TestClient(app)

_FAQ = [{"question": "Do you ship to Saudi Arabia (KSA)?", "answer": "Yes, 3-5 business days.", "similarity": 0.65}]
_REC = [{"id": 1, "name": "iPhone 14 Pro Max", "price": 4699.0, "currency": "AED",
         "category": "Electronics", "brand": "Apple", "reason": "Flagship"}]
_DEAL = {"title": "Summer Sale", "discount": "20% off", "code": "SUMMER20"}

PATHS = [
    ("product_faq", {"retrieve_faq_context": _FAQ}),
    ("product_faq", {"retrieve_faq_context": []}),
    ("recommend_product", {"get_recommendations": _REC}),
    ("recommend_product", {"get_recommendations": []}),
    ("deal_inquiry", {"get_deals": _DEAL}),
    ("deal_inquiry", {"get_deals": None}),
    ("track_order_help", {}),
    ("general_chat", {}),
]


def test_scope_rule_is_general_and_in_the_base_prompt():
    assert SCOPE_RULE in SYSTEM_PROMPT
    for phrase in ("RRVDXB's own store, orders and products", "another company", "travel", "transport"):
        assert phrase in SCOPE_RULE
    assert SCOPE_RULE in build_rag_system_prompt(_FAQ)
    assert "SCOPE RULE" in build_rag_system_prompt(_FAQ).split("RAG INSTRUCTIONS:")[1]


def test_scope_rule_does_not_change_the_intent_classifier_prompt():
    assert SCOPE_RULE not in INTENT_CLASSIFICATION_PROMPT


@pytest.mark.parametrize("intent, stubs", PATHS, ids=[
    "faq-match", "faq-no-match", "recommend", "recommend-none", "deal", "deal-none", "track", "general",
])
def test_scope_rule_reaches_the_llm_on_every_path(intent, stubs):
    patches = [
        patch("app.services.chatbot_service.classify_intent",
              return_value=IntentResult(intent=intent, confidence=1.0)),
    ] + [patch(f"app.services.chatbot_service.{name}", return_value=value) for name, value in stubs.items()]
    with patch("app.services.chatbot_service.send_chat_message", return_value="Happy to help!") as llm:
        for p in patches:
            p.start()
        try:
            response = client.post(
                "/api/v1/ai/chat", json={"message": "Does another shop deliver to Riyadh?"},
                headers=auth_headers(f"scope-{intent}"),
            )
        finally:
            for p in patches:
                p.stop()
    assert response.status_code == 200
    kwargs = llm.call_args.kwargs
    effective_prompt = kwargs["system_prompt_override"] or kwargs["system_prompt"]
    assert SCOPE_RULE in effective_prompt
    # Products and the deal are shown by the page as structured fields: the
    # model is told not to repeat them, and only on the paths that return them.
    assert (SHOWN_PRODUCTS_NOTE in effective_prompt) == (stubs.get("get_recommendations") == _REC)
    assert (SHOWN_DEAL_NOTE in effective_prompt) == (stubs.get("get_deals") == _DEAL)


def test_reply_format_is_plain_text_without_follow_up_promises():
    assert "plain text only" in SYSTEM_PROMPT
    assert "no Markdown" in SYSTEM_PROMPT
    assert "Use bullet points" not in SYSTEM_PROMPT
    # No promise of a later check that the demo can never deliver.
    assert "Let me check that for you" not in SYSTEM_PROMPT
    assert "One moment" not in GUARD_REPLY
    assert extract_price_figures(GUARD_REPLY) == set()


def test_prompt_does_not_ask_for_currency_conversion():
    assert "AED for UAE/Saudi" not in SYSTEM_PROMPT
    assert "Never convert them to another currency" in SYSTEM_PROMPT
