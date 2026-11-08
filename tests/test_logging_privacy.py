"""
Logging privacy: no raw user messages, API-key characters or tokens in logs.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.ai.chatbot import llm_client
from app.ai.chatbot.intent import IntentResult
from app.ai.chatbot.memory import save_turn
from app.core.config import Settings, settings
from app.core.database import engine as app_engine
from app.core.database import make_engine
from app.db.migrate import upgrade_to_head
from app.main import app
from sqlalchemy.orm import Session
from tests.support import auth_headers, make_token

pytestmark = pytest.mark.db

client = TestClient(app)
CHAT = "/api/v1/ai/chat"
MARKER = "PRIVMARKER-7f3a9c"
FAKE_KEY = "gsk_FAKEKEYCHARS0123456789"

_FAQ = [{"question": "Q?", "answer": "A.", "similarity": 0.9}]
_REC = [{"id": 1, "name": "iPhone 14 Pro Max", "price": 4699.0, "currency": "AED",
         "category": "Electronics", "brand": "Apple", "reason": "match"}]
PATHS = [
    ("product_faq", {"retrieve_faq_context": _FAQ}),
    ("product_faq", {"retrieve_faq_context": []}),
    ("recommend_product", {"get_recommendations": _REC}),
    ("recommend_product", {"get_recommendations": []}),
    ("deal_inquiry", {"get_deals": {"title": "S", "discount": "1%", "code": "C"}}),
    ("deal_inquiry", {"get_deals": None}),
    ("track_order_help", {}),
    ("general_chat", {}),
]


@pytest.mark.parametrize("intent, extras", PATHS)
def test_user_message_is_not_logged_on_any_path(caplog, intent, extras):
    caplog.set_level(logging.DEBUG)
    patches = [
        patch("app.services.chatbot_service.send_chat_message", return_value="Sure thing."),
        patch(
            "app.services.chatbot_service.classify_intent",
            return_value=IntentResult(intent=intent, confidence=1.0),
        ),
    ] + [patch(f"app.services.chatbot_service.{k}", return_value=v) for k, v in extras.items()]
    for p in patches:
        p.start()
    try:
        response = client.post(
            CHAT, json={"message": f"{MARKER} my private request"}, headers=auth_headers("priv-1")
        )
    finally:
        for p in reversed(patches):
            p.stop()
    assert response.status_code == 200
    assert caplog.records, "expected the chat flow to log something"
    assert MARKER not in caplog.text


@pytest.mark.parametrize(
    "llm_output",
    [
        f'["{MARKER} echoed"]',                                   # non-object JSON
        f'{{"intent": "{MARKER}", "confidence": 0.9}}',          # unknown intent
        f"not json {MARKER}",                                     # malformed
    ],
)
def test_intent_classifier_output_is_not_logged(caplog, llm_output):
    caplog.set_level(logging.DEBUG)
    with patch("app.ai.chatbot.intent.send_chat_message", return_value=llm_output), patch(
        "app.services.chatbot_service.send_chat_message", return_value="ok"
    ):
        response = client.post(
            CHAT, json={"message": "hmm something nice maybe"}, headers=auth_headers("priv-2")
        )
    assert response.status_code == 200
    assert MARKER not in caplog.text


def _failing_groq():
    fake = MagicMock()
    fake.chat.completions.create.side_effect = Exception(f"Invalid API Key: {FAKE_KEY}")
    return fake


def test_groq_error_text_with_key_characters_is_not_logged(caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(settings, "groq_api_key", SecretStr(FAKE_KEY))
    monkeypatch.setattr(llm_client, "_groq_client", None)
    with patch("app.ai.chatbot.llm_client.Groq", return_value=_failing_groq()), patch(
        "app.services.chatbot_service.classify_intent",
        return_value=IntentResult(intent="general_chat", confidence=1.0),
    ):
        response = client.post(CHAT, json={"message": "hello"}, headers=auth_headers("priv-3"))
    monkeypatch.setattr(llm_client, "_groq_client", None)
    assert response.status_code == 200
    assert "RRVDXB shopping assistant" in response.json()["reply"]
    assert "FAKEKEYCHARS" not in caplog.text
    assert "Groq API call failed (Exception, status=None)" in caplog.text


@pytest.mark.parametrize("token", ["not-a-jwt-PRIVTOKEN", make_token("u1", exp_delta=-10)])
def test_rejected_tokens_are_not_logged(caplog, token):
    caplog.set_level(logging.DEBUG)
    response = client.post(
        CHAT, json={"message": "hello"}, headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 401
    assert token not in caplog.text
    assert token not in response.text


def test_settings_repr_hides_secrets():
    cfg = Settings(groq_api_key=SecretStr(FAKE_KEY), internal_jwt_secret=SecretStr("s" * 40))
    text = repr(cfg) + str(cfg) + str(cfg.model_dump())
    assert FAKE_KEY not in text
    assert "s" * 40 not in text


def test_app_engine_does_not_echo_sql():
    assert app_engine.echo is False


def test_sql_parameters_hidden_even_with_engine_logging_on(caplog, tmp_path):
    """hide_parameters=True: enabling sqlalchemy.engine INFO logs statements
    but never the bound user message."""
    engine = make_engine(f"sqlite:///{(tmp_path / 'p.db').as_posix()}")
    try:
        with engine.begin() as conn:
            upgrade_to_head(connection=conn)
        caplog.set_level(logging.INFO, logger="sqlalchemy.engine")
        with Session(engine) as db:
            save_turn(db, "priv-sql", f"{MARKER} in a message", "reply")
        assert "INSERT INTO chat_history" in caplog.text  # logging really was on
        assert MARKER not in caplog.text
    finally:
        engine.dispose()
