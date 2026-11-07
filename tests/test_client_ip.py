"""
Proxy-aware client IP (TRUST_PROXY / TRUST_PROXY_HOPS) and the per-IP limits:
demo-token issuance and the chat IP cap (CHAT_IP_RATE_PER_MINUTE).
"""

import logging
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.ai.chatbot.intent import IntentResult
from app.core.client_ip import client_ip
from app.core.config import settings
from app.main import app
from tests.support import auth_headers, make_token

pytestmark = pytest.mark.db

client = TestClient(app)
CHAT = "/api/v1/ai/chat"
DEMO = "/api/v1/auth/demo-token"


def _request(headers=(), peer=("10.0.0.9", 1234)) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers]
    return Request({"type": "http", "headers": raw, "client": peer})


def _mocks():
    return (
        patch("app.services.chatbot_service.send_chat_message", return_value="Happy to help!"),
        patch(
            "app.services.chatbot_service.classify_intent",
            return_value=IntentResult(intent="general_chat", confidence=1.0),
        ),
    )


# --------------------------------------------------------------------------
# IP selection
# --------------------------------------------------------------------------

def test_trust_proxy_off_ignores_forwarded_for(monkeypatch):
    monkeypatch.setattr(settings, "trust_proxy", False)
    req = _request([("X-Forwarded-For", "203.0.113.7")])
    assert client_ip(req) == "10.0.0.9"


@pytest.mark.parametrize(
    "header, hops, expected",
    [
        ("203.0.113.7", 1, "203.0.113.7"),
        ("1.1.1.1, 203.0.113.7", 1, "203.0.113.7"),
        ("1.1.1.1, 203.0.113.7, 198.51.100.2", 2, "203.0.113.7"),
        (" 1.1.1.1 ,, 203.0.113.7 , ", 1, "203.0.113.7"),
    ],
)
def test_trust_proxy_uses_entry_hops_from_the_right(monkeypatch, header, hops, expected):
    monkeypatch.setattr(settings, "trust_proxy", True)
    monkeypatch.setattr(settings, "trust_proxy_hops", hops)
    assert client_ip(_request([("X-Forwarded-For", header)])) == expected


def test_several_forwarded_for_headers_are_joined(monkeypatch):
    monkeypatch.setattr(settings, "trust_proxy", True)
    monkeypatch.setattr(settings, "trust_proxy_hops", 1)
    req = _request([("X-Forwarded-For", "1.1.1.1"), ("X-Forwarded-For", "203.0.113.7")])
    assert client_ip(req) == "203.0.113.7"


@pytest.mark.parametrize("header", [None, "", " , ", "203.0.113.7"])
def test_too_few_entries_falls_back_to_peer(monkeypatch, header):
    monkeypatch.setattr(settings, "trust_proxy", True)
    monkeypatch.setattr(settings, "trust_proxy_hops", 2)
    headers = [("X-Forwarded-For", header)] if header is not None else []
    assert client_ip(_request(headers)) == "10.0.0.9"


def test_missing_peer_is_unknown(monkeypatch):
    monkeypatch.setattr(settings, "trust_proxy", False)
    assert client_ip(_request(peer=None)) == "unknown"


def test_trust_proxy_hops_must_be_positive():
    from pydantic import ValidationError

    from app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(trust_proxy_hops=0)


# --------------------------------------------------------------------------
# Demo-token limit keyed on the proxy-aware IP
# --------------------------------------------------------------------------

def test_demo_token_limit_ignores_spoofed_left_entries(monkeypatch):
    monkeypatch.setattr(settings, "demo_tokens_enabled", True)
    monkeypatch.setattr(settings, "trust_proxy", True)
    monkeypatch.setattr(settings, "trust_proxy_hops", 1)
    for i in range(settings.demo_token_rate_per_minute):
        headers = {"X-Forwarded-For": f"198.51.100.{i}, 203.0.113.7"}
        assert client.post(DEMO, headers=headers).status_code == 200
    spoofed = {"X-Forwarded-For": "198.51.100.99, 203.0.113.7"}
    assert client.post(DEMO, headers=spoofed).status_code == 429
    # A different real client IP (right-most entry) has its own budget.
    assert client.post(DEMO, headers={"X-Forwarded-For": "203.0.113.8"}).status_code == 200


def test_demo_token_limit_without_trust_proxy_uses_peer(monkeypatch):
    monkeypatch.setattr(settings, "demo_tokens_enabled", True)
    monkeypatch.setattr(settings, "trust_proxy", False)
    for i in range(settings.demo_token_rate_per_minute):
        headers = {"X-Forwarded-For": f"203.0.113.{i}"}
        assert client.post(DEMO, headers=headers).status_code == 200
    assert client.post(DEMO, headers={"X-Forwarded-For": "203.0.113.200"}).status_code == 429


# --------------------------------------------------------------------------
# Chat per-IP cap
# --------------------------------------------------------------------------

def test_chat_ip_cap_is_shared_by_subjects_from_one_ip(monkeypatch, caplog):
    monkeypatch.setattr(settings, "chat_ip_rate_per_minute", 20)
    monkeypatch.setattr(settings, "trust_proxy", True)
    monkeypatch.setattr(settings, "trust_proxy_hops", 1)
    xff = {"X-Forwarded-For": "203.0.113.7"}
    llm, intent = _mocks()
    with llm, intent:
        for i in range(20):
            sub = "ip-user-a" if i < 11 else "ip-user-b"
            headers = {**auth_headers(sub), **xff}
            assert client.post(CHAT, json={"message": "hi"}, headers=headers).status_code == 200
        token = make_token("ip-user-c")
        with caplog.at_level(logging.WARNING, logger="app.middleware.rate_limit"):
            blocked = client.post(
                CHAT, json={"message": "hi"}, headers={"Authorization": f"Bearer {token}", **xff}
            )
    assert blocked.status_code == 429
    assert blocked.json() == {
        "error": "Rate limit exceeded",
        "detail": "20 requests per minute allowed per client IP",
        "status_code": 429,
    }
    assert "Rate limit exceeded for client IP 203.0.113.7" in caplog.text
    assert token not in caplog.text


def test_chat_ip_cap_keyed_on_right_most_entry(monkeypatch):
    monkeypatch.setattr(settings, "chat_ip_rate_per_minute", 3)
    monkeypatch.setattr(settings, "trust_proxy", True)
    monkeypatch.setattr(settings, "trust_proxy_hops", 1)
    llm, intent = _mocks()
    with llm, intent:
        for i in range(3):
            headers = {**auth_headers(f"xff-user-{i}"), "X-Forwarded-For": f"198.51.100.{i}, 203.0.113.7"}
            assert client.post(CHAT, json={"message": "hi"}, headers=headers).status_code == 200
        spoofed = {**auth_headers("xff-user-9"), "X-Forwarded-For": "198.51.100.50, 203.0.113.7"}
        assert client.post(CHAT, json={"message": "hi"}, headers=spoofed).status_code == 429
        other = {**auth_headers("xff-user-10"), "X-Forwarded-For": "203.0.113.8"}
        assert client.post(CHAT, json={"message": "hi"}, headers=other).status_code == 200


def test_per_user_limit_is_checked_before_the_ip_cap(monkeypatch):
    monkeypatch.setattr(settings, "chat_ip_rate_per_minute", 1000)
    from app.middleware.rate_limit import RATE_LIMIT_LIMIT

    llm, intent = _mocks()
    with llm, intent:
        for _ in range(RATE_LIMIT_LIMIT):
            assert client.post(CHAT, json={"message": "hi"}, headers=auth_headers("solo")).status_code == 200
        blocked = client.post(CHAT, json={"message": "hi"}, headers=auth_headers("solo"))
    assert blocked.status_code == 429
    assert blocked.json()["detail"] == f"{RATE_LIMIT_LIMIT} requests per minute allowed"


def test_invalid_messages_still_count_toward_the_limits(monkeypatch):
    """Empty messages (422/400, no LLM call) are counted: the smoke check relies on it."""
    monkeypatch.setattr(settings, "chat_ip_rate_per_minute", 1000)
    from app.middleware.rate_limit import RATE_LIMIT_LIMIT

    statuses = [
        client.post(CHAT, json={"message": ""}, headers=auth_headers("empty-sender")).status_code
        for _ in range(RATE_LIMIT_LIMIT + 1)
    ]
    assert statuses[-1] == 429
    assert set(statuses[:-1]) == {statuses[0]}
    assert statuses[0] in (400, 422)


def test_no_token_is_rejected_before_any_counting(monkeypatch):
    monkeypatch.setattr(settings, "chat_ip_rate_per_minute", 1)
    for _ in range(3):
        assert client.post(CHAT, json={"message": "hi"}).status_code == 401
    llm, intent = _mocks()
    with llm, intent:
        assert client.post(CHAT, json={"message": "hi"}, headers=auth_headers("first")).status_code == 200
