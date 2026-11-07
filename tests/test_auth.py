"""
Bearer-token authentication, demo tokens and the startup secret guard.
"""

from unittest.mock import patch

import jwt
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.ai.chatbot.intent import IntentResult
from app.core import security
from app.core.config import Settings, settings
from app.main import app
from app.middleware.rate_limit import RATE_LIMIT_LIMIT
from app.models.chat_history import ChatHistory
from tests.support import TestingSessionLocal, auth_headers, make_token

pytestmark = pytest.mark.db

client = TestClient(app)
CHAT ="/api/v1/ai/chat"
DEMO = "/api/v1/auth/demo-token"
INVALID = {
    "error": "Authentication required",
    "detail": "Invalid or expired token",
    "status_code": 401,
}


def _mocks(reply: str = "Happy to help!"):
    """Patch the LLM reply and intent so no network call is made."""
    return (
        patch("app.services.chatbot_service.send_chat_message", return_value=reply),
        patch(
            "app.services.chatbot_service.classify_intent",
            return_value=IntentResult(intent="general_chat", confidence=1.0),
        ),
    )


def _post_with_token(token: str):
    llm, intent = _mocks()
    with llm, intent:
        return client.post(
            CHAT, json={"message": "hello"}, headers={"Authorization": f"Bearer {token}"}
        )


# --------------------------------------------------------------------------
# Rejections: every one is a 401 with the same fixed body
# --------------------------------------------------------------------------

def test_no_token_401():
    response = client.post(CHAT, json={"message": "hello"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Bearer token missing"


def test_non_bearer_scheme_401():
    response = client.post(
        CHAT, json={"message": "hello"}, headers={"Authorization": "Basic abc"}
    )
    assert response.status_code == 401


def test_malformed_token_401():
    response = _post_with_token("not-a-jwt")
    assert response.status_code == 401
    assert response.json() == INVALID


def test_bad_signature_401():
    response = _post_with_token(make_token("u1", secret="another-secret-" + "y" * 25))
    assert response.status_code == 401
    assert response.json() == INVALID


def test_expired_token_401():
    response = _post_with_token(make_token("u1", exp_delta=-10))
    assert response.status_code == 401
    assert response.json() == INVALID


@pytest.mark.parametrize("claim", ["iss", "aud"])
def test_wrong_issuer_or_audience_401(claim):
    response = _post_with_token(make_token("u1", **{claim: "someone-else"}))
    assert response.status_code == 401


@pytest.mark.parametrize("claim", ["sub", "exp", "iat", "iss", "aud"])
def test_missing_required_claim_401(claim):
    kwargs = {claim: None} if claim != "sub" else {}
    token = make_token(None if claim == "sub" else "u1", **kwargs)
    response = _post_with_token(token)
    assert response.status_code == 401


def test_alg_none_401():
    token = jwt.encode(
        {"sub": "u1", "iat": 1, "exp": 9999999999, "iss": settings.jwt_issuer,
         "aud": settings.jwt_audience},
        key=None,
        algorithm="none",
    )
    assert _post_with_token(token).status_code == 401


def test_hs512_token_rejected():
    """Only HS256 is accepted, even with the right secret."""
    assert _post_with_token(make_token("u1", algorithm="HS512")).status_code == 401


def test_malformed_subject_401():
    assert _post_with_token(make_token("bad sub with spaces")).status_code == 401


def test_x_user_id_header_alone_is_not_trusted():
    response = client.post(CHAT, json={"message": "hello"}, headers={"X-User-Id": "1"})
    assert response.status_code == 401


def test_x_user_id_header_is_ignored_when_token_present():
    """The token subject is the identity; a spoofed header changes nothing."""
    llm, intent = _mocks()
    with llm, intent:
        headers = {**auth_headers("tok-user"), "X-User-Id": "victim"}
        assert client.post(CHAT, json={"message": "hi there"}, headers=headers).status_code == 200
    db = TestingSessionLocal()
    try:
        assert db.query(ChatHistory).filter(ChatHistory.user_id == "tok-user").count() == 1
        assert db.query(ChatHistory).filter(ChatHistory.user_id == "victim").count() == 0
    finally:
        db.close()


# --------------------------------------------------------------------------
# Isolation and rate limiting on the verified subject
# --------------------------------------------------------------------------

def test_two_users_never_see_each_others_memory():
    secret_text = "my private gift plan PRIV-A-123"
    with patch(
        "app.services.chatbot_service.send_chat_message", return_value="Noted!"
    ), patch(
        "app.services.chatbot_service.classify_intent",
        return_value=IntentResult(intent="general_chat", confidence=1.0),
    ):
        assert client.post(
            CHAT, json={"message": secret_text}, headers=auth_headers("alice-1")
        ).status_code == 200

    with patch(
        "app.services.chatbot_service.send_chat_message", return_value="Hi!"
    ) as llm_b, patch(
        "app.services.chatbot_service.classify_intent",
        return_value=IntentResult(intent="general_chat", confidence=1.0),
    ):
        assert client.post(
            CHAT, json={"message": "hello"}, headers=auth_headers("bob-1")
        ).status_code == 200
        assert secret_text not in llm_b.call_args.kwargs["history"]
        assert "Noted!" not in llm_b.call_args.kwargs["history"]

    with patch(
        "app.services.chatbot_service.send_chat_message", return_value="Sure"
    ) as llm_a, patch(
        "app.services.chatbot_service.classify_intent",
        return_value=IntentResult(intent="general_chat", confidence=1.0),
    ):
        client.post(CHAT, json={"message": "again"}, headers=auth_headers("alice-1"))
        assert secret_text in llm_a.call_args.kwargs["history"]


def test_rate_limit_is_keyed_on_token_subject():
    llm, intent = _mocks()
    with llm, intent:
        for _ in range(RATE_LIMIT_LIMIT):
            assert client.post(
                CHAT, json={"message": "hi"}, headers=auth_headers("busy-user")
            ).status_code == 200
        # A new token for the SAME subject shares the same budget.
        assert client.post(
            CHAT, json={"message": "hi"}, headers=auth_headers("busy-user")
        ).status_code == 429
        # Changing X-User-Id does not help.
        spoof = {**auth_headers("busy-user"), "X-User-Id": "someone-else"}
        assert client.post(CHAT, json={"message": "hi"}, headers=spoof).status_code == 429
        # A different verified subject is unaffected.
        assert client.post(
            CHAT, json={"message": "hi"}, headers=auth_headers("calm-user")
        ).status_code == 200


# --------------------------------------------------------------------------
# Demo tokens (opt-in)
# --------------------------------------------------------------------------

def test_demo_token_disabled_by_default_404():
    response = client.post(DEMO)
    assert response.status_code == 404
    assert response.json() == {"error": "Not Found", "detail": "Not Found", "status_code": 404}


def test_demo_token_enabled_issues_working_token(monkeypatch):
    monkeypatch.setattr(settings, "demo_tokens_enabled", True)
    response = client.post(DEMO)
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == settings.demo_token_ttl_seconds
    claims = jwt.decode(
        body["access_token"],
        settings.internal_jwt_secret.get_secret_value(),
        algorithms=["HS256"],
        audience=settings.jwt_audience,
        issuer=settings.jwt_issuer,
    )
    assert claims["sub"].startswith("demo-")
    assert claims["exp"] - claims["iat"] == settings.demo_token_ttl_seconds

    second = client.post(DEMO).json()["access_token"]
    second_sub = jwt.decode(second, options={"verify_signature": False})["sub"]
    assert second_sub != claims["sub"]  # every visitor gets their own subject

    assert _post_with_token(body["access_token"]).status_code == 200


def test_demo_token_rate_limited_per_ip(monkeypatch):
    monkeypatch.setattr(settings, "demo_tokens_enabled", True)
    for _ in range(settings.demo_token_rate_per_minute):
        assert client.post(DEMO).status_code == 200
    blocked = client.post(DEMO)
    assert blocked.status_code == 429
    assert blocked.json()["error"] == "Rate limit exceeded"


# --------------------------------------------------------------------------
# Startup guard
# --------------------------------------------------------------------------

@pytest.mark.parametrize("env", ["production", "development"])
def test_short_secret_refuses_to_start(env):
    short = "tooshort-abc123"
    cfg = Settings(app_env=env, internal_jwt_secret=SecretStr(short))
    with pytest.raises(RuntimeError) as excinfo:
        security.validate_security_settings(cfg)
    assert "at least 32 characters" in str(excinfo.value)
    assert short not in str(excinfo.value)


def test_empty_secret_refuses_to_start_in_production():
    cfg = Settings(app_env="production", internal_jwt_secret=SecretStr(""))
    with pytest.raises(RuntimeError):
        security.validate_security_settings(cfg)


def test_long_secret_starts_in_production():
    cfg = Settings(app_env="production", internal_jwt_secret=SecretStr("z" * 32))
    security.validate_security_settings(cfg)  # no exception


def test_test_mode_skips_length_check():
    cfg = Settings(app_env="test", internal_jwt_secret=SecretStr(""))
    security.validate_security_settings(cfg)  # no exception


def test_create_application_runs_the_guard(monkeypatch):
    from app.main import create_application

    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "internal_jwt_secret", SecretStr("short"))
    with pytest.raises(RuntimeError):
        create_application()
