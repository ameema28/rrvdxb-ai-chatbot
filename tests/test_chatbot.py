"""
pytest suite for the chatbot endpoint and conversation memory.

Day 8 adds auth + rate-limiting + error-handling coverage:
  - Missing X-User-Id            -> 401 with the canonical error body
  - Invalid X-User-Id ("abc")    -> 400 with the canonical error body
  - Per-user rate limit exceeded -> 429 with the canonical error body
  - Unhandled internal error     -> 500 generic body, no traceback leaked

Day 9 adds:
  - Cross-user memory isolation (no history leakage between users)
  - track_order_help branch (ORDER TRACKING GUIDANCE prompt, history kept)
  - general_chat branch (standard prompt + catalog context, made explicit)
  - Framework-level error shape: 404 and 405 via http_exception_handler
  - Malformed request body -> canonical 400 (validation_error_handler)
  - Hallucination guard: an ungrounded price figure in the LLM reply is
    replaced with a safety message (and only the safe text is persisted),
    while a figure that IS grounded in the injected context is kept.

Everything that existed before (memory, RAG, recommendations, deals,
timeout/fallback, intent persistence) still passes unchanged.
"""

from unittest.mock import patch

import asyncio

import pytest

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.ai.chatbot.intent import IntentResult
from app.core.database import Base
from app.core.dependencies import get_db
from app.middleware.rate_limit import RATE_LIMIT_LIMIT, rate_limit_store

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


@pytest.fixture(autouse=True)
def reset_rate_limit_store():
    """Give every test a clean per-user budget (module-level store singleton)."""
    rate_limit_store.reset()
    yield
    rate_limit_store.reset()


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
# Day 8 — Auth, rate limiting, centralized error handling
# ==========================================================================

def test_chat_endpoint_requires_user_id_header():
    """Missing X-User-Id -> 401 with the centralized error body."""
    payload = {"message": "Test without auth"}
    response = client.post("/api/v1/ai/chat", json=payload)
    assert response.status_code == 401
    data = response.json()
    assert data == {
        "error": "Authentication required",
        "detail": "X-User-Id header missing",
        "status_code": 401,
    }


def test_invalid_user_id_returns_400_standardized_body():
    """Non-integer X-User-Id ('abc') -> 400 with the centralized body."""
    payload = {"message": "Test with a bad user id"}
    response = client.post(
        "/api/v1/ai/chat", json=payload, headers={"X-User-Id": "abc"}
    )
    assert response.status_code == 400
    data = response.json()
    assert data == {
        "error": "Validation error",
        "detail": "X-User-Id must be an integer",
        "status_code": 400,
    }


def test_rate_limit_exceeded_returns_429_standardized_body():
    """The (limit+1)th request from one user -> 429 with the centralized body."""
    user_id = 777
    payload = {"message": "rate limited request"}
    mock_reply = "Here is a canned answer."

    with patch(
        "app.services.chatbot_service.send_chat_message",
        return_value=mock_reply,
    ), _mock_intent("general_chat"):
        for _ in range(RATE_LIMIT_LIMIT):
            ok = client.post(
                "/api/v1/ai/chat",
                json=payload,
                headers={"X-User-Id": str(user_id)},
            )
            assert ok.status_code == 200, f"Expected 200, got {ok.status_code}"

        blocked = client.post(
            "/api/v1/ai/chat",
            json=payload,
            headers={"X-User-Id": str(user_id)},
        )

    assert blocked.status_code == 429
    data = blocked.json()
    assert data == {
        "error": "Rate limit exceeded",
        "detail": "20 requests per minute allowed",
        "status_code": 429,
    }
    # A DIFFERENT user is not affected (per-user budgets are independent).
    with patch(
        "app.services.chatbot_service.send_chat_message",
        return_value=mock_reply,
    ), _mock_intent("general_chat"):
        other = client.post(
            "/api/v1/ai/chat", json=payload, headers={"X-User-Id": "778"}
        )
    assert other.status_code == 200


def test_unhandled_error_returns_500_standardized_body():
    """
    A bug that escapes EVERYTHING -> clean 500 body, no traceback/leak.

    The UnhandledErrorMiddleware (not TestClient raise_server_exceptions
    flags) guarantees this: it catches the ValueError, logs the traceback
    server-side, and returns the canonical JSON body even with DEBUG=True,
    so ServerErrorMiddleware never re-raises or leaks a traceback.
    """
    with patch(
        "app.services.chatbot_service.send_chat_message",
        side_effect=ValueError("Simulated internal error"),
    ), _mock_intent("general_chat"):
        response = client.post(
            "/api/v1/ai/chat",
            json={"message": "trigger the internal error", "user_id": 999},
            headers={"X-User-Id": "999"},
        )

    assert response.status_code == 500
    data = response.json()
    assert data == {
        "error": "Internal server error",
        "detail": "Something went wrong",
        "status_code": 500,
    }
    assert "Traceback" not in response.text
    assert "Simulated internal error" not in response.text


# ==========================================================================
# Day 9 — cross-user memory isolation, remaining intent branches,
#         framework-level error shape, validation handler, hallucination guard
# ==========================================================================

def test_memory_does_not_leak_between_users():
    """
    Cross-pollution guard: a different user must NEVER see prior turns,
    while the owning user still sees their own on the next turn.
    """
    user_a, user_b = 500, 501
    a_message = "my secret gift plan for my wife"
    a_reply = "Great - tell me the occasion and your budget!"

    with patch(
        "app.services.chatbot_service.send_chat_message",
        return_value=a_reply,
    ) as mock_llm, _mock_intent("recommend_product"):
        first = client.post(
            "/api/v1/ai/chat",
            json={"message": a_message, "user_id": user_a},
            headers={"X-User-Id": str(user_a)},
        )
        assert first.status_code == 200
        assert _extract_history_arg(mock_llm.call_args) == ""

    b_reply = "Hi! I'm Sara - how can I help you shop today?"
    with patch(
        "app.services.chatbot_service.send_chat_message",
        return_value=b_reply,
    ) as mock_llm_b, _mock_intent("general_chat"):
        second = client.post(
            "/api/v1/ai/chat",
            json={"message": "hello", "user_id": user_b},
            headers={"X-User-Id": str(user_b)},
        )
        assert second.status_code == 200
        history_b = _extract_history_arg(mock_llm_b.call_args)
        assert a_message not in history_b
        assert a_reply not in history_b

    a_reply2 = "Here are gift ideas under 500 AED."
    with patch(
        "app.services.chatbot_service.send_chat_message",
        return_value=a_reply2,
    ) as mock_llm_a, _mock_intent("recommend_product"):
        third = client.post(
            "/api/v1/ai/chat",
            json={"message": "under 500 AED", "user_id": user_a},
            headers={"X-User-Id": str(user_a)},
        )
        assert third.status_code == 200
        history_a = _extract_history_arg(mock_llm_a.call_args)
        assert a_message in history_a

    db = TestingSessionLocal()
    rows_a = db.query(ChatHistory).filter(ChatHistory.user_id == user_a).count()
    rows_b = db.query(ChatHistory).filter(ChatHistory.user_id == user_b).count()
    db.close()
    assert rows_a == 2
    assert rows_b == 1


def test_track_order_help_uses_tracking_guidance_and_keeps_history():
    """
    track_order_help must ground the reply in the ORDER TRACKING GUIDANCE
    block (no invented tracking info), and history must still be appended.
    """
    user_id = 600
    first_reply = "I'd be happy to help you track your order!"
    with patch(
        "app.services.chatbot_service.send_chat_message",
        return_value=first_reply,
    ) as mock_llm, _mock_intent("track_order_help"):
        r1 = client.post(
            "/api/v1/ai/chat",
            json={"message": "track my order", "user_id": user_id},
            headers={"X-User-Id": str(user_id)},
        )
        assert r1.status_code == 200
        assert r1.json()["intent"] == "track_order_help"
        kwargs1 = mock_llm.call_args.kwargs
        assert "ORDER TRACKING GUIDANCE" in kwargs1["system_prompt"]
        assert "NEVER invent a tracking number" in kwargs1["system_prompt"]
        assert kwargs1.get("system_prompt_override") is None
        assert _extract_history_arg(mock_llm.call_args) == ""

    second_reply = "Your order is with our courier - details are in your email."
    with patch(
        "app.services.chatbot_service.send_chat_message",
        return_value=second_reply,
    ) as mock_llm2, _mock_intent("track_order_help"):
        r2 = client.post(
            "/api/v1/ai/chat",
            json={"message": "where is it now?", "user_id": user_id},
            headers={"X-User-Id": str(user_id)},
        )
        assert r2.status_code == 200
        history2 = _extract_history_arg(mock_llm2.call_args)
        assert "track my order" in history2
        assert first_reply in history2
        assert "ORDER TRACKING GUIDANCE" in mock_llm2.call_args.kwargs["system_prompt"]


def test_general_chat_uses_standard_prompt_with_catalog():
    """
    general_chat explicit: base persona prompt (STRICT GUARDRAILS), no RAG
    override, no tracking guidance, and the product catalog injected.
    """
    mock_reply = "Hi there! I'm Sara - how can I help you shop today?"
    with patch(
        "app.services.chatbot_service.send_chat_message",
        return_value=mock_reply,
    ) as mock_llm, _mock_intent("general_chat"):
        response = client.post(
            "/api/v1/ai/chat",
            json={"message": "hello", "user_id": 700},
            headers={"X-User-Id": "700"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "general_chat"
    assert data["reply"] == mock_reply
    kwargs = mock_llm.call_args.kwargs
    assert kwargs.get("system_prompt_override") is None
    assert "STRICT GUARDRAILS" in kwargs["system_prompt"]
    assert "FAQ CONTEXT:" not in kwargs["system_prompt"]
    assert "ORDER TRACKING GUIDANCE" not in kwargs["system_prompt"]
    assert kwargs["product_context"] != ""


def test_unknown_route_returns_canonical_404_body():
    """
    http_exception_handler (untested until now): a framework-level 404 must
    return the SAME {"error", "detail", "status_code"} shape as our own
    typed errors.
    """
    response = client.get("/api/v1/does-not-exist")
    assert response.status_code == 404
    assert response.json() == {
        "error": "Not Found",
        "detail": "Not Found",
        "status_code": 404,
    }


def test_wrong_method_returns_canonical_405_body():
    """
    http_exception_handler: a 405 (GET on the POST-only chat route) must also
    match the canonical error body - consistency even at the framework edge.
    """
    response = client.get("/api/v1/ai/chat")
    assert response.status_code == 405
    assert response.json() == {
        "error": "Method Not Allowed",
        "detail": "Method Not Allowed",
        "status_code": 405,
    }


def test_malformed_request_body_returns_canonical_400():
    """
    validation_error_handler (untested until now): FastAPI's 422 validation
    errors surface as our canonical 400 body. An empty message violates
    ChatRequest.min_length=1.
    """
    response = client.post(
        "/api/v1/ai/chat",
        json={"message": ""},
        headers={"X-User-Id": "1"},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "Validation error"
    assert body["status_code"] == 400
    assert len(body["detail"]) > 0


def test_hallucinated_price_in_reply_is_replaced():
    """
    The post-generation guard must stop an invented price figure from ever
    reaching the customer: the reply is replaced with a safety message, and
    ONLY the safe text is persisted to chat_history.

    The fake RAG chunks contain no "6,499" figure, the catalog is NOT
    injected on the grounded FAQ path, and the user's message has no price —
    so the figure the (mocked) LLM quoted is provably ungrounded. Note the
    "AED 6,499" currency-first spelling, which the guard's pattern handles.
    """
    user_id = 800
    payload = {"message": "how much is the iphone 14 pro max?", "user_id": user_id}
    hallucinated_reply = (
        "The iPhone 14 Pro Max is available at a great price of AED 6,499."
    )
    with patch(
        "app.services.chatbot_service.retrieve_faq_context",
        return_value=_FAKE_FAQ_CHUNKS,
    ), patch(
        "app.services.chatbot_service.send_chat_message",
        return_value=hallucinated_reply,
    ), _mock_intent("product_faq"):
        response = client.post(
            "/api/v1/ai/chat",
            json=payload,
            headers={"X-User-Id": str(user_id)},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "product_faq"
    assert "6,499" not in data["reply"]
    assert data["reply"] == (
        "Let me double-check that exact figure for you — I don't want to quote "
        "anything that isn't accurate. One moment, please."
    )

    db = TestingSessionLocal()
    row = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).one()
    db.close()
    assert row.ai_response == data["reply"]


def test_grounded_price_from_faq_context_is_kept():
    """
    False-positive check: a figure that IS present in the injected context
    (the RAG FAQ chunks — free shipping over 489 AED) passes the guard
    untouched. The guard must not censor grounded figures.
    """
    payload = {"message": "is shipping free?", "user_id": 801}
    grounded_reply = "Yes — shipping is free on orders over 489 AED."
    with patch(
        "app.services.chatbot_service.retrieve_faq_context",
        return_value=_FAKE_FAQ_CHUNKS,
    ), patch(
        "app.services.chatbot_service.send_chat_message",
        return_value=grounded_reply,
    ), _mock_intent("product_faq"):
        response = client.post(
            "/api/v1/ai/chat",
            json=payload,
            headers={"X-User-Id": "801"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == grounded_reply
    assert "489 AED" in data["reply"]


# ==========================================================================
# Existing tests (pre-Day-9 behavior — all still pass)
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

    # NOTE: the mock quotes THREE figures — 500 AED (the customer's own
    # budget, from the message) plus 399 AED and 450 AED, which are the real
    # catalog prices of the Men's Classic Polo Shirt and Harak Perfume Oud
    # Edition. All three are grounded, so the Day 9 guard keeps the reply
    # intact while still proving memory works.
    second_mock_reply = (
        "Great! Here are some gift ideas under 500 AED: "
        "the Men's Classic Polo Shirt (399 AED) and "
        "Harak Perfume Oud Edition (450 AED)."
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