"""
Price guard: currency coverage, the per-product check, and every reply path.
"""

import json
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.ai.chatbot.intent import IntentResult
from app.ai.chatbot.price_guard import (
    GUARD_REPLY,
    build_catalog,
    extract_price_figures,
    extract_price_mentions,
    find_product_price_mismatches,
    validate_reply_prices,
)
from app.main import app
from app.models.chat_history import ChatHistory
from tests.support import TestingSessionLocal, auth_headers

PRODUCTS = json.loads(
    (Path(__file__).resolve().parents[1] / "app" / "mock_data" / "products.json").read_text(
        encoding="utf-8"
    )
)
CATALOG = build_catalog(PRODUCTS)
client = TestClient(app)


# ==========================================================================
# Unit: currency extraction and normalisation
# ==========================================================================

@pytest.mark.parametrize(
    "text, figure",
    [
        ("AED 6,499", "6499.00 AED"),
        ("6,499 AED", "6499.00 AED"),
        ("aed 10", "10.00 AED"),
        ("USD 99.50", "99.50 USD"),
        ("99.50 usd", "99.50 USD"),
        ("SAR 1,200", "1200.00 SAR"),
        ("1,200 SAR", "1200.00 SAR"),
        ("GBP 75", "75.00 GBP"),
        ("75 GBP", "75.00 GBP"),
        ("PKR 150,000", "150000.00 PKR"),
        ("150,000 PKR", "150000.00 PKR"),
        ("Rs 4,699", "4699.00 PKR"),
        ("Rs. 4,699", "4699.00 PKR"),
        ("Rs.4699", "4699.00 PKR"),
        ("4,699 Rs", "4699.00 PKR"),
        ("₨ 4,699", "4699.00 PKR"),
        ("₨4699", "4699.00 PKR"),
        ("₹ 999", "999.00 INR"),
        ("$1,299", "1299.00 USD"),
        ("$ 1,299.99", "1299.99 USD"),
        ("1299$", "1299.00 USD"),
        ("£50", "50.00 GBP"),
        ("50 £", "50.00 GBP"),
        ("489 dirhams", "489.00 AED"),
        ("1 dirham", "1.00 AED"),
        ("dirhams 489", "489.00 AED"),
        ("AED-489", "489.00 AED"),
    ],
)
def test_every_currency_form_is_recognised(text, figure):
    assert extract_price_figures(f"It costs {text} today.") == {figure}


@pytest.mark.parametrize(
    "text",
    ["Mrs 50 visitors", "users 20 times", "the WH-1000XM5 is great", "2nd Generation", "about 500"],
)
def test_non_prices_are_not_matched(text):
    assert extract_price_figures(text) == set()


def test_decimal_amounts_are_compared_exactly():
    assert extract_price_figures("4,699.00 AED") == extract_price_figures("4699 AED")
    assert extract_price_figures("4,699.50 AED") != extract_price_figures("4,699 AED")
    result = validate_reply_prices("Now 4,699.50 AED.", ["4,699.00 AED"], "", "", [])
    assert result.tripped and result.ungrounded == {"4699.50 AED"}


def test_mention_spans_cover_the_whole_token():
    (mention,) = extract_price_mentions("Only Rs. 4,699 today")
    assert "Only Rs. 4,699 today"[mention.start:mention.end] == "Rs. 4,699"
    assert mention.amount == Decimal("4699.00") and mention.currency == "PKR"


# ==========================================================================
# Unit: catalog aliases and the per-product binding heuristic
# ==========================================================================

def test_every_product_has_aliases_and_they_are_unique():
    seen = {}
    brands = {}
    for p in PRODUCTS:
        assert p["aliases"], f"product {p['id']} has no aliases"
        brands.setdefault(p["brand"].lower(), set()).add(p["id"])
        for name in [p["name"], *p["aliases"]]:
            key = name.lower()
            assert seen.setdefault(key, p["id"]) == p["id"], f"alias {name!r} is shared"
    shared_brands = {b for b, ids in brands.items() if len(ids) > 1}
    for p in PRODUCTS:
        assert not {a.lower() for a in p["aliases"]} & shared_brands


def test_correct_prices_for_two_products_in_one_sentence():
    reply = "Try the Men's Classic Polo Shirt (399 AED) and Harak Perfume Oud Edition (450 AED)."
    assert find_product_price_mismatches(reply, CATALOG) == []


def test_swapped_prices_are_mismatches():
    reply = "Try the Polo Shirt (450 AED) and the Harak Perfume (399 AED)."
    assert find_product_price_mismatches(reply, CATALOG) == [
        (10, "450.00 AED"),
        (7, "399.00 AED"),
    ]


def test_qualified_budget_is_not_bound_to_a_product():
    reply = "Gift ideas under 500 AED: the polo at 399 AED."
    assert find_product_price_mismatches(reply, CATALOG) == []


def test_price_in_next_sentence_is_not_bound():
    reply = "The AirPods Pro are lovely. Shipping is free over AED 489."
    assert find_product_price_mismatches(reply, CATALOG) == []


def test_wrong_currency_for_a_product_is_a_mismatch():
    assert find_product_price_mismatches("The iPhone 14 Pro Max is $4,699.", CATALOG) == [
        (1, "4699.00 USD")
    ]


def test_dirhams_match_an_aed_catalog_price():
    assert find_product_price_mismatches("The iPhone 14 Pro Max is 4,699 dirhams.", CATALOG) == []


def test_rs_period_does_not_end_the_clause():
    assert find_product_price_mismatches("The DualSense is Rs. 349 here.", CATALOG) == [
        (2, "349.00 PKR")
    ]


def test_documented_limit_price_before_product_is_not_bound():
    """Known limit: a figure written before the product name is not bound."""
    reply = "For 899 AED you get the iPhone 14 Pro Max."
    assert find_product_price_mismatches(reply, CATALOG) == []


# ==========================================================================
# Endpoint: every reply path
# ==========================================================================

_FAQ_CHUNKS = [
    {
        "question": "What is the free shipping threshold?",
        "answer": "Free standard shipping is available on all orders over AED 489.",
        "similarity": 0.9,
    }
]
_RECS = [
    {"id": 1, "name": "iPhone 14 Pro Max", "price": 4699.0, "currency": "AED",
     "category": "Electronics", "brand": "Apple", "reason": "Matches phone"},
    {"id": 6, "name": "AirPods Pro (2nd Generation)", "price": 899.0, "currency": "AED",
     "category": "Electronics", "brand": "Apple", "reason": "Matches apple"},
]
_DEAL = {"title": "Summer Sale", "discount": "20% off", "code": "SUMMER20"}

# path name -> (intent, extra patches)
PATHS = {
    "product_faq_match": ("product_faq", {"retrieve_faq_context": _FAQ_CHUNKS}),
    "product_faq_no_match": ("product_faq", {"retrieve_faq_context": []}),
    "recommend_with_recs": ("recommend_product", {"get_recommendations": _RECS}),
    "recommend_no_recs": ("recommend_product", {"get_recommendations": []}),
    "deal_found": ("deal_inquiry", {"get_deals": _DEAL}),
    "deal_none": ("deal_inquiry", {"get_deals": None}),
    "track_order_help": ("track_order_help", {}),
    "general_chat": ("general_chat", {}),
}

# A grounded reply each path must KEEP (figures the model actually saw).
GROUNDED = {
    "product_faq_match": ("is shipping free?", "Shipping is free on orders over AED 489."),
    "deal_found": (
        "any deals under 500 AED?",
        "Our Summer Sale takes 20% off, including items under 500 AED.",
    ),
}
DEFAULT_GROUNDED = ("how much is the iphone?", "The iPhone 14 Pro Max is 4,699 AED.")

WRONG_BUT_REAL = "The iPhone 14 Pro Max is 899 AED."  # a real AirPods/handbag price
INVENTED = "The iPhone 14 Pro Max is now AED 6,499."


def _chat(path: str, message: str, reply: str, sub: str):
    intent, extras = PATHS[path]
    patches = [
        patch("app.services.chatbot_service.send_chat_message", return_value=reply),
        patch(
            "app.services.chatbot_service.classify_intent",
            return_value=IntentResult(intent=intent, confidence=1.0),
        ),
    ] + [
        patch(f"app.services.chatbot_service.{name}", return_value=value)
        for name, value in extras.items()
    ]
    for p in patches:
        p.start()
    try:
        response = client.post(CHAT_URL, json={"message": message}, headers=auth_headers(sub))
    finally:
        for p in reversed(patches):
            p.stop()
    assert response.status_code == 200
    return response.json()


CHAT_URL = "/api/v1/ai/chat"


def _saved_reply(sub: str) -> str:
    db = TestingSessionLocal()
    try:
        return db.query(ChatHistory).filter(ChatHistory.user_id == sub).one().ai_response
    finally:
        db.close()


@pytest.mark.db
@pytest.mark.parametrize("path", list(PATHS))
def test_wrong_but_real_price_is_replaced_on_every_path(path):
    sub = f"pg-wrong-{path}"
    data = _chat(path, "how much is the iphone?", WRONG_BUT_REAL, sub)
    assert data["reply"] == GUARD_REPLY
    assert _saved_reply(sub) == GUARD_REPLY


@pytest.mark.db
@pytest.mark.parametrize("path", list(PATHS))
def test_invented_price_is_replaced_on_every_path(path):
    sub = f"pg-invented-{path}"
    data = _chat(path, "how much is the iphone?", INVENTED, sub)
    assert data["reply"] == GUARD_REPLY
    assert _saved_reply(sub) == GUARD_REPLY


@pytest.mark.db
@pytest.mark.parametrize("path", list(PATHS))
def test_grounded_price_is_kept_on_every_path(path):
    sub = f"pg-grounded-{path}"
    message, reply = GROUNDED.get(path, DEFAULT_GROUNDED)
    data = _chat(path, message, reply, sub)
    assert data["reply"] == reply
    assert _saved_reply(sub) == reply


def test_wrong_but_real_price_is_caught_by_the_product_check_not_presence():
    """On a catalog path 899 AED IS in the context, so only the per-product
    check can catch it."""
    catalog_context = "\n".join(f"- {p['name']} | {p['price']:.2f} AED" for p in PRODUCTS)
    result = validate_reply_prices(WRONG_BUT_REAL, [catalog_context], "", "", CATALOG)
    assert result.ungrounded == set()
    assert result.mismatched == [(1, "899.00 AED")]
    assert result.reply == GUARD_REPLY


@pytest.mark.db
@pytest.mark.parametrize(
    "figure",
    ["AED 12,345", "12,345 USD", "SAR 777", "777 GBP", "PKR 777", "Rs 777", "Rs. 777",
     "₨777", "₹777", "$777", "£777", "777 dirhams", "1 dirham"],
)
def test_ungrounded_figure_in_any_currency_is_replaced(figure):
    sub = "pg-cur-" + str(abs(hash(figure)))
    data = _chat("general_chat", "hello", f"That one is {figure} today.", sub)
    assert data["reply"] == GUARD_REPLY
