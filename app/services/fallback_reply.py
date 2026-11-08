"""
Deterministic replies for when the LLM is unavailable (error or timeout).

The text is built only from content the request already retrieved for the
intent: the matched FAQ answer, the recommended products with their catalog
prices, the deal text, or the order-tracking guidance. Nothing is generated
or invented here. The service still runs the price guard on the result.
"""

from typing import Any, Dict, List, Optional, Sequence

# Facts about order tracking. Shared with the LLM prompt so both stay in sync.
TRACKING_FACTS: Sequence[str] = (
    "RRVDXB emails/SMSes tracking details once an order ships.",
    "Customers can also check live status under 'My Orders'.",
)

SUPPORT_OFFER = "Would you like me to connect you with human support?"

GENERIC_FALLBACK = (
    "I'm here to help with your RRVDXB shopping experience. "
    "I'm having a little trouble right now, but our team is on it. "
    + SUPPORT_OFFER
)

NO_FAQ_MATCH = (
    "I couldn't find a matching answer in our FAQ for that question. "
    + SUPPORT_OFFER
)

NO_DEAL = "I don't have a current offer to show right now. " + SUPPORT_OFFER

NO_RECOMMENDATION = (
    "I couldn't find a matching product in our catalog. Try naming a product "
    "type, for example a perfume, shoes or headphones. " + SUPPORT_OFFER
)

HELP_MESSAGE = (
    "I'm Sara, the RRVDXB shopping assistant. I can recommend products from "
    "our catalog, answer questions about returns, shipping and payments, "
    "tell you about current deals, and explain how to track your order. "
    "Try asking about one of these."
)


def build_fallback_reply(
    intent: str,
    faq_chunks: Optional[List[Dict[str, Any]]] = None,
    recommendations: Optional[List[Dict[str, Any]]] = None,
    deal: Optional[str] = None,
) -> tuple[str, bool]:
    """
    Return (reply, has_content) for the intent from already-retrieved data.

    has_content is True only when the reply carries content retrieved for
    this request (FAQ answer, products, deal, tracking guidance). It is False
    for the static help text, the "nothing found" messages (which keep the
    human-support offer) and the generic message.
    """
    if intent == "product_faq":
        if faq_chunks:
            return str(faq_chunks[0]["answer"]).strip(), True
        return NO_FAQ_MATCH, False

    if intent == "recommend_product":
        if recommendations:
            lines = [
                f"- {rec['name']} ({rec['brand']}): "
                f"{rec.get('currency', 'AED')} {float(rec['price']):.2f}"
                for rec in recommendations
            ]
            return "Here are some products from our catalog:\n" + "\n".join(lines), True
        return NO_RECOMMENDATION, False

    if intent == "deal_inquiry":
        if deal:
            return f"Current offer: {deal}", True
        return NO_DEAL, False

    if intent == "track_order_help":
        return "Here is how order tracking works:\n" + "\n".join(
            f"- {fact}" for fact in TRACKING_FACTS
        ), True

    if intent == "general_chat":
        # A static help text, not retrieved content: a timeout keeps its own
        # short message instead of appending it.
        return HELP_MESSAGE, False

    return GENERIC_FALLBACK, False
