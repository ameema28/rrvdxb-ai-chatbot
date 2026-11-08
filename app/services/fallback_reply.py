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

NO_FAQ_MATCH = (
    "I couldn't find a matching answer in our FAQ for that question. "
    + SUPPORT_OFFER
)

PRODUCTS_LEAD_IN = "Here are some products from our catalog that match your request:"
DEAL_LEAD_IN = "Here's our current offer:"

NO_DEAL ="I don't have a current offer to show right now. " + SUPPORT_OFFER

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
    for the static help text and the "nothing found" messages (which keep the
    human-support offer).
    """
    if intent == "product_faq":
        if faq_chunks:
            return str(faq_chunks[0]["answer"]).strip(), True
        return NO_FAQ_MATCH, False

    if intent == "recommend_product":
        if recommendations:
            # The service always returns these as ChatResponse.recommended_products
            # and the page renders them, so the text only introduces them.
            return PRODUCTS_LEAD_IN, True
        return NO_RECOMMENDATION, False

    if intent == "deal_inquiry":
        if deal:
            # The deal is also returned as ChatResponse.deal; the page shows it.
            return DEAL_LEAD_IN, True
        return NO_DEAL, False

    if intent == "track_order_help":
        return "Here is how order tracking works:\n" + "\n".join(
            f"- {fact}" for fact in TRACKING_FACTS
        ), True

    # general_chat (the only remaining intent): a static help text, not
    # retrieved content, so a timeout keeps its own short message instead of
    # appending it.
    return HELP_MESSAGE, False
