"""
Deal finder stub (Day 7).

Teammate-owned service placeholder. Defines the interface the deals team's
real service must match, with a deterministic local mock so deal_inquiry
flows never block on an external service.

Interface:
    get_deals(user_id: int, message: str) -> dict | None

When the real service lands, swap only the import in chatbot_service.py.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Keywords that trigger a mock deal.
_DEAL_KEYWORDS = ("discount", "sale", "deal", "offer", "coupon", "promo")


def get_deals(user_id: int, message: str) -> Optional[Dict[str, Any]]:
    """
    Return a mock deal dict when the message mentions deals, else None.

    Args:
        user_id: Authenticated user identifier (unused for keyword matching).
        message: Raw customer message.

    Returns:
        {"title": str, "discount": str, "code": str} or None when no deal
        keyword is present.
    """
    lowered = message.lower()
    if any(kw in lowered for kw in _DEAL_KEYWORDS):
        logger.info("deal_finder_stub: deal matched for user_id=%s", user_id)
        return {
            "title": "Summer Sale",
            "discount": "20% off",
            "code": "SUMMER20",
        }
    logger.debug("deal_finder_stub: no deal keyword in message")
    return None