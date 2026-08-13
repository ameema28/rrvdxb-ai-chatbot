"""
Intent recognition module for the RRVDXB AI Shopping Chatbot.

Day 4 — Intent Recognition + Routing.

This module decides WHAT the user wants BEFORE the chatbot decides HOW to answer.
It runs as its own explicit pipeline step (not embedded in the main chat prompt)
so the service layer can route the request to the right code path.

Pipeline:
  Step 1 — Regex fast-path: instant, free, deterministic. Returns immediately.
  Step 2 — LLM classification: only when regex misses. Uses the EXISTING
           Groq client via send_chat_message() — no second API client.
  Step 3 — Hardened parsing + graceful degradation on any failure.

Day 9 — plural forms added to the track/deal patterns ("discounts", "deals",
"offers", "tracking"...) so common phrasings keep the zero-cost fast-path.
"""

import json
import logging
import re
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field

from app.ai.chatbot.llm_client import send_chat_message
from app.ai.chatbot.prompts import INTENT_CLASSIFICATION_PROMPT

logger = logging.getLogger(__name__)

# The 5 valid intents, shared with the classifier prompt and the DB column.
IntentType = Literal[
    "recommend_product",
    "track_order_help",
    "deal_inquiry",
    "product_faq",
    "general_chat",
]

VALID_INTENTS: tuple[str, ...] = (
    "recommend_product",
    "track_order_help",
    "deal_inquiry",
    "product_faq",
    "general_chat",
)

# LLM-labelled intents below this confidence are overridden to general_chat.
CONFIDENCE_THRESHOLD = 0.7


class IntentResult(BaseModel):
    """
    Structured outcome of intent classification.

    Attributes:
        intent: One of the 5 valid intents.
        confidence: How sure we are (0.0–1.0). 1.0 for regex matches.
        entities: Optional keywords extracted from the message (e.g. budget).
    """

    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence 0.0–1.0")
    entities: Dict[str, Any] = Field(
        default_factory=dict, description="Extracted keywords (optional)"
    )


# --------------------------------------------------------------------------
# Step 1 — Regex fast-path
# --------------------------------------------------------------------------
# Order matters: the FIRST pattern that matches wins (mirrors the spec).
# Patterns are compiled once at import time and searched case-insensitively.
# `s?` pluralization: "discounts"/"deals"/"offers"/"sales"/"coupons"/"promos"
# and "tracking"/"tracks" must hit the fast-path exactly like the singulars —
# otherwise every plural phrasing burns an LLM classification call.
# --------------------------------------------------------------------------

_INTENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "recommend_product",
        re.compile(
            r"\brecommend\b|\bsuggest\b|\bbuy\b|best.*product|what.*buy|"
            r"(?:gift|present)s?\s+(?:ideas?|suggestions?)|"
            r"(?:gift|present)\s+(?:for|idea)",
            re.IGNORECASE,
        ),
    ),
    (
        "track_order_help",
        re.compile(
            r"\btrack(?:ing|s)?\b|order.*status|where.*order|shipping.*status",
            re.IGNORECASE,
        ),
    ),
    (
        "deal_inquiry",
        re.compile(
            r"\bdeals?\b|\bdiscounts?\b|\bsales?\b|\boffers?\b|\bcoupons?\b|\bpromos?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "product_faq",
        re.compile(
            r"\breturn\b.*\bpolicy\b|\bwarranty\b|how.*\breturn\b|"
            r"\bfaq\b|question.*about|"
            r"\b(?:ship|shipping)\s+to\b|"
            r"\bshipping\s+(?:times?|options?|costs?|rates?|policy)\b|"
            r"\bdeliver(?:y|ies)?\s+(?:times?|options?|regions?|to)\b|"
            r"\bin\s+stock\b|\bstock\s+(?:availability|status|levels?|check)\b",
            re.IGNORECASE,
        ),
    ),
]


def _regex_classify(message: str) -> IntentResult | None:
    """Return a full-confidence IntentResult on the first regex match, else None."""
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(message):
            return IntentResult(intent=intent, confidence=1.0, entities={})
    return None


# --------------------------------------------------------------------------
# Step 2 — LLM classification + parsing (only when regex misses)
# --------------------------------------------------------------------------

def _parse_intent_json(raw: str) -> IntentResult:
    """
    Safely convert the LLM's raw output into an IntentResult.

    Tolerances:
      - Strips ```json ... ``` markdown fences if the model wraps output.
      - Uses json.loads() inside try/except.
      - Rejects unknown intent values.
      - Clamps confidence into [0.0, 1.0].

    Raises:
        ValueError: if the payload is not a valid JSON object or the intent
                    is not one of the 5 valid values.
    """
    text = raw.strip()

    # Remove markdown code fences, e.g. ```json\n{...}\n```
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Intent classifier returned malformed JSON: %s", exc)
        raise ValueError("Malformed JSON from intent classifier") from exc

    # Guard against the model returning a JSON array/string/number, not an object
    if not isinstance(data, dict):
        logger.warning("Intent classifier returned non-object JSON: %r", text[:120])
        raise ValueError("Expected JSON object from intent classifier")

    intent = data.get("intent")
    if intent not in VALID_INTENTS:
        logger.warning("Intent classifier returned unknown intent: %r", intent)
        raise ValueError(f"Unknown intent: {intent!r}")

    # Confidence: coerce to float (handles "0.85" strings), then clamp [0,1]
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)

    entities = data.get("entities") or {}

    return IntentResult(intent=intent, confidence=confidence, entities=entities)


def classify_intent(user_message: str) -> IntentResult:
    """
    Classify the user's intent. Regex first, LLM second, always graceful.

    Args:
        user_message: Raw customer message.

    Returns:
        IntentResult. Guaranteed to hold one of the 5 valid intents —
        never raises, because any failure degrades to general_chat.
    """
    # Step 1 — regex fast-path (zero LLM cost, zero latency)
    matched = _regex_classify(user_message)
    if matched is not None:
        logger.debug("Regex classified intent=%s (no LLM call)", matched.intent)
        return matched

    # Step 2 — LLM classification, only reached when regex misses
    try:
        raw = send_chat_message(
            system_prompt=INTENT_CLASSIFICATION_PROMPT,
            user_message=user_message,
            product_context="",  # intent classification needs no catalog
            history="",          # each message is classified independently
        )
        result = _parse_intent_json(raw)
    except Exception as exc:
        # Covers API timeout/rate-limit/auth errors, malformed JSON, unknown
        # intents, invalid confidence. Never log the API key — only the error.
        logger.warning(
            "Intent classification failed (%s: %s) — falling back to general_chat",
            type(exc).__name__,
            exc,
        )
        return IntentResult(intent="general_chat", confidence=0.0, entities={})

    # Step 3 — confidence threshold: uncertain labels default to general_chat.
    # We KEEP the original confidence in the result for observability.
    if result.confidence < CONFIDENCE_THRESHOLD:
        logger.info(
            "Intent %s below threshold (%.2f) — overriding to general_chat",
            result.intent,
            result.confidence,
        )
        return IntentResult(
            intent="general_chat",
            confidence=result.confidence,
            entities=result.entities,
        )

    return result