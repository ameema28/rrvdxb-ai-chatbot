"""
Chatbot service layer — business logic for processing messages.

This layer sits between the API route and the AI client. It handles:
  - Classifying the user's intent (Day 4)
  - Routing to the right code path based on that intent
  - Loading recent conversation history from the database
  - Building the prompt with history + intent-specific context
  - Calling the Groq LLM
  - Persisting the new turn (with intent) to chat_history
  - Returning a structured ChatResponse
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.ai.chatbot.intent import classify_intent
from app.ai.chatbot.llm_client import send_chat_message
from app.ai.chatbot.memory import (
    format_history_for_prompt,
    load_recent_history,
    save_turn,
)
from app.ai.chatbot.prompts import SYSTEM_PROMPT
from app.schemas.chatbot_schema import ChatResponse

logger = logging.getLogger(__name__)

# Path to mock data files
_MOCK_DIR = os.path.join(os.path.dirname(__file__), "..", "mock_data")


def _load_json(filename: str) -> Any:
    """Helper to load JSON mock data."""
    path = os.path.join(_MOCK_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_product_context(products: List[Dict[str, Any]]) -> str:
    """
    Format the product catalog into a concise string for the LLM system prompt.

    We include only the fields the AI needs to answer customer questions:
    name, price, currency, category, brand, and stock level.
    """
    lines = []
    for p in products:
        line = (
            f"- {p['name']} | {p['brand']} | {p['category']} | "
            f"{p['price']:.2f} {p.get('currency', 'AED')} | "
            f"Stock: {p['stock']} units"
        )
        lines.append(line)
    return "\n".join(lines)


def _build_system_prompt_for_intent(
    intent: str, product_context: str
) -> Tuple[str, str]:
    """
    Route the request based on intent and prepare (system_prompt, context).

    This is where different intents diverge into different code paths:
      - recommend_product: inject the full product catalog (existing behavior).
      - product_faq:      future RAG grounding (TODO Day 6).
      - deal_inquiry:     future deal engine (TODO Day 7).
      - track_order_help: append order-tracking guidance to the prompt.
      - general_chat:     standard flow.

    Returns:
        A tuple of (system_prompt_to_use, context_to_inject).
    """
    if intent == "recommend_product":
        # Existing behavior: the catalog lets the model ground its suggestions.
        return SYSTEM_PROMPT, product_context

    if intent == "product_faq":
        # TODO: Day 6 — RAG pipeline will retrieve faqs.json + product docs
        # and inject the grounded snippets here instead of the raw catalog.
        return SYSTEM_PROMPT, product_context

    if intent == "deal_inquiry":
        # TODO: Day 7 — Deal Finder integration will inject active
        # promotions/deals context here instead of the raw catalog.
        return SYSTEM_PROMPT, product_context

    if intent == "track_order_help":
        # Order tracking has no product data to lean on — instead we give the
        # model accurate process guidance and forbid inventing tracking info.
        tracking_note = (
            "\n\nORDER TRACKING GUIDANCE:\n"
            "- RRVDXB emails/SMSes tracking details once an order ships.\n"
            "- Customers can also check live status under 'My Orders'.\n"
            "- NEVER invent a tracking number, carrier, or delivery date."
        )
        return SYSTEM_PROMPT + tracking_note, product_context

    # general_chat (and any future default) — standard chat flow.
    return SYSTEM_PROMPT, product_context


async def handle_chat_message(
    user_id: Optional[int],
    message: str,
    db: Session,
) -> ChatResponse:
    """
    Process a user message and return an AI chat response with memory.

    Day 4 flow:
    - Step 1: Classify intent (regex fast-path, else LLM).
    - Step 2: Route to the intent-specific code path (context prep).
    - Step 3: Load recent history and build the prompt.
    - Step 4: Call Groq LLM.
    - Step 5: Persist the turn WITH the classified intent.
    - Step 6: Return ChatResponse (now including intent).

    Args:
        user_id: Authenticated user identifier (None -> anonymous 0).
        message: Raw user input.
        db: Active SQLAlchemy session.

    Returns:
        ChatResponse with the AI reply, empty recommended_products/deal,
        and the classified intent.
    """
    # Fallback for unauthenticated / anonymous users
    effective_user_id = user_id if user_id is not None else 0

    # ------------------------------------------------------------------
    # 1. Classify intent FIRST — decide the path before we build anything
    # ------------------------------------------------------------------
    intent_result = classify_intent(message)
    intent = intent_result.intent
    logger.debug(
        "Classified intent=%s (confidence=%.2f) for message=%r",
        intent,
        intent_result.confidence,
        message[:80],
    )

    # ------------------------------------------------------------------
    # 2. Load recent conversation history for this user
    # ------------------------------------------------------------------
    recent_turns = load_recent_history(
        db_session=db, user_id=effective_user_id, limit=5
    )
    history_string = format_history_for_prompt(recent_turns)

    # ------------------------------------------------------------------
    # 3. Load product catalog and build context string
    # ------------------------------------------------------------------
    try:
        products = _load_json("products.json")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.error("Failed to load product catalog: %s", exc)
        products = []

    product_context = _build_product_context(products)

    # ------------------------------------------------------------------
    # 4. Route: pick system prompt + context based on the intent
    # ------------------------------------------------------------------
    system_prompt, context = _build_system_prompt_for_intent(intent, product_context)

    # ------------------------------------------------------------------
    # 5. Call the LLM via Groq SDK, passing history + routed context
    # ------------------------------------------------------------------
    try:
        reply = send_chat_message(
            system_prompt=system_prompt,
            user_message=message,
            product_context=context,
            history=history_string,
        )
    except RuntimeError as exc:
        # The LLM client already logged the technical details.
        logger.warning("LLM call failed, returning fallback: %s", exc)
        reply = (
            "I'm here to help with your RRVDXB shopping experience. "
            "I'm having a little trouble right now, but our team is on it. "
            "Would you like me to connect you with human support?"
        )

    # ------------------------------------------------------------------
    # 6. Persist the new turn, including the classified intent
    # ------------------------------------------------------------------
    save_turn(
        db_session=db,
        user_id=effective_user_id,
        message=message,
        ai_response=reply,
        intent=intent,
    )

    # ------------------------------------------------------------------
    # 7. Return structured response (intent exposed for observability)
    # ------------------------------------------------------------------
    return ChatResponse(
        reply=reply,
        recommended_products=[],
        deal=None,
        intent=intent,
        confidence=intent_result.confidence,
    )