"""
Chatbot service layer — business logic for processing messages.

This layer sits between the API route and the AI client. It handles:
  - Classifying the user's intent (Day 4)
  - Routing to the right code path based on that intent
  - Loading recent conversation history from the database (Day 3)
  - Retrieving FAQ context for product_faq intents (Day 6 RAG)
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
from app.ai.chatbot.prompts import SYSTEM_PROMPT, build_rag_system_prompt
from app.ai.chatbot.rag.retriever import retrieve_faq_context
from app.schemas.chatbot_schema import ChatResponse

logger = logging.getLogger(__name__)

# Path to mock data files
_MOCK_DIR = os.path.join(os.path.dirname(__file__), "..", "mock_data")

# Day 6 — appended to the general SYSTEM_PROMPT when a product_faq has no
# matching FAQ chunk, so the chatbot admits the gap instead of inventing one.
_NO_FAQ_MATCH_NOTE = (
    "\n\nFAQ NOTE:\n"
    "The customer asked a product or policy question, but no matching FAQ "
    "entry was found in our knowledge base. Do NOT invent a policy, price, "
    "or availability figure. Politely explain that you do not have that "
    "specific information right now and offer to connect them with human "
    "support."
)


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
    Route NON-RAG intents and prepare (system_prompt, context).

    product_faq is NOT routed here (Day 6): it is handled earlier in
    handle_chat_message() by the RAG block, which must call the retriever
    before it can decide between a grounded answer and the graceful fallback.

    Returns:
        A tuple of (system_prompt_to_use, context_to_inject).
    """
    if intent == "recommend_product":
        # Existing behaviour: the catalog lets the model ground its suggestions.
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

    Day 4-6 flow:
    - Step 1: Classify intent (regex fast-path, else LLM).
    - Step 2: Load recent history (memory).
    - Step 3: Load product catalog context.
    - Step 4: Route — product_faq goes through the Day-6 RAG retriever;
              every other intent uses the intent-aware router.
    - Step 5: Call the Groq LLM (history + routed context, optional RAG
              system_prompt_override).
    - Step 6: Persist the turn WITH the classified intent.
    - Step 7: Return ChatResponse (reply, intent, confidence).

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
    # 2. Load recent conversation history for this user (memory, Day 3)
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
    # 4. Route — RAG path for product_faq, intent router for everything else
    # ------------------------------------------------------------------
    system_prompt_override: Optional[str] = None

    if intent == "product_faq":
        # 4a. Day 6 — pull trustworthy FAQ chunks for this exact question.
        faq_chunks = retrieve_faq_context(message)
        if faq_chunks:
            logger.info(
                "RAG: retrieved %d FAQ chunk(s) for message=%r",
                len(faq_chunks),
                message[:80],
            )
            # Grounded prompt built in prompts.py (guardrails + FAQ context).
            # product_context stays "" so the raw catalog is NOT layered
            # on top of the FAQ context.
            system_prompt_override = build_rag_system_prompt(faq_chunks)
            system_prompt, context = SYSTEM_PROMPT, ""
        else:
            # 4b. Graceful degradation: nothing was relevant enough. Fall
            # back to the general flow, but tell the model to admit the gap
            # instead of inventing a policy (concept: honesty over guessing).
            logger.info(
                "RAG: no FAQ match for message=%r — graceful fallback",
                message[:80],
            )
            system_prompt, context = (
                SYSTEM_PROMPT + _NO_FAQ_MATCH_NOTE,
                product_context,
            )
    else:
        # 4c. Non-RAG intents keep the existing intent-aware router.
        system_prompt, context = _build_system_prompt_for_intent(
            intent, product_context
        )

    # ------------------------------------------------------------------
    # 5. Call the Groq LLM. When system_prompt_override is set it fully
    #    replaces the base prompt and skips catalog injection, but history
    #    is still appended inside llm_client — memory survives the RAG flow.
    # ------------------------------------------------------------------
    try:
        reply = send_chat_message(
            system_prompt=system_prompt,
            user_message=message,
            product_context=context,
            history=history_string,
            system_prompt_override=system_prompt_override,
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