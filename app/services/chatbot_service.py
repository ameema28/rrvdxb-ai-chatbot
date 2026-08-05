"""
Chatbot service layer — business logic for processing messages.

This layer sits between the API route and the AI client. It handles:
  - Loading recent conversation history from the database
  - Building the prompt with history + product context
  - Calling the Groq LLM
  - Persisting the new turn to chat_history
  - Returning a structured ChatResponse
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

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


async def handle_chat_message(
    user_id: Optional[int],
    message: str,
    db: Session,
) -> ChatResponse:
    """
    Process a user message and return an AI chat response with memory.

    Day 3 implementation:
    - Loads the last N conversation turns for this user from chat_history
    - Formats history into the prompt so the AI remembers context
    - Loads product catalog and injects it into the system prompt
    - Calls Groq LLM with system prompt + history + user message
    - Persists the new turn to chat_history
    - Returns the AI reply in a ChatResponse

    Args:
        user_id: Authenticated user identifier. If None, falls back to
                 anonymous user_id 0 so the turn is still persisted.
        message: Raw user input.
        db: Active SQLAlchemy session.

    Returns:
        ChatResponse with AI reply. recommended_products and deal are
        left empty until Day 5+.
    """
    # Fallback for unauthenticated / anonymous users
    effective_user_id = user_id if user_id is not None else 0

    # ------------------------------------------------------------------
    # 1. Load recent conversation history for this user
    # ------------------------------------------------------------------
    recent_turns = load_recent_history(
        db_session=db, user_id=effective_user_id, limit=5
    )
    history_string = format_history_for_prompt(recent_turns)

    # ------------------------------------------------------------------
    # 2. Load product catalog and build context string
    # ------------------------------------------------------------------
    try:
        products = _load_json("products.json")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.error("Failed to load product catalog: %s", exc)
        products = []

    product_context = _build_product_context(products)

    # ------------------------------------------------------------------
    # 3. Call the LLM via Groq SDK, passing history + product context
    # ------------------------------------------------------------------
    try:
        reply = send_chat_message(
            system_prompt=SYSTEM_PROMPT,
            user_message=message,
            product_context=product_context,
            history=history_string,
        )
    except RuntimeError as exc:
        # The LLM client already logged the technical details.
        # We return a friendly fallback to the user.
        logger.warning("LLM call failed, returning fallback: %s", exc)
        reply = (
            "I'm here to help with your RRVDXB shopping experience. "
            "I'm having a little trouble right now, but our team is on it. "
            "Would you like me to connect you with human support?"
        )

    # ------------------------------------------------------------------
    # 4. Persist the new turn to the database
    # ------------------------------------------------------------------
    save_turn(
        db_session=db,
        user_id=effective_user_id,
        message=message,
        ai_response=reply,
        intent=None,  # Will be classified by AI pipeline on Day 4+
    )

    # ------------------------------------------------------------------
    # 5. Return structured response
    # ------------------------------------------------------------------
    return ChatResponse(
        reply=reply,
        recommended_products=[],
        deal=None,
    )