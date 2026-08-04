"""
Chatbot service layer — business logic for processing messages.

This layer sits between the API route and the AI client. It handles:
  - Persisting chat history to the database
  - Loading product catalog context
  - Building the prompt and calling the Groq LLM
  - Returning a structured ChatResponse
"""

import json
import logging
import os
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from app.ai.chatbot.llm_client import send_chat_message
from app.ai.chatbot.prompts import SYSTEM_PROMPT
from app.models.chat_history import ChatHistory
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
    user_id: int,
    message: str,
    db: Session,
) -> ChatResponse:
    """
    Process a user message and return an AI chat response.

    Day 2 implementation:
    - Loads product catalog from JSON
    - Builds a product context string
    - Calls Groq LLM with SYSTEM_PROMPT + context + user message
    - Persists the turn to chat_history
    - Returns the AI reply in a ChatResponse

    Args:
        user_id: Authenticated user identifier.
        message: Raw user input.
        db: Active SQLAlchemy session.

    Returns:
        ChatResponse with AI reply. recommended_products and deal are
        left empty until Day 3 (intent classification) and Day 5 (deals).
    """
    # ------------------------------------------------------------------
    # 1. Load product catalog and build context string
    # ------------------------------------------------------------------
    try:
        products = _load_json("products.json")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.error("Failed to load product catalog: %s", exc)
        products = []

    product_context = _build_product_context(products)

    # ------------------------------------------------------------------
    # 2. Call the LLM via Groq SDK
    # ------------------------------------------------------------------
    try:
        reply = send_chat_message(
            system_prompt=SYSTEM_PROMPT,
            user_message=message,
            product_context=product_context,
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
    # 3. Persist to database
    # ------------------------------------------------------------------
    chat_turn = ChatHistory(
        user_id=user_id,
        message=message,
        ai_response=reply,
        intent=None,  # Will be classified by AI pipeline on Day 3
    )
    db.add(chat_turn)
    db.commit()

    # ------------------------------------------------------------------
    # 4. Return structured response
    # ------------------------------------------------------------------
    # Day 2: We return the AI reply only. recommended_products and deal
    # will be populated when we add intent classification (Day 3) and
    # vector search (Day 6).
    return ChatResponse(
        reply=reply,
        recommended_products=[],
        deal=None,
    )