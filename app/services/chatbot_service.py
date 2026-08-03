"""
Chatbot service layer — business logic for processing messages.

This layer sits between the API route and the AI client. It handles:
  - Persisting chat history to the database
  - Loading mock product/FAQ context
  - Building the prompt and calling the AI (placeholder for now)
  - Returning a structured ChatResponse
"""

from typing import Optional
import json
import os

from sqlalchemy.orm import Session

from app.schemas.chatbot_schema import ChatResponse, RecommendedProduct
from app.models.chat_history import ChatHistory


# Path to mock data files
_MOCK_DIR = os.path.join(os.path.dirname(__file__), "..", "mock_data")


def _load_json(filename: str):
    """Helper to load JSON mock data."""
    path = os.path.join(_MOCK_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def handle_chat_message(
    user_id: int,
    message: str,
    db: Session,
) -> ChatResponse:
    """
    Process a user message and return an AI chat response.

    PLACEHOLDER IMPLEMENTATION (Day 1):
    - Echoes the message back
    - Loads mock products to simulate recommendations
    - Saves the turn to chat_history
    - Does NOT call OpenAI yet (will be added in Day 2/3)

    Args:
        user_id: Authenticated user identifier.
        message: Raw user input.
        db: Active SQLAlchemy session.

    Returns:
        ChatResponse with reply and optional product recommendations.
    """
    # ------------------------------------------------------------------
    # 1. Load mock context (in production this will be a vector DB call)
    # ------------------------------------------------------------------
    products = _load_json("products.json")
    faqs = _load_json("faqs.json")

    # Pick up to 2 products as "recommendations" for the placeholder
    recommended = [
        RecommendedProduct(
            id=p["id"],
            name=p["name"],
            price=p["price"],
            currency=p.get("currency", "AED"),
            category=p["category"],
            brand=p["brand"],
            reason=f"Popular in {p['category']}",
        )
        for p in products[:2]
    ]

    # ------------------------------------------------------------------
    # 2. Build a placeholder reply
    # ------------------------------------------------------------------
    reply = (
        f"Hello! You asked: '{message}'. "
        "I'm currently learning about our catalog. "
        "Here are a couple of items you might like!"
    )

    # ------------------------------------------------------------------
    # 3. Persist to database
    # ------------------------------------------------------------------
    chat_turn = ChatHistory(
        user_id=user_id,
        message=message,
        ai_response=reply,
        intent=None,  # Will be classified by AI pipeline later
    )
    db.add(chat_turn)
    db.commit()

    # ------------------------------------------------------------------
    # 4. Return structured response
    # ------------------------------------------------------------------
    return ChatResponse(
        reply=reply,
        recommended_products=recommended,
        deal=None,
    )