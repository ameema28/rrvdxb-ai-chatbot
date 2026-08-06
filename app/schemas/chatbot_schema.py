"""
Pydantic schemas for the chatbot API.

Schemas define the shape of request/response data and provide automatic
validation, serialization, and OpenAPI documentation generation.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Incoming chat message from the user.

    Attributes:
        message: The user's natural-language query.
        user_id: Optional explicit user ID (fallback if header is unavailable).
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's chat message",
        examples=["Do you have the iPhone 14 Pro Max in stock?"],
    )
    user_id: Optional[int] = Field(
        None,
        description="Optional user ID override",
    )


class RecommendedProduct(BaseModel):
    """A product suggested by the AI based on user intent."""

    id: int
    name: str
    price: float
    currency: str = "AED"
    category: str
    brand: str
    reason: Optional[str] = Field(
        None,
        description="Why this product was recommended",
    )


class ChatResponse(BaseModel):
    """
    AI chatbot response returned to the client.

    Attributes:
        reply: The natural-language answer from the AI.
        recommended_products: Optional product suggestions.
        deal: Optional promotional deal or offer mentioned.
        intent: The classified intent of the user's message (Day 4).
    """

    reply: str = Field(
        ...,
        description="AI-generated response text",
    )
    recommended_products: Optional[List[RecommendedProduct]] = Field(
        default=None,
        description="Products the AI thinks the user may be interested in",
    )
    deal: Optional[str] = Field(
        None,
        description="Any special deal or promotion mentioned in the response",
    )
    intent: Optional[str] = Field(
        None,
        description="Classified user intent (e.g. recommend_product, deal_inquiry)",
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence of the intent classification (0.0–1.0)",
    )