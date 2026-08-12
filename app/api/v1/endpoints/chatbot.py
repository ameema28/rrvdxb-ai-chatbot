"""
Chatbot API endpoint: POST /api/v1/ai/chat  (PRIVATE — authentication + rate limited)

This module defines the HTTP interface for the AI shopping assistant.
Business logic is delegated to the chatbot_service layer to keep
routes thin and testable.

Day 8 hardening:
  - The endpoint is private: X-User-Id is REQUIRED (401 when missing,
    400 when invalid) via get_current_user_id.
  - Per-user rate limiting (429) via the check_rate_limit dependency.
  - All errors flow to the centralized exception handlers — the route does
    NOT catch-and-silence; it only ever returns a ChatResponse or lets a
    typed error propagate.
"""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_db, get_current_user_id
from app.middleware.rate_limit import check_rate_limit
from app.schemas.chatbot_schema import ChatRequest, ChatResponse
from app.services import chatbot_service

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message to the AI shopping assistant (private)",
    description=(
        "Receives a user message, processes it through the AI pipeline, "
        "and returns a contextual response with optional product "
        "recommendations. Private endpoint: requires the X-User-Id header "
        "(401/400) and is rate-limited to 20 requests per minute per user (429)."
    ),
)
async def chat(
    request: ChatRequest,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    _rate_limit_check: None = Depends(check_rate_limit),
) -> ChatResponse:
    """
    Handle a chat message from an authenticated user.

    Args:
        request: The chat payload containing the user's message.
        user_id: Injected from the X-User-Id header (stub for now;
                 JWT replaces it when the Auth API ships).
        db: SQLAlchemy session injected per-request.
        _rate_limit_check: None — dependency only; enforces the per-user
            budget and raises RateLimitExceeded (429) when exhausted.

    Returns:
        ChatResponse with the AI reply and optional product recommendations.
    """
    # Delegate all business logic to the service layer.
    # The route handler only concerns itself with HTTP semantics. No
    # try/except here: any error propagates to the centralized handlers.
    response = await chatbot_service.handle_chat_message(
        user_id=user_id,
        message=request.message,
        db=db,
    )
    return response