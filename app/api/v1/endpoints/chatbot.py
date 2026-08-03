"""
Chatbot API endpoint: POST /api/ai/chat

This module defines the HTTP interface for the AI shopping assistant.
Business logic is delegated to the chatbot_service layer to keep
routes thin and testable.
"""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_db, get_current_user_id
from app.schemas.chatbot_schema import ChatRequest, ChatResponse
from app.services import chatbot_service

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message to the AI shopping assistant",
    description=(
        "Receives a user message, processes it through the AI pipeline, "
        "and returns a contextual response with optional product recommendations."
    ),
)
async def chat(
    request: ChatRequest,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
) -> ChatResponse:
    """
    Handle a chat message from a user.

    Args:
        request: The chat payload containing the user's message.
        user_id: Injected from the X-User-Id header (stub for now).
        db: SQLAlchemy session injected per-request.

    Returns:
        ChatResponse with the AI reply and optional product recommendations.
    """
    # Delegate all business logic to the service layer.
    # The route handler should only concern itself with HTTP semantics.
    response = await chatbot_service.handle_chat_message(
        user_id=user_id,
        message=request.message,
        db=db,
    )
    return response