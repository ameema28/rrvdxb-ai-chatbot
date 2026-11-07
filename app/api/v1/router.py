"""
API v1 router aggregator.

All v1 endpoints are mounted here so the main app only needs to
include this single router under /api/v1/.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import auth, chatbot

api_router = APIRouter()

# POST /api/v1/auth/demo-token (404 unless DEMO_TOKENS_ENABLED=true)
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])

# Mount the chatbot router under /api/ai/chat
# The full path becomes: /api/v1/ai/chat
api_router.include_router(
    chatbot.router,
    prefix="/ai",
    tags=["ai-chatbot"],
)