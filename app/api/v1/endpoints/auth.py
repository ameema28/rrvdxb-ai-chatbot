"""
POST /api/v1/auth/demo-token: anonymous token for the public demo (opt-in).

When DEMO_TOKENS_ENABLED is false (the default) the route answers 404, the
same as an unknown path. When enabled, each call issues a short-lived token
whose "sub" is a fresh random id, so every visitor gets their own empty
memory and their own chat rate-limit budget. Issuance is rate-limited per
client IP.

Known limit: behind a reverse proxy, request.client.host is the proxy's
address, so the per-IP limit becomes one shared limit.
"""

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from app.core.config import settings
from app.core.security import create_token, new_demo_subject
from app.middleware.error_handler import RateLimitExceeded
from app.middleware.rate_limit import RATE_LIMIT_WINDOW_SECONDS, rate_limit_store

router = APIRouter()


class DemoTokenResponse(BaseModel):
    """Bearer token for an anonymous demo visitor."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


@router.post(
    "/demo-token",
    response_model=DemoTokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Issue an anonymous demo token (only when DEMO_TOKENS_ENABLED=true)",
)
async def issue_demo_token(request: Request) -> DemoTokenResponse:
    if not settings.demo_tokens_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    client_ip = request.client.host if request.client else "unknown"
    result = await rate_limit_store.allow_request(
        key=f"ip:{client_ip}",
        limit=settings.demo_token_rate_per_minute,
        window_seconds=RATE_LIMIT_WINDOW_SECONDS,
    )
    if not result.allowed:
        raise RateLimitExceeded(
            detail=f"{settings.demo_token_rate_per_minute} demo tokens per minute allowed"
        )

    ttl = settings.demo_token_ttl_seconds
    return DemoTokenResponse(
        access_token=create_token(new_demo_subject(), ttl),
        expires_in=ttl,
    )
