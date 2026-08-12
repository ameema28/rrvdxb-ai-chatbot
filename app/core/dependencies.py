"""
FastAPI dependency injection functions.

Dependencies are injected into route handlers using FastAPI's Depends().
This promotes testability and keeps route handlers clean.
"""

from typing import Generator, Optional

from fastapi import Header
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.middleware.error_handler import AuthenticationError, ValidationError


def get_db() -> Generator[Session, None, None]:
    """
    Yield a database session for the lifetime of a single request.

    FastAPI calls this automatically for any endpoint that declares
    `db: Session = Depends(get_db)`. The session is closed after the
    response is sent, even if an exception occurs.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_user_id(
    x_user_id: Optional[str] = Header(
        default=None,
        alias="X-User-Id",
        description="User ID from client/gateway (STUB — replaced by JWT later)",
    )
) -> int:
    """
    STUB: Extract and validate the current user ID from the X-User-Id header.

    In production this will be replaced by real JWT verification that decodes
    an Authorization token and returns the authenticated user's ID from the
    token's `sub` claim. Because every route depends on this single function,
    that swap touches NO route code.

    Missing header -> AuthenticationError (401)
    Non-integer     -> ValidationError   (400)

    Returns:
        The authenticated user's ID as an int.

    Raises:
        AuthenticationError: If the header is missing.
        ValidationError:     If the header is not a valid integer.
    """
    # TODO: Replace with JWT validation when Auth API is ready
    if x_user_id is None:
        raise AuthenticationError(detail="X-User-Id header missing")
    try:
        return int(x_user_id)
    except (TypeError, ValueError):
        raise ValidationError(detail="X-User-Id must be an integer")