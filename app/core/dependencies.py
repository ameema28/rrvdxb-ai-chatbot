"""
FastAPI dependency injection functions.

Dependencies are injected into route handlers using FastAPI's Depends().
This promotes testability and keeps route handlers clean.
"""

from typing import Generator, Optional

from fastapi import Header, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import SessionLocal


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
    x_user_id: Optional[int] = Header(None, alias="X-User-Id", description="User ID from client/gateway")
) -> int:
    """
    STUB: Extract the current user ID from the X-User-Id header.

    In production, this will be replaced by a real JWT verification layer
    that decodes an Authorization token and returns the authenticated user's ID.

    Args:
        x_user_id: Integer user ID passed in the X-User-Id header.

    Returns:
        The authenticated user's ID.

    Raises:
        HTTPException(401): If the header is missing.
    """
    if x_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-User-Id header. Authentication required.",
        )
    return x_user_id