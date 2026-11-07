"""
FastAPI dependency injection functions.

Dependencies are injected into route handlers using FastAPI's Depends().
This promotes testability and keeps route handlers clean.
"""

from typing import Generator, Optional

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.core.security import decode_token
from app.middleware.error_handler import AuthenticationError

# auto_error=False: we raise our own AuthenticationError so the 401 body uses
# the canonical {"error", "detail", "status_code"} shape.
_bearer_scheme = HTTPBearer(auto_error=False)


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
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> str:
    """
    Return the verified user id: the "sub" claim of the HS256 bearer token.

    Any X-User-Id header is ignored; identity comes only from the token.

    Raises:
        AuthenticationError (401): no bearer token, or the token fails
            verification (signature, expiry, issuer, audience, claims).
    """
    if credentials is None:
        raise AuthenticationError(detail="Bearer token missing")
    return decode_token(credentials.credentials)
