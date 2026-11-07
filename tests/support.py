"""Helpers shared by test modules (tokens, auth headers, the test DB)."""

import os
import time
from typing import Any, Dict, Optional

import jwt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.database import Base
from app.core.dependencies import get_db
from app.main import app

TEST_SECRET = os.environ["INTERNAL_JWT_SECRET"]


def make_token(
    sub: Optional[str],
    *,
    exp_delta: int = 3600,
    secret: Optional[str] = None,
    algorithm: str = "HS256",
    **overrides: Any,
) -> str:
    """Sign a token like the app does; overrides set or (with None) drop claims."""
    now = int(time.time())
    payload: Dict[str, Any] = {
        "sub": sub,
        "iat": now,
        "exp": now + exp_delta,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
    }
    payload.update(overrides)
    payload = {k: v for k, v in payload.items() if v is not None}
    return jwt.encode(payload, secret or TEST_SECRET, algorithm=algorithm)


def auth_headers(sub: Any) -> Dict[str, str]:
    """Authorization header carrying a valid token for `sub`."""
    return {"Authorization": f"Bearer {make_token(str(sub))}"}


# --------------------------------------------------------------------------
# In-memory SQLite DB shared by the endpoint tests (isolated, no file I/O)
# --------------------------------------------------------------------------
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
Base.metadata.create_all(bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
