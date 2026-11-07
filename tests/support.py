"""Helpers shared by test modules (tokens, auth headers, the test DB)."""

import os
import time
from typing import Any, Dict, Optional

import jwt
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.database import make_engine
from app.core.dependencies import get_db
from app.db.migrate import upgrade_to_head
from app.main import app
from app.models.chat_history import ChatHistory

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
# Test database: in-memory SQLite by default, or TEST_DATABASE_URL (e.g. a
# PostgreSQL service container). The schema is built by the Alembic
# migration, not create_all, so every DB test also exercises the migration.
# --------------------------------------------------------------------------
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "sqlite:///:memory:")
IS_SQLITE = TEST_DATABASE_URL.startswith("sqlite")

if TEST_DATABASE_URL == "sqlite:///:memory:":
    test_engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
else:
    test_engine = make_engine(TEST_DATABASE_URL)

with test_engine.begin() as _conn:
    upgrade_to_head(connection=_conn)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def clear_chat_history() -> None:
    """Empty chat_history so tests (and repeated Postgres runs) start clean."""
    with test_engine.begin() as conn:
        conn.execute(delete(ChatHistory))


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
