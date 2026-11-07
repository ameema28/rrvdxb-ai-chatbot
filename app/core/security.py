"""
HS256 bearer-token signing and verification.

- Tokens are signed with INTERNAL_JWT_SECRET (HS256 only; any other
  algorithm, including "none", is rejected).
- Required claims: sub, iat, exp, iss, aud. The user id is the "sub" claim.
- Errors never echo the token or the secret: every failure becomes the same
  AuthenticationError (401) with a fixed detail message.
"""

import re
import time
import uuid

import jwt

from app.core.config import Settings, settings
from app.middleware.error_handler import AuthenticationError

ALGORITHM = "HS256"
MIN_SECRET_LENGTH = 32
_REQUIRED_CLAIMS = ["sub", "iat", "exp", "iss", "aud"]
_SUB_PATTERN = re.compile(r"[A-Za-z0-9_.:-]{1,64}")
_INVALID_TOKEN = "Invalid or expired token"


def validate_security_settings(cfg: Settings = settings) -> None:
    """
    Refuse to start with a weak signing secret.

    Outside APP_ENV=test the secret must be at least 32 characters. The
    error names the requirement only, never the value.
    """
    if cfg.app_env == "test":
        return
    if len(cfg.internal_jwt_secret.get_secret_value()) < MIN_SECRET_LENGTH:
        raise RuntimeError(
            f"INTERNAL_JWT_SECRET must be at least {MIN_SECRET_LENGTH} characters"
        )


def _secret() -> str:
    return settings.internal_jwt_secret.get_secret_value()


def create_token(sub: str, ttl_seconds: int) -> str:
    """Sign a token for `sub` that expires after `ttl_seconds`."""
    now = int(time.time())
    payload = {
        "sub": sub,
        "iat": now,
        "exp": now + ttl_seconds,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
    }
    return jwt.encode(payload, _secret(), algorithm=ALGORITHM)


def new_demo_subject() -> str:
    """A random, unguessable subject for an anonymous demo visitor."""
    return f"demo-{uuid.uuid4().hex}"


def decode_token(token: str) -> str:
    """
    Verify `token` and return its subject.

    Raises:
        AuthenticationError: bad signature, wrong algorithm, expired, wrong
            issuer/audience, missing claims, or a malformed subject.
    """
    secret = _secret()
    if not secret:
        raise AuthenticationError(detail=_INVALID_TOKEN)
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=[ALGORITHM],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            options={"require": _REQUIRED_CLAIMS},
        )
    except jwt.PyJWTError:
        raise AuthenticationError(detail=_INVALID_TOKEN) from None
    sub = claims.get("sub")
    if not isinstance(sub, str) or not _SUB_PATTERN.fullmatch(sub):
        raise AuthenticationError(detail=_INVALID_TOKEN)
    return sub
