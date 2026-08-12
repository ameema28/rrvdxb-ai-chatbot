"""
Centralized exception types and request handlers for the RRVDXB chatbot.

WHY THIS MODULE EXISTS
----------------------
Scattered try/except blocks produce inconsistent error bodies and risk
leaking stack traces or secrets to the client. Instead, any layer can raise
a typed exception from this module, and ONE registered handler (see
app/main.py) turns it into the canonical response body:

    {"error": "...", "detail": "...", "status_code": <int>}

Handlers never echo tracebacks or API keys to the client; full details are
logged server-side by the error paths below.
"""

import http as http_status
import logging
from typing import Any, Dict

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger(__name__)


class AppError(Exception):
    """
    Base class for every application-level error.

    Subclasses fix `error` (the human title) and `status_code`; callers pass
    only a `detail` message. `to_dict()` produces the canonical JSON body.
    """

    error: str = "Error"
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"{self.error}: {detail}")

    def to_dict(self) -> Dict[str, Any]:
        """The one JSON body every error response uses."""
        return {
            "error": self.error,
            "detail": self.detail,
            "status_code": self.status_code,
        }


class AuthenticationError(AppError):
    """401 — the caller did not prove who they are."""

    error = "Authentication required"
    status_code = status.HTTP_401_UNAUTHORIZED


class ValidationError(AppError):
    """400 — the caller sent something well-formed but invalid (e.g. 'abc' as an id)."""

    error = "Validation error"
    status_code = status.HTTP_400_BAD_REQUEST


class RateLimitExceeded(AppError):
    """429 — the caller exceeded their per-minute budget."""

    error = "Rate limit exceeded"
    status_code = status.HTTP_429_TOO_MANY_REQUESTS


class AIServiceError(AppError):
    """
    502 — the upstream AI provider failed in a non-recoverable way.

    Day 8 note: defined so the service layer CAN raise it for hard upstream
    failures. Today's chatbot_service deliberately degrades gracefully on LLM
    errors/timeouts (HTTP 200 + canned reply per Day 7), so nothing raises
    it yet — but the lever exists for the real integrations.
    """

    error = "AI service error"
    status_code = status.HTTP_502_BAD_GATEWAY


def _status_phrase(status_code: int) -> str:
    """Map 404 -> 'Not Found', 405 -> 'Method Not Allowed', etc."""
    try:
        return http_status.HTTPStatus(status_code).phrase
    except ValueError:
        return "Error"


# ---------------------------------------------------------------------------
# Request handlers — registered in app/main.py via add_exception_handler.
# Each must match the signature: async def (request: Request, exc: X).
# ---------------------------------------------------------------------------


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """
    Single handler for every AppError subclass.

    FastAPI resolves exception handlers by walking the raised exception's MRO,
    so registering this for `AppError` catches AuthenticationError,
    ValidationError, RateLimitExceeded and AIServiceError in one place.
    """
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    FastAPI's built-in 422 (bad request body) -> our 400 shape.

    We flatten only the FIRST validation error into a readable detail string;
    no raw exception internals are ever echoed to the client.
    """
    errors = exc.errors()
    if errors:
        first = errors[0]
        loc = first.get("loc") or []
        msg = first.get("msg", "Invalid request")
        detail = f"Invalid value for {loc[-1]}: {msg}" if loc else msg
    else:
        detail = "Invalid request body"
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Validation error", "detail": detail, "status_code": 400},
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """
    Starlette HTTPExceptions (404, 405, ...) -> the same canonical shape so
    even framework-level errors are consistent with our own.
    """
    phrase = _status_phrase(exc.status_code)
    detail = exc.detail if isinstance(exc.detail, str) else phrase
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": phrase, "detail": detail, "status_code": exc.status_code},
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Last-resort 500 for anything unexpected.

    The full traceback is logged server-side; the client only ever receives
    the generic message. This is the guarantee that stack traces and secrets
    are NEVER exposed in a response.
    """
    logger.error(
        "Unhandled exception on %s %s",
        request.method,
        request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": "Something went wrong",
            "status_code": 500,
        },
    )


# ---------------------------------------------------------------------------
# Catch-all middleware — the DEBUG-proof guarantee for the generic 500.
# ---------------------------------------------------------------------------


class UnhandledErrorMiddleware(BaseHTTPMiddleware):
    """
    Guarantees the canonical JSON 500 body for ANY unexpected exception.

    WHY THIS EXISTS:
      FastAPI routes the handler registered for Exception/500 to Starlette's
      ServerErrorMiddleware. In DEBUG mode (settings.debug=True) that
      middleware returns a raw traceback PlainTextResponse and NEVER calls
      our handler — an explicit "no stack traces in responses" violation.
      TestClient also re-raises the exception by default. Because user
      middleware is mounted INSIDE ServerErrorMiddleware but ABOVE the router,
      this class catches any non-typed exception there, logs the full
      traceback server-side, and returns the canonical JSON body — so
      ServerErrorMiddleware never sees an exception and debug mode can't leak.

    Typed errors (AppError, RequestValidationError, StarletteHTTPException)
    are re-raised so ExceptionMiddleware's registered handlers still emit the
    exact 401/400/404/422 bodies.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        try:
            response = await call_next(request)
        except (AppError, RequestValidationError, StarletteHTTPException):
            raise  # let ExceptionMiddleware's specific handlers respond
        except Exception as exc:
            logger.error(
                "Unhandled exception in %s %s",
                request.method,
                request.url.path,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal server error",
                    "detail": "Something went wrong",
                    "status_code": 500,
                },
            )
        return response