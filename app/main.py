"""
RRVDXB AI Shopping Chatbot — FastAPI application entry point.

This module:
  1. Creates the FastAPI app instance.
  2. Registers centralized exception handlers (Day 8).
  3. Mounts the catch-all error middleware (Day 8).
  4. Includes API routers.
  5. Applies Alembic migrations at startup when DB_AUTO_MIGRATE=true.
  6. Warms up the embedding model and FAQ index when RAG_WARMUP=true.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.ai.chatbot.rag import vector_store
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging_setup import configure_logging
from app.core.security import validate_security_settings
from app.db.migrate import upgrade_to_head
from app.demo_page import DEMO_PAGE_HEADERS, DEMO_PAGE_HTML
from app.middleware.error_handler import (
    AppError,
    UnhandledErrorMiddleware,
    app_error_handler,
    http_exception_handler,
    unhandled_exception_handler,
    validation_error_handler,
)

@asynccontextmanager
async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
    """
    Startup: run `alembic upgrade head` when DB_AUTO_MIGRATE=true (default).

    Idempotent, so single-instance deployments and local dev need no extra
    step. With several instances, set DB_AUTO_MIGRATE=false and run the
    migration once as a release step instead (concurrent upgrades can race).

    Then, when RAG_WARMUP=true (default), load the embedding model and the
    FAQ index and run one query. Uvicorn accepts connections only after this
    returns, so a TCP startup probe waits for a warm app. A failed warm-up
    is logged and the app still starts; /health then reports "degraded".
    """
    configure_logging(settings.log_level)
    if settings.db_auto_migrate:
        upgrade_to_head(settings.database_url)
    if settings.rag_warmup:
        await asyncio.to_thread(vector_store.warm_up)
    yield


def _register_exception_handlers(application: FastAPI) -> None:
    """
    Wire every error-producing path to the canonical JSON shape
    ({"error", "detail", "status_code"}).

    Resolution order note: Starlette/FastAPI pick the MOST SPECIFIC handler
    for the raised exception. AppError covers all our typed errors;
    Exception is the last-resort 500 catch-all (belt) — and the
    UnhandledErrorMiddleware below is the guarantee (suspenders), because
    ServerErrorMiddleware bypasses the Exception handler when DEBUG=True.
    """
    application.add_exception_handler(AppError, app_error_handler)
    application.add_exception_handler(
        RequestValidationError, validation_error_handler
    )
    application.add_exception_handler(
        StarletteHTTPException, http_exception_handler
    )
    application.add_exception_handler(Exception, unhandled_exception_handler)


def create_application() -> FastAPI:
    """
    Application factory pattern.

    Using a factory makes testing easier because we can create
    isolated app instances with overridden dependencies.

    Raises:
        RuntimeError: INTERNAL_JWT_SECRET shorter than 32 characters outside
            APP_ENV=test (the app refuses to start).
    """
    validate_security_settings()

    application = FastAPI(
        title=settings.app_name,
        description="AI-powered shopping assistant for RRVDXB premium e-commerce.",
        version="0.1.0",
        debug=settings.debug,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    _register_exception_handlers(application)

    # Catch anything that slips past the typed handlers — immune to DEBUG
    # mode and TestClient's raise_server_exceptions. Mounts INSIDE
    # ServerErrorMiddleware (see error_handler.py docstring).
    application.add_middleware(UnhandledErrorMiddleware)

    # Mount all v1 routes under /api/v1
    application.include_router(api_router, prefix="/api/v1")

    @application.get("/", include_in_schema=False, response_class=HTMLResponse)
    async def demo_page() -> HTMLResponse:
        """Public demo chat page (same origin; not rate limited)."""
        return HTMLResponse(DEMO_PAGE_HTML, headers=DEMO_PAGE_HEADERS)

    @application.get("/health", tags=["health"])
    async def health_check() -> dict:
        """
        Liveness probe (always HTTP 200) with RAG readiness.

        rag_ready: the embedding model and FAQ index answered a warm-up query.
        status: "degraded" when warm-up is enabled but did not succeed (FAQ
        answers then use the no-match path), otherwise "ok".
        """
        rag_ready = vector_store.is_ready()
        degraded = settings.rag_warmup and not rag_ready
        return {
            "status": "degraded" if degraded else "ok",
            "service": settings.app_name,
            "rag_ready": rag_ready,
        }

    return application


# The object imported by uvicorn: `uvicorn app.main:app --reload`
app = create_application()