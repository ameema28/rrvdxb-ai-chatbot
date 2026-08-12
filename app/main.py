"""
RRVDXB AI Shopping Chatbot — FastAPI application entry point.

This module:
  1. Creates the FastAPI app instance.
  2. Registers centralized exception handlers (Day 8).
  3. Mounts the catch-all error middleware (Day 8).
  4. Includes API routers.
  5. Creates database tables on startup (convenience for local dev).
"""

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.database import Base, engine
from app.middleware.error_handler import (
    AppError,
    UnhandledErrorMiddleware,
    app_error_handler,
    http_exception_handler,
    unhandled_exception_handler,
    validation_error_handler,
)

# Create all SQLAlchemy tables if they don't exist.
# In production, use Alembic migrations instead of create_all.
Base.metadata.create_all(bind=engine)


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
    """
    application = FastAPI(
        title=settings.app_name,
        description="AI-powered shopping assistant for RRVDXB premium e-commerce.",
        version="0.1.0",
        debug=settings.debug,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    _register_exception_handlers(application)

    # Catch anything that slips past the typed handlers — immune to DEBUG
    # mode and TestClient's raise_server_exceptions. Mounts INSIDE
    # ServerErrorMiddleware (see error_handler.py docstring).
    application.add_middleware(UnhandledErrorMiddleware)

    # Mount all v1 routes under /api/v1
    application.include_router(api_router, prefix="/api/v1")

    @application.get("/health", tags=["health"])
    async def health_check() -> dict:
        """Liveness probe for load balancers and monitoring."""
        return {"status": "ok", "service": settings.app_name}

    return application


# The object imported by uvicorn: `uvicorn app.main:app --reload`
app = create_application()