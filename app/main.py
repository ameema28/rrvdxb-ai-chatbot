"""
RRVDXB AI Shopping Chatbot — FastAPI application entry point.

This module:
  1. Creates the FastAPI app instance.
  2. Includes API routers.
  3. Creates database tables on startup (convenience for local dev).
"""

from fastapi import FastAPI

from app.core.config import settings
from app.core.database import engine, Base
from app.api.v1.router import api_router


# Create all SQLAlchemy tables if they don't exist.
# In production, use Alembic migrations instead of create_all.
Base.metadata.create_all(bind=engine)


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

    # Mount all v1 routes under /api/v1
    application.include_router(api_router, prefix="/api/v1")

    @application.get("/health", tags=["health"])
    async def health_check() -> dict:
        """Liveness probe for load balancers and monitoring."""
        return {"status": "ok", "service": settings.app_name}

    return application


# The object imported by uvicorn: `uvicorn app.main:app --reload`
app = create_application()