"""
Application configuration using Pydantic BaseSettings.

BaseSettings automatically loads values from environment variables and .env files.
This is the single source of truth for all configuration in the app.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Centralized configuration for the RRVDXB AI Shopping Chatbot.

    Attributes:
        app_name: Human-readable app name (useful for logging/docs).
        database_url: SQLAlchemy connection string. SQLite for local dev.
        groq_api_key: Secret key for Groq API calls (Groq is the only provider).
        llm_model: Groq model name.
        internal_jwt_secret: Used for signing/verifying internal service tokens.
        debug: Enables FastAPI debug mode and verbose logging.
        rate_limit_per_minute: Per-user chat request budget (sliding window).
    """

    # --- App metadata ---
    app_name: str = Field(default="RRVDXB AI Shopping Chatbot", description="Application display name")

    # --- Database ---
    database_url: str = Field(
        default="sqlite:///./rrvdxb.db",
        description="SQLAlchemy DB URI — SQLite for local development",
    )

    # --- LLM (Groq only) ---
    llm_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Groq model name",
    )
    groq_api_key: str = Field(
        default="",
        description="Groq API key (without it every reply is the fallback text)",
    )

    # --- Security ---
    internal_jwt_secret: str = Field(
        default="change-me-in-production",
        description="Secret for internal JWT signing — MUST be rotated in production",
    )

    # --- Runtime ---
    debug: bool = Field(default=True, description="FastAPI debug mode")

    # --- Performance / NFR (Day 7) ---
    llm_timeout_seconds: float = Field(
        default=3.0,
        description="Hard ceiling for the Groq call (keeps the chat under the 4s NFR)",
    )

    # --- Rate limiting (Day 10: config knob, was a code constant) ---
    rate_limit_per_minute: int = Field(
        default=20,
        description="Per-user chat request budget per minute (sliding window)",
    )

    # Pydantic v2: use SettingsConfigDict instead of nested class Config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra env vars not declared here (prevents crashes)
    )


# Singleton instance — imported everywhere settings are needed.
# This avoids re-parsing the .env file on every import.
settings = Settings()