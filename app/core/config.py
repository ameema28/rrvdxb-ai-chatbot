"""
Application configuration using Pydantic BaseSettings.

BaseSettings automatically loads values from environment variables and .env files.
This is the single source of truth for all configuration in the app.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr


class Settings(BaseSettings):
    """
    Centralized configuration for the RRVDXB AI Shopping Chatbot.

    Attributes:
        app_name: Human-readable app name (useful for logging/docs).
        database_url: SQLAlchemy connection string. SQLite for local dev.
        groq_api_key: Secret key for Groq API calls (Groq is the only provider).
        llm_model: Groq model name.
        app_env: development / test / production.
        internal_jwt_secret: HS256 secret used to sign and verify bearer tokens.
        jwt_issuer / jwt_audience: Required 'iss' / 'aud' claim values.
        demo_tokens_enabled: Opt-in anonymous demo-token endpoint.
        debug: Enables FastAPI debug mode (SQL echo is always off).
        rate_limit_per_minute: Per-user chat request budget (sliding window).
    """

    # --- App metadata ---
    app_name: str = Field(default="RRVDXB AI Shopping Chatbot", description="Application display name")

    # --- Database ---
    database_url: str = Field(
        default="sqlite:///./rrvdxb.db",
        description="SQLAlchemy DB URI: SQLite, or PostgreSQL (postgres://, postgresql:// "
        "and postgresql+psycopg:// all use psycopg 3)",
    )
    db_auto_migrate: bool = Field(
        default=True,
        description="Run `alembic upgrade head` at startup (set false for multi-instance prod)",
    )

    # --- LLM (Groq only) ---
    llm_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Groq model name",
    )
    groq_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Groq API key (without it every reply is the fallback text)",
    )

    # --- Environment ---
    app_env: Literal["development", "test", "production"] = Field(
        default="production",
        description="Runtime environment; only 'test' relaxes the JWT secret length check",
    )

    # --- Security (HS256 bearer tokens) ---
    internal_jwt_secret: SecretStr = Field(
        default=SecretStr(""),
        description="HS256 signing secret; at least 32 characters outside APP_ENV=test",
    )
    jwt_issuer: str = Field(default="rrvdxb-chatbot", description="Required 'iss' claim")
    jwt_audience: str = Field(default="rrvdxb-chat", description="Required 'aud' claim")

    # --- Public demo tokens (opt-in) ---
    demo_tokens_enabled: bool = Field(
        default=False,
        description="Expose POST /api/v1/auth/demo-token (404 when false)",
    )
    demo_token_ttl_seconds: int = Field(
        default=1800, gt=0, description="Lifetime of an anonymous demo token"
    )
    demo_token_rate_per_minute: int = Field(
        default=5, gt=0, description="Demo tokens issued per client IP per minute"
    )

    # --- Runtime ---
    debug: bool = Field(default=True, description="FastAPI debug mode")
    rag_warmup: bool = Field(
        default=True,
        description="Load the embedding model and FAQ index at startup (reported in /health)",
    )

    # --- Performance / NFR (Day 7) ---
    # One total budget per chat request; each blocking step also has its own
    # cap and gets min(cap, time left in the budget).
    request_budget_seconds: float = Field(
        default=4.0, gt=0, description="Total time budget for one chat request"
    )
    intent_timeout_seconds: float = Field(
        default=1.0, gt=0, description="Cap for intent classification (LLM fallback)"
    )
    retrieval_timeout_seconds: float = Field(
        default=1.0, gt=0, description="Cap for FAQ retrieval (embedding + ChromaDB)"
    )
    llm_timeout_seconds: float = Field(
        default=3.0,
        gt=0,
        description="Cap for the reply LLM call (also the Groq client timeout)",
    )

    # --- Rate limiting (Day 10: config knob, was a code constant) ---
    rate_limit_per_minute: int = Field(
        default=20,
        description="Per-user chat request budget per minute (sliding window)",
    )
    chat_ip_rate_per_minute: int = Field(
        default=20,
        gt=0,
        description="Chat requests per client IP per minute, on top of the per-user limit",
    )

    # --- Client IP behind a reverse proxy ---
    trust_proxy: bool = Field(
        default=False,
        description="Take the client IP from X-Forwarded-For (only behind a proxy that appends it)",
    )
    trust_proxy_hops: int = Field(
        default=1,
        ge=1,
        description="With trust_proxy: use the X-Forwarded-For entry this many places from the right",
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