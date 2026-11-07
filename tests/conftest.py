"""
Shared pytest setup.

Runs before any test module imports the app, so the settings singleton sees
APP_ENV=test and a fixed test signing secret. No real API keys are used:
every LLM call in the suite is mocked.
"""

import os

os.environ["APP_ENV"] = "test"
# Deliberately low-entropy, test-only value (40 chars; not a real secret).
os.environ["INTERNAL_JWT_SECRET"] = "test-secret-" + "x" * 28
os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("DEMO_TOKENS_ENABLED", None)

import pytest  # noqa: E402

from app.middleware.rate_limit import rate_limit_store  # noqa: E402


@pytest.fixture(autouse=True)
def reset_rate_limit_store():
    """Give every test a clean per-user / per-IP budget (module-level singleton)."""
    rate_limit_store.reset()
    yield
    rate_limit_store.reset()
