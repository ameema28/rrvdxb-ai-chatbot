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
os.environ.pop("TRUST_PROXY", None)
os.environ.pop("TRUST_PROXY_HOPS", None)
# Every TestClient request comes from one IP; a high per-IP chat cap keeps
# multi-user tests independent. tests/test_client_ip.py lowers it explicitly.
os.environ["CHAT_IP_RATE_PER_MINUTE"] = "1000"
# No model loading at app startup in tests; tests/test_warmup.py covers it.
os.environ["RAG_WARMUP"] = "false"
os.environ.pop("RAG_SIMILARITY_THRESHOLD", None)

import pytest  # noqa: E402

from app.middleware.rate_limit import rate_limit_store  # noqa: E402
from tests.support import clear_chat_history  # noqa: E402


@pytest.fixture(autouse=True)
def clean_database():
    """Every test starts with an empty chat_history table."""
    clear_chat_history()
    yield


@pytest.fixture(autouse=True)
def reset_rate_limit_store():
    """Give every test a clean per-user / per-IP budget (module-level singleton)."""
    rate_limit_store.reset()
    yield
    rate_limit_store.reset()
