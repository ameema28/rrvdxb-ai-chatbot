"""Configuration surface tests."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.middleware.error_handler import RateLimitExceeded
from app.middleware.rate_limit import RATE_LIMIT_WINDOW_SECONDS


def test_debug_is_off_by_default():
    assert Settings.model_fields["debug"].default is False


@pytest.mark.parametrize("value", [0, -1])
def test_per_user_rate_limit_must_be_positive(value):
    with pytest.raises(ValidationError):
        Settings(rate_limit_per_minute=value)


def test_retry_after_matches_the_rate_limit_window():
    assert RateLimitExceeded.headers == {"Retry-After": str(int(RATE_LIMIT_WINDOW_SECONDS))}


def test_unused_openai_settings_are_gone():
    """Groq is the only provider: no OpenAI key or provider switch exists."""
    fields = set(Settings.model_fields)
    assert "openai_api_key" not in fields
    assert "llm_provider" not in fields
