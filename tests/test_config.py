"""Configuration surface tests."""

from app.core.config import Settings


def test_unused_openai_settings_are_gone():
    """Groq is the only provider: no OpenAI key or provider switch exists."""
    fields = set(Settings.model_fields)
    assert "openai_api_key" not in fields
    assert "llm_provider" not in fields
