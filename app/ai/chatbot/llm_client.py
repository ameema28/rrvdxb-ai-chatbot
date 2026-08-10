"""
Groq LLM client wrapper for the RRVDXB AI Shopping Chatbot.

Provides a single entry point for sending chat completion requests to Groq.
Handles client initialization, error handling, and response extraction.

Day 6: stays a THIN Groq wrapper. Prompt engineering (including the RAG
prompt) lives in prompts.py — this module has no RAG-specific logic.
"""

import logging
from typing import Optional

from groq import Groq

from app.core.config import settings

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Singleton Groq client
# --------------------------------------------------------------------------
# We cache the client at module level to reuse HTTP connections across
# requests. Creating a new Groq() instance on every call would trigger a
# fresh TCP/TLS handshake (~100-300ms overhead per request).
# --------------------------------------------------------------------------

_groq_client: Optional[Groq] = None


def _get_groq_client() -> Groq:
    """Return the cached Groq client, creating it if necessary."""
    global _groq_client
    if _groq_client is None:
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Please configure it in your .env file."
            )
        _groq_client = Groq(api_key=settings.groq_api_key)
        logger.info("Groq client initialized (model: %s)", settings.llm_model)
    return _groq_client


def send_chat_message(
    system_prompt: str,
    user_message: str,
    product_context: str,
    history: str = "",
    system_prompt_override: Optional[str] = None,
) -> str:
    """
    Send a chat completion request to Groq and return the assistant's reply.

    Day 3 update: Accepts an optional `history` string containing prior
    conversation turns. The history is injected into the system prompt so
    the model has full conversational context.

    Day 6 update: Accepts an optional `system_prompt_override`. When set, it
    REPLACES `system_prompt` entirely and SKIPS the product-catalog injection
    (the override is a complete, final prompt built in prompts.py, e.g. the
    RAG-grounded prompt). History is still appended so the RAG flow keeps
    conversation memory. The client stays a THIN Groq wrapper — no RAG logic.

    Args:
        system_prompt: The guardrailed persona and scope instructions.
        user_message: The customer's raw query.
        product_context: Formatted product catalog data injected into context.
        history: Formatted prior conversation turns (empty string if none).
        system_prompt_override: Optional final prompt that fully replaces
            `system_prompt` (e.g. build_rag_system_prompt() output).

    Returns:
        Clean string response from the LLM.

    Raises:
        RuntimeError: If the Groq API key is missing or the API call fails.
    """
    client = _get_groq_client()

    # An override is taken verbatim — it is already the complete final prompt.
    # Otherwise compose base prompt + catalog + history exactly as before,
    # so every existing caller (intent.py, general chat) is unaffected.
    if system_prompt_override is not None:
        full_system = system_prompt_override
    else:
        full_system = system_prompt
        if product_context:
            full_system += f"\n\nCURRENT PRODUCT CATALOG:\n{product_context}"

    # History is appended for ALL paths — memory survives the RAG flow too.
    if history:
        full_system += f"\n\nPRIOR CONVERSATION:\n{history}"

    messages = [
        {"role": "system", "content": full_system},
        {"role": "user", "content": user_message},
    ]

    try:
        completion = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.7,
            max_tokens=500,
            top_p=0.9,
        )
    except Exception as exc:
        # Groq SDK raises various exceptions (AuthenticationError, RateLimitError,
        # APIConnectionError, APIError, APITimeoutError). We catch them all,
        # log the technical details, and raise a user-friendly RuntimeError.
        logger.error("Groq API call failed (%s): %s", type(exc).__name__, exc)
        raise RuntimeError("AI service is temporarily unavailable. Please try again.") from exc

    # Extract the assistant's reply. choices[0] is safe because we use n=1 (default).
    reply = completion.choices[0].message.content
    if reply is None:
        logger.warning("Groq returned empty content")
        return "I'm sorry, I didn't catch that. Could you rephrase your question?"

    # Strip leading/trailing whitespace for cleanliness
    return reply.strip()