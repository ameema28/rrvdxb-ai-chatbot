"""
DB-backed conversation memory for the RRVDXB AI Shopping Chatbot.

This module provides three primitives:
  - save_turn:     Persist one user/AI exchange to chat_history.
  - load_recent_history: Fetch the last N turns for a given user.
  - format_history_for_prompt: Convert raw DB rows into a string the LLM
    can read as prior conversation context.

Why DB-backed memory?
- LLM APIs are stateless. Each call is independent; the model has no
  memory of prior turns unless we explicitly feed them back in.
- In-process memory (dicts, lists) is lost on server restart or when
  requests hit a different worker. A database survives restarts and
  scales across multiple server instances.
- SQLite is sufficient for local dev; PostgreSQL can be swapped in
  production with zero code changes.

Context windowing (limit=N):
- We load only the last N turns, not the full history, because:
  1. Token cost: Every prior turn burns tokens in the prompt.
  2. Latency: Longer prompts increase Groq response time.
  3. Relevance: The most recent 3-5 turns usually contain the context
     needed to answer follow-up questions (e.g., "Something under 500 AED").
- N=5 is chosen as a sweet spot. N<3 loses too much context; N>10
  bloats the prompt with stale information and increases cost/latency.
  Tune N based on observed conversation patterns and token budget.
"""

import logging
from typing import List, Dict, Any

from sqlalchemy.orm import Session

from app.models.chat_history import ChatHistory

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_LIMIT = 5


def save_turn(
    db_session: Session,
    user_id: int,
    message: str,
    ai_response: str,
    intent: str | None = None,
) -> ChatHistory:
    """
    Persist a single conversation turn to the database.

    Args:
        db_session: Active SQLAlchemy session.
        user_id: Authenticated user identifier.
        message: Raw user input.
        ai_response: AI-generated reply.
        intent: Optional intent classification (populated on Day 3+).

    Returns:
        The newly created ChatHistory ORM instance.
    """
    turn = ChatHistory(
        user_id=user_id,
        message=message,
        ai_response=ai_response,
        intent=intent,
    )
    db_session.add(turn)
    db_session.commit()
    db_session.refresh(turn)
    logger.debug("Saved chat turn id=%s for user_id=%s", turn.id, user_id)
    return turn


def load_recent_history(
    db_session: Session,
    user_id: int,
    limit: int = DEFAULT_HISTORY_LIMIT,
) -> List[Dict[str, Any]]:
    """
    Fetch the most recent N conversation turns for a user, newest last.

    Args:
        db_session: Active SQLAlchemy session.
        user_id: User to look up.
        limit: Max number of turns to return (default 5).

    Returns:
        List of dicts, each with keys: role, content, created_at.
        Ordered oldest -> newest so the LLM reads chronologically.
    """
    rows = (
        db_session.query(ChatHistory)
        .filter(ChatHistory.user_id == user_id)
        .order_by(ChatHistory.created_at.desc())
        .limit(limit)
        .all()
    )

    # Reverse so the LLM sees the conversation in chronological order
    rows = list(reversed(rows))

    turns: List[Dict[str, Any]] = []
    for row in rows:
        # Map DB columns to a simple dict shape
        turns.append(
            {
                "role": "user",
                "content": row.message,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
        )
        turns.append(
            {
                "role": "assistant",
                "content": row.ai_response,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
        )

    logger.debug(
        "Loaded %d turns (%d messages) for user_id=%s",
        len(rows),
        len(turns),
        user_id,
    )
    return turns


def format_history_for_prompt(turns: List[Dict[str, Any]]) -> str:
    """
    Convert a list of message dicts into a single string for the LLM prompt.

    Format:
        User: I am looking for a gift.
        Assistant: I'd be happy to help! What occasion is it for?
        User: Something under 500 AED.

    Args:
        turns: Output from load_recent_history().

    Returns:
        Formatted conversation history string, or empty string if no turns.
    """
    if not turns:
        return ""

    lines: List[str] = []
    for turn in turns:
        speaker = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {turn['content']}")

    return "\n".join(lines)