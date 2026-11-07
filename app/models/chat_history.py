"""
SQLAlchemy 2.0 model for the chat_history table.

Columns: id (PK), user_id (token subject), message, ai_response,
intent (nullable), created_at (UTC, timezone-aware).
The table is created by the Alembic migration in app/db/migrations.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ChatHistory(Base):  # pylint: disable=too-few-public-methods
    """
    Represents a single turn in a user-AI conversation.

    Attributes:
        id: Auto-incrementing primary key.
        user_id: The verified token subject ("sub" claim) of the sender.
        message: The raw text input from the user.
        ai_response: The text returned by the AI assistant (post-guard).
        intent: Classified intent, one of the 5 valid intents.
        created_at: UTC timestamp when the row was inserted.
    """

    __tablename__ = "chat_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    message: Mapped[str] = mapped_column(Text)
    ai_response: Mapped[str] = mapped_column(Text)
    intent: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
