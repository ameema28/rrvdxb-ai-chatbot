"""
SQLAlchemy model for the chat_history table.

This model maps 1:1 to the schema defined in the project brief:
    id SERIAL PRIMARY KEY, user_id INTEGER, message TEXT,
    ai_response TEXT, intent VARCHAR(100), created_at TIMESTAMP
"""

from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, Text, DateTime

from app.core.database import Base


class ChatHistory(Base):  # pylint: disable=too-few-public-methods
    """
    Represents a single turn in a user-AI conversation.

    Attributes:
        id: Auto-incrementing primary key.
        user_id: Identifies which user sent the message (links to user service).
        message: The raw text input from the user.
        ai_response: The text returned by the AI assistant.
        intent: Classified intent (e.g., "product_inquiry", "shipping_question").
                Nullable until intent classification is implemented.
        created_at: UTC timestamp when the row was inserted.
    """

    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    message = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    intent = Column(String(100), nullable=True, index=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )