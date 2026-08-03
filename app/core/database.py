"""
SQLAlchemy database setup for local development with SQLite.

This module creates:
  - engine:    The connection pool to the database.
  - SessionLocal: A factory for creating new DB sessions (used per-request).
  - Base:      The declarative base class all models inherit from.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import settings

# SQLite-specific: check_same_thread=False allows the same connection
# to be used across different threads (safe for FastAPI's async workers).
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
    echo=settings.debug,  # Logs every SQL statement when debug=True
)

# SessionLocal is a factory — call it to get a new session object.
# autocommit=False means we must explicitly call session.commit().
# autoflush=False delays sending SQL to the DB until commit() or explicit flush().
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# All SQLAlchemy models must inherit from this Base.
Base = declarative_base()