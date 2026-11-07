"""
SQLAlchemy 2.0 database setup: SQLite (dev/tests) or PostgreSQL (psycopg 3).

This module creates:
  - engine:       The connection pool for settings.database_url.
  - SessionLocal: A factory for creating new DB sessions (used per-request).
  - Base:         The declarative base class all models inherit from.

The schema is owned by Alembic (app/db/migrations), not by create_all.
"""

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings

_PSYCOPG_PREFIX = "postgresql+psycopg://"


def normalize_database_url(url: str) -> str:
    """
    Map plain PostgreSQL URLs to the psycopg 3 driver.

    Hosting providers often hand out postgres:// or postgresql:// URLs;
    SQLAlchemy would pick psycopg2 (not installed) for those.
    """
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return _PSYCOPG_PREFIX + url[len(prefix):]
    return url


def make_engine(url: str) -> Engine:
    """
    Create an engine for `url`.

    SQLite: check_same_thread=False lets FastAPI's worker threads share
    connections. echo stays False: SQL echo logs bound parameters, which
    would put raw user messages into the logs.
    """
    url = normalize_database_url(url)
    is_sqlite = url.startswith("sqlite")
    return create_engine(
        url,
        connect_args={"check_same_thread": False} if is_sqlite else {},
        pool_pre_ping=not is_sqlite,
        echo=False,
    )


engine = make_engine(settings.database_url)

# autocommit=False: callers commit explicitly. autoflush=False delays SQL
# until commit() or an explicit flush().
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Declarative base for every ORM model."""
