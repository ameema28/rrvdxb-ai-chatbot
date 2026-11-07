"""
Alembic migration tests (run on SQLite and, in CI, on PostgreSQL).
"""

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text

from app.core.config import settings
from app.core.database import Base, normalize_database_url
from app.db.migrate import downgrade_to_base, upgrade_to_head
from tests.support import TEST_DATABASE_URL, test_engine

pytestmark = pytest.mark.db


def test_model_and_migration_agree():
    """Autogenerate finds no difference between Base.metadata and the DB."""
    with test_engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == []


def test_upgrade_creates_table_and_indexes():
    insp = inspect(test_engine)
    assert "chat_history" in insp.get_table_names()
    index_names = {ix["name"] for ix in insp.get_indexes("chat_history")}
    assert {"ix_chat_history_user_id", "ix_chat_history_intent"} <= index_names


def test_downgrade_then_upgrade_round_trip():
    with test_engine.begin() as conn:
        downgrade_to_base(connection=conn)
    assert "chat_history" not in inspect(test_engine).get_table_names()
    with test_engine.begin() as conn:
        upgrade_to_head(connection=conn)
    assert "chat_history" in inspect(test_engine).get_table_names()


@pytest.mark.skipif(
    "sslmode=require" not in TEST_DATABASE_URL,
    reason="needs a PostgreSQL TEST_DATABASE_URL with sslmode=require",
)
def test_connection_uses_tls_when_required():
    with test_engine.connect() as conn:
        uses_tls = conn.execute(
            text("SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()")
        ).scalar()
    assert uses_tls is True


def test_upgrade_is_idempotent():
    with test_engine.begin() as conn:
        upgrade_to_head(connection=conn)
        upgrade_to_head(connection=conn)
    assert "chat_history" in inspect(test_engine).get_table_names()


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("postgres://u@h/db", "postgresql+psycopg://u@h/db"),
        ("postgresql://u@h/db", "postgresql+psycopg://u@h/db"),
        ("postgresql+psycopg://u@h/db", "postgresql+psycopg://u@h/db"),
        ("sqlite:///./x.db", "sqlite:///./x.db"),
    ],
)
def test_database_url_normalization(raw, expected):
    assert normalize_database_url(raw) == expected


def test_startup_auto_migrates_a_fresh_sqlite_file(tmp_path, monkeypatch):
    """DB_AUTO_MIGRATE=true (default): app startup creates the schema."""
    from app.main import create_application

    url = f"sqlite:///{(tmp_path / 'fresh.db').as_posix()}"
    monkeypatch.setattr(settings, "database_url", url)
    monkeypatch.setattr(settings, "db_auto_migrate", True)
    with TestClient(create_application()) as c:
        assert c.get("/health").status_code == 200
    engine = create_engine(normalize_database_url(url))
    try:
        assert "chat_history" in inspect(engine).get_table_names()
    finally:
        engine.dispose()


def test_startup_skips_migration_when_disabled(tmp_path, monkeypatch):
    from app.main import create_application

    url = f"sqlite:///{(tmp_path / 'untouched.db').as_posix()}"
    monkeypatch.setattr(settings, "database_url", url)
    monkeypatch.setattr(settings, "db_auto_migrate", False)
    with TestClient(create_application()) as c:
        assert c.get("/health").status_code == 200
    engine = create_engine(url)
    try:
        assert "chat_history" not in inspect(engine).get_table_names()
    finally:
        engine.dispose()
