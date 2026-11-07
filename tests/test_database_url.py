"""
DATABASE_URL forms used for the live demo: a hosted PostgreSQL URL with TLS
query parameters (e.g. Neon) and the SQLite file in /tmp used by the Docker
image when no DATABASE_URL is set. No database connection is made here.
"""

import pytest

from app.core.database import make_engine, normalize_database_url

SSL_QUERY = "?sslmode=require&channel_binding=require"


@pytest.mark.parametrize("prefix", ["postgres://", "postgresql://", "postgresql+psycopg://"])
def test_ssl_query_params_are_kept(prefix):
    url = f"{prefix}demo_user:pw@ep-example.eu-central-1.aws.example.tech/app{SSL_QUERY}"
    normalized = normalize_database_url(url)
    assert normalized.startswith("postgresql+psycopg://")
    assert normalized.endswith(SSL_QUERY)


def test_psycopg_receives_the_ssl_options():
    engine = make_engine(f"postgresql://demo_user:pw@db.example.test:5432/app{SSL_QUERY}")
    try:
        assert engine.dialect.driver == "psycopg"
        _args, kwargs = engine.dialect.create_connect_args(engine.url)
        assert kwargs["sslmode"] == "require"
        assert kwargs["channel_binding"] == "require"
        assert kwargs["host"] == "db.example.test"
        assert kwargs["dbname"] == "app"
    finally:
        engine.dispose()


def test_docker_default_sqlite_tmp_url_is_unchanged():
    url = "sqlite:////tmp/rrvdxb.db"
    assert normalize_database_url(url) == url
    engine = make_engine(url)
    try:
        assert engine.dialect.name == "sqlite"
        assert engine.url.database == "/tmp/rrvdxb.db"
    finally:
        engine.dispose()
