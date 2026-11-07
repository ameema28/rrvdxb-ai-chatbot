"""
Alembic environment.

URL resolution order: an injected connection (config.attributes["connection"],
used by tests), then sqlalchemy.url from the Alembic config, then
settings.database_url. Logging is deliberately NOT reconfigured here, so
running migrations at app startup leaves the app's logging untouched.
"""

from alembic import context
from sqlalchemy.engine import Connection

import app.models.chat_history  # noqa: F401  (registers the model on Base.metadata)
from app.core.config import settings
from app.core.database import Base, make_engine, normalize_database_url

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    return normalize_database_url(
        config.get_main_option("sqlalchemy.url") or settings.database_url
    )


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=connection.dialect.name == "sqlite",
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    injected = config.attributes.get("connection")
    if injected is not None:
        _run(injected)
        return
    engine = make_engine(_url())
    try:
        with engine.begin() as connection:
            _run(connection)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
