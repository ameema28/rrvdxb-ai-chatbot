"""
Programmatic Alembic entry points.

The migration scripts live inside the app package (app/db/migrations) so any
image that ships app/ also ships its migrations. The repo-root alembic.ini
points at the same folder for CLI use (`alembic upgrade head`).
"""

from pathlib import Path
from typing import Optional

from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Connection

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def alembic_config(
    url: Optional[str] = None, connection: Optional[Connection] = None
) -> Config:
    """Build an Alembic Config without needing alembic.ini on disk."""
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    if url:
        cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    if connection is not None:
        cfg.attributes["connection"] = connection
    return cfg


def upgrade_to_head(
    url: Optional[str] = None, connection: Optional[Connection] = None
) -> None:
    """Apply every pending migration (idempotent: a no-op when up to date)."""
    command.upgrade(alembic_config(url, connection), "head")


def downgrade_to_base(
    url: Optional[str] = None, connection: Optional[Connection] = None
) -> None:
    """Revert every migration (used by tests)."""
    command.downgrade(alembic_config(url, connection), "base")
