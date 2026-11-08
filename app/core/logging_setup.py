"""
Minimal application logging: one stderr handler on the root logger.

Uvicorn configures only its own loggers, so without this the app's INFO
records are dropped. Log calls in the app never include message text, tokens
or keys (tests/test_logging_privacy.py).
"""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

_HANDLER_MARK = "_rrvdxb_handler"


def configure_logging(level: str) -> None:
    """
    Attach one stream handler to the root logger and set its level.

    Idempotent: a second call only updates the level. sqlalchemy.engine is
    kept at WARNING so SQL statements are never logged at INFO.
    """
    root = logging.getLogger()
    root.setLevel(level)
    if not any(getattr(h, _HANDLER_MARK, False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        setattr(handler, _HANDLER_MARK, True)
        root.addHandler(handler)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
