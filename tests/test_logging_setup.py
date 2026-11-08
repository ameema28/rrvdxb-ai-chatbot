"""
configure_logging: INFO records of app loggers are emitted, level is
configurable, the handler is added once, and SQL statements stay off.
"""

import io
import logging

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.logging_setup import LOG_FORMAT, _HANDLER_MARK, configure_logging


@pytest.fixture
def clean_root():
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = list(root.handlers)
    saved_sql = logging.getLogger("sqlalchemy.engine").level
    for h in saved_handlers:
        if getattr(h, _HANDLER_MARK, False):
            root.removeHandler(h)
    yield root
    for h in list(root.handlers):
        if getattr(h, _HANDLER_MARK, False):
            root.removeHandler(h)
    root.setLevel(saved_level)
    logging.getLogger("sqlalchemy.engine").setLevel(saved_sql)


def _ours(root):
    return [h for h in root.handlers if getattr(h, _HANDLER_MARK, False)]


def _capture(root) -> io.StringIO:
    stream = io.StringIO()
    _ours(root)[0].setStream(stream)
    return stream


def test_info_records_are_emitted_with_timestamp_level_and_name(clean_root):
    configure_logging("INFO")
    stream = _capture(clean_root)
    logging.getLogger("app.services.chatbot_service").info("hello %s", "world")
    line = stream.getvalue().strip()
    assert line.endswith("INFO app.services.chatbot_service: hello world")
    assert line[:4].isdigit()
    assert LOG_FORMAT.startswith("%(asctime)s")


def test_warning_level_suppresses_info(clean_root):
    configure_logging("WARNING")
    stream = _capture(clean_root)
    logging.getLogger("app.x").info("quiet")
    logging.getLogger("app.x").warning("loud")
    assert "quiet" not in stream.getvalue()
    assert "loud" in stream.getvalue()


def test_second_call_adds_no_second_handler(clean_root):
    configure_logging("INFO")
    configure_logging("DEBUG")
    assert len(_ours(clean_root)) == 1
    assert clean_root.level == logging.DEBUG


def test_sql_engine_logger_stays_at_warning(clean_root):
    configure_logging("DEBUG")
    assert logging.getLogger("sqlalchemy.engine").level == logging.WARNING


def test_log_level_setting_is_normalised_and_validated():
    assert Settings(log_level="debug").log_level == "DEBUG"
    assert Settings().log_level == "INFO"
    with pytest.raises(ValidationError):
        Settings(log_level="LOUD")
