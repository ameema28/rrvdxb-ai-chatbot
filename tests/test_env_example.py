"""Repo hygiene: .env.example matches the settings exactly; test artifacts are ignored."""

import re
from pathlib import Path

from app.core.config import Settings

ROOT = Path(__file__).resolve().parents[1]


def _env_example_keys() -> list:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    return re.findall(r"^([A-Z][A-Z0-9_]*)=", text, flags=re.MULTILINE)


def test_env_example_keys_match_settings_exactly():
    keys = _env_example_keys()
    assert len(keys) == len(set(keys)), "duplicate key in .env.example"
    assert set(keys) == {name.upper() for name in Settings.model_fields}


def test_env_example_has_no_real_secret():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert re.search(r"^INTERNAL_JWT_SECRET=$", text, flags=re.MULTILINE)
    assert "APP_ENV=development" in text
    assert "secrets.token_urlsafe(48)" in text


def test_gitignore_covers_test_and_lint_artifacts():
    lines = {line.strip() for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()}
    for pattern in (".pytest_cache/", ".coverage", "coverage.xml", "htmlcov/", ".ruff_cache/"):
        assert pattern in lines
