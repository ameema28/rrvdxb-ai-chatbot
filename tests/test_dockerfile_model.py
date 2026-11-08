"""
Dockerfile layer order: the embedding model is downloaded before app code is
copied, and the name matches the one the app loads.
"""

import re
from pathlib import Path

from app.ai.chatbot.rag import vector_store

DOCKERFILE = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(encoding="utf-8")


def _instructions() -> list[str]:
    return [ln for ln in DOCKERFILE.splitlines() if ln and not ln.lstrip().startswith("#")]


def test_dockerfile_downloads_the_model_the_app_loads():
    match = re.search(r"SentenceTransformer\('([^']+)'\)", DOCKERFILE)
    assert match is not None
    assert match.group(1) == vector_store._MODEL_NAME


def test_model_download_comes_before_app_code_is_copied():
    lines = _instructions()
    download = next(i for i, ln in enumerate(lines) if "SentenceTransformer(" in ln)
    copy_app = next(i for i, ln in enumerate(lines) if ln.startswith("COPY") and " app " in ln)
    build_index = next(i for i, ln in enumerate(lines) if "build_vector_store.py" in ln and ln.startswith("RUN"))
    assert download < copy_app < build_index


def test_index_build_runs_offline():
    lines = _instructions()
    offline = next(i for i, ln in enumerate(lines) if "HF_HUB_OFFLINE=1" in ln)
    build_index = next(i for i, ln in enumerate(lines) if ln.startswith("RUN") and "build_vector_store.py" in ln)
    download = next(i for i, ln in enumerate(lines) if "SentenceTransformer(" in ln)
    assert download < offline < build_index
