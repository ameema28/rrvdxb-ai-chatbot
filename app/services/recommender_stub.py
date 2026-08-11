"""Stub recommendation service for the RRVDXB chatbot.

Temporary placeholder that lets the chatbot populate
``ChatResponse.recommended_products`` while the recommendations team builds the
real service. The real recommender will be dropped in behind this same function
signature so the chatbot never changes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Product IDs shipped in the mock catalog (used by the intent classifiers).
VALID_PRODUCT_IDS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

#: Candidate locations for the mock product catalog, tried in order.
_CANDIDATE_PATHS: tuple[Path, ...] = (
    Path(__file__).resolve().parents[2] / "mock_data" / "products.json",
    Path(__file__).resolve().parents[1] / "mock_data" / "products.json",
    Path("mock_data") / "products.json",
    Path("products.json"),
)


def _find_products_file() -> Path | None:
    """Return the first existing catalog path, or None."""
    for candidate in _CANDIDATE_PATHS:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file():
            return resolved
    return None


def _load_products() -> list[dict[str, Any]]:
    """Load the mock catalog defensively.

    Any failure (missing file, bad JSON, wrong shape) returns ``[]`` so the
    chatbot always has something safe to render; this never raises into the
    request path.
    """
    products_file = _find_products_file()
    if products_file is None:
        logger.warning("product catalog not found; tried %s", _CANDIDATE_PATHS)
        return []

    try:
        payload = json.loads(products_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("could not load product catalog: %r", products_file)
        return []

    products = payload if isinstance(payload, list) else payload.get("products", [])
    if not isinstance(products, list):
        return []
    return [p for p in products if isinstance(p, dict)]


def _keywords(message: str) -> list[str]:
    """Extract searchable, plural-normalized keyword tokens from a message."""
    words: list[str] = []
    for raw in message.lower().split():
        token = "".join(ch for ch in raw if ch.isalnum())
        if len(token) < 3:
            continue
        if token.endswith("s") and len(token) > 3:
            token = token[:-1]
        if token not in words:
            words.append(token)
    return words


def get_recommendations(user_id: int, message: str) -> list[dict[str, Any]]:
    """Return up to three mock products matching the user's request.

    A message keyword counts when it equals a product's name/brand/category
    word or appears as that word's prefix/suffix. This keeps "phone" ->
    "iPhone" while rejecting substring traps like "phone" inside
    "headphones".
    """
    tokens = _keywords(message)
    if not tokens:
        return []

    scored: list[tuple[int, dict[str, Any], list[str]]] = []
    for product in _load_products():
        try:
            product_id = int(product["id"])
        except (KeyError, TypeError, ValueError):
            continue

        name = str(product.get("name", "")).lower()
        brand = str(product.get("brand", "")).lower()
        category = str(product.get("category", "")).lower()
        words = f"{name} {brand} {category}".split()

        hits = [
            token
            for token in tokens
            if any(w == token or w.endswith(token) or w.startswith(token) for w in words)
        ]
        if hits:
            scored.append((product_id, product, hits))

    scored.sort(key=lambda entry: -len(entry[2]))

    recommendations = []
    for product_id, product, hits in scored[:3]:
        try:
            price = float(product["price"])
        except (KeyError, TypeError, ValueError):
            price = 0.0

        recommendations.append(
            {
                "id": product_id,
                "name": product.get("name", ""),
                "price": price,
                "currency": "AED",
                "category": product.get("category", ""),
                "brand": product.get("brand", ""),
                "reason": f"Matches your request for {hits[0]}",
            }
        )
    return recommendations