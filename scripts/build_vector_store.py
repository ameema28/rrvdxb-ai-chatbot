"""
One-shot build script for the RRVDXB FAQ vector store (Day 5).

Run from the project root:

    python scripts/build_vector_store.py                          # build only
    python scripts/build_vector_store.py --query "do you ship to Pakistan?"   # build + demo search

Fully standalone — does NOT start FastAPI. Safe to re-run (idempotent).
"""

import argparse
import logging
import sys
from pathlib import Path

# scripts/build_vector_store.py -> parents[1] is the project root.
# Insert it into sys.path so `from app...` resolves when run as a plain script.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import AFTER the sys.path fix so `python scripts/...` works from anywhere.
from app.ai.chatbot.rag.vector_store import build_vector_store, search_faqs  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("build_vector_store")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the RRVDXB FAQ vector store (ChromaDB + sentence-transformers)."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional demo query: after building, run a similarity search and print hits.",
    )
    args = parser.parse_args()

    logger.info("Building vector store from app/mock_data/faqs.json ...")
    count = build_vector_store()

    if count == 0:
        logger.error("Nothing embedded. Check faqs.json exists and is valid JSON.")
        return 1

    print(
        f"\n=== SUCCESS ===\n"
        f"Embedded {count} FAQ chunk(s) into app/ai/chatbot/rag/chroma_db/\n"
    )

    if args.query:
        logger.info("Demo similarity query: %r", args.query)
        hits = search_faqs(args.query, top_k=3)
        if not hits:
            print("No results returned (store is empty).")
        for i, hit in enumerate(hits, start=1):
            print(f"[{i}] similarity={hit['similarity']:.3f}")
            print(f"    Q: {hit['question']}")
            print(f"    A: {hit['answer']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())