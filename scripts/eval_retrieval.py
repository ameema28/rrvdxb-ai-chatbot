"""Retrieval evaluation for the FAQ RAG index (no LLM, no API key).

Builds a fresh ChromaDB index from app/mock_data/faqs.json in a temporary
directory (the repo's own index is not touched), runs every question in
data/eval/retrieval_eval.jsonl through search_faqs(top_k=3), and reports:

Gated metrics (what the chat flow uses: only hits with similarity >= the
retriever threshold count, currently 0.6):
  hit@1, hit@3, MRR@3, in-scope false-rejection rate (no hit passes the gate),
  off-topic rejection rate (no hit passes the gate for an off-topic question).
Raw ranking metrics (no threshold), to separate ranking misses from
threshold misses: hit@1, hit@3, MRR@3.

Usage:
    python scripts/eval_retrieval.py --out C:/path/outside/repo/retrieval_eval.json
"""

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET_PATH = REPO_ROOT / "data" / "eval" / "retrieval_eval.jsonl"
TOP_K = 3

Search = Callable[[str, int], List[Dict]]


def resolve_out_path(out: str) -> Path:
    """Absolute output path; refuses anything inside the repository."""
    path = Path(out).resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise ValueError(f"--out must be outside the repository ({REPO_ROOT})")
    return path


def load_dataset(path: Path = DATASET_PATH) -> List[Dict]:
    """Rows of {"q": str, "expected_id": int | None}; raises ValueError on a bad row."""
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("q"), str) or not row["q"].strip():
            raise ValueError(f"line {number}: 'q' must be a non-empty string")
        expected = row.get("expected_id", "missing")
        if expected is not None and not isinstance(expected, int):
            raise ValueError(f"line {number}: 'expected_id' must be an int or null")
        rows.append({"q": row["q"], "expected_id": expected})
    return rows


def _rank(expected_id: int, hits: List[Dict]) -> Optional[int]:
    for position, hit in enumerate(hits, start=1):
        if hit["id"] == expected_id:
            return position
    return None


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 4) if denominator else None


def evaluate(rows: List[Dict], search: Search, threshold: float, top_k: int = TOP_K) -> Dict:
    """Run every row through `search` and compute gated and raw metrics."""
    per_question = []
    for row in rows:
        hits = search(row["q"], top_k)
        gated = [h for h in hits if h["similarity"] >= threshold]
        record = {
            "q": row["q"],
            "expected_id": row["expected_id"],
            "top": [{"id": h["id"], "similarity": h["similarity"]} for h in hits],
            "passes_gate": bool(gated),
        }
        if row["expected_id"] is not None:
            record["raw_rank"] = _rank(row["expected_id"], hits)
            record["gated_rank"] = _rank(row["expected_id"], gated)
        per_question.append(record)

    in_scope = [r for r in per_question if r["expected_id"] is not None]
    off_topic = [r for r in per_question if r["expected_id"] is None]

    def metrics(rank_key: str) -> Dict:
        ranks = [r[rank_key] for r in in_scope]
        return {
            "hit_at_1": _rate(sum(rank == 1 for rank in ranks), len(ranks)),
            f"hit_at_{top_k}": _rate(sum(rank is not None for rank in ranks), len(ranks)),
            f"mrr_at_{top_k}": round(sum(1 / rank for rank in ranks if rank) / len(ranks), 4)
            if ranks
            else None,
        }

    gated_metrics = metrics("gated_rank")
    gated_metrics["in_scope_false_rejection_rate"] = _rate(
        sum(not r["passes_gate"] for r in in_scope), len(in_scope)
    )
    gated_metrics["off_topic_rejection_rate"] = _rate(
        sum(not r["passes_gate"] for r in off_topic), len(off_topic)
    )
    return {
        "threshold": threshold,
        "top_k": top_k,
        "in_scope_questions": len(in_scope),
        "off_topic_questions": len(off_topic),
        "gated": gated_metrics,
        "raw": metrics("raw_rank"),
        "misses": {
            "gated_not_top1": [r["q"] for r in in_scope if r["gated_rank"] != 1],
            "raw_not_top1": [r["q"] for r in in_scope if r["raw_rank"] != 1],
            "off_topic_passed_gate": [r["q"] for r in off_topic if r["passes_gate"]],
        },
        "per_question": per_question,
    }


def run_on_fresh_index(rows: List[Dict]) -> Dict:
    """Build the FAQ index in a temporary directory and evaluate `rows` on it."""
    from app.ai.chatbot.rag import retriever, vector_store

    saved = (vector_store._PERSIST_DIR, vector_store._collection)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        vector_store._PERSIST_DIR = Path(tmp) / "chroma_db"
        vector_store._collection = None
        try:
            chunks = vector_store.build_vector_store()
            result = evaluate(
                rows,
                lambda q, k: vector_store.search_faqs(q, top_k=k),
                retriever.DEFAULT_SIMILARITY_THRESHOLD,
            )
        finally:
            vector_store._PERSIST_DIR, vector_store._collection = saved
    result["indexed_chunks"] = chunks
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="JSON results file, outside the repository")
    args = parser.parse_args(argv)
    try:
        out = resolve_out_path(args.out)
    except ValueError as err:
        parser.error(str(err))

    logging.basicConfig(level=logging.WARNING)
    rows = load_dataset()
    started = time.perf_counter()
    result = run_on_fresh_index(rows)
    result["seconds"] = round(time.perf_counter() - started, 2)
    result["run_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    gated, raw = result["gated"], result["raw"]
    print(
        f"{result['in_scope_questions']} in-scope + {result['off_topic_questions']} off-topic questions, "
        f"{result['indexed_chunks']} chunks, threshold {result['threshold']}"
    )
    print(
        f"Gated: hit@1={gated['hit_at_1']} hit@3={gated['hit_at_3']} MRR@3={gated['mrr_at_3']} "
        f"false-rejection={gated['in_scope_false_rejection_rate']} "
        f"off-topic rejection={gated['off_topic_rejection_rate']}"
    )
    print(f"Raw:   hit@1={raw['hit_at_1']} hit@3={raw['hit_at_3']} MRR@3={raw['mrr_at_3']}")
    for row in result["per_question"]:
        top = ", ".join(f"{h['id']}:{h['similarity']:.3f}" for h in row["top"])
        print(f"  expected={row['expected_id']} gate={'pass' if row['passes_gate'] else 'reject'} [{top}] {row['q']}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
