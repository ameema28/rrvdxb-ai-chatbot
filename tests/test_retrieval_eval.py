"""
Retrieval evaluation (scripts/eval_retrieval.py, data/eval/retrieval_eval.jsonl).

The dataset checks and the metric arithmetic run everywhere. The real-index
test (marker `chroma`) fails if any metric gets worse than the values measured
on 2026-11-07 (30 in-scope + 7 off-topic questions, threshold 0.6):

  gated  hit@1 = hit@3 = MRR@3 = 0.40, in-scope false rejection = 0.60,
         off-topic rejection = 1.00
  raw    hit@1 = hit@3 = MRR@3 = 1.00
"""

import difflib
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("eval_retrieval", ROOT / "scripts" / "eval_retrieval.py")
eval_retrieval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_retrieval)

FAQS = json.loads((ROOT / "app" / "mock_data" / "faqs.json").read_text(encoding="utf-8"))

MEASURED_FLOORS = {
    ("gated", "hit_at_1"): 0.40,
    ("gated", "hit_at_3"): 0.40,
    ("gated", "mrr_at_3"): 0.40,
    ("gated", "off_topic_rejection_rate"): 1.00,
    ("raw", "hit_at_1"): 1.00,
    ("raw", "hit_at_3"): 1.00,
    ("raw", "mrr_at_3"): 1.00,
}
MEASURED_MAX_FALSE_REJECTION = 0.60


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

def test_dataset_size_and_labels():
    rows = eval_retrieval.load_dataset()
    in_scope = [r for r in rows if r["expected_id"] is not None]
    off_topic = [r for r in rows if r["expected_id"] is None]
    assert len(in_scope) >= 25
    assert len(off_topic) >= 5
    faq_ids = {faq["id"] for faq in FAQS}
    assert {r["expected_id"] for r in in_scope} <= faq_ids
    assert {r["expected_id"] for r in in_scope} == faq_ids, "every FAQ has eval questions"
    assert len({r["q"] for r in rows}) == len(rows), "duplicate question"


def test_questions_do_not_copy_faq_wording():
    faq_questions = [_normalize(faq["question"]) for faq in FAQS]
    for row in eval_retrieval.load_dataset():
        q = _normalize(row["q"])
        assert q not in faq_questions, row["q"]
        for faq_q in faq_questions:
            ratio = difflib.SequenceMatcher(None, q, faq_q).ratio()
            assert ratio < 0.85, f"{row['q']!r} is too close to {faq_q!r} ({ratio:.2f})"


def test_load_dataset_rejects_bad_rows(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"q": "hello", "expected_id": "5"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        eval_retrieval.load_dataset(bad)
    bad.write_text('{"q": "", "expected_id": 5}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        eval_retrieval.load_dataset(bad)


def test_out_path_is_refused_inside_the_repo(tmp_path):
    with pytest.raises(ValueError):
        eval_retrieval.resolve_out_path(str(ROOT / "eval.json"))
    with pytest.raises(ValueError):
        eval_retrieval.resolve_out_path(str(ROOT))
    assert eval_retrieval.resolve_out_path(str(tmp_path / "eval.json")) == (tmp_path / "eval.json").resolve()


# --------------------------------------------------------------------------
# Metric arithmetic (fake search, no model)
# --------------------------------------------------------------------------

def test_evaluate_computes_gated_and_raw_metrics():
    rows = [
        {"q": "a", "expected_id": 1},
        {"q": "b", "expected_id": 2},
        {"q": "c", "expected_id": 3},
        {"q": "off-1", "expected_id": None},
        {"q": "off-2", "expected_id": None},
    ]
    results = {
        "a": [{"id": 1, "similarity": 0.9}, {"id": 2, "similarity": 0.3}],
        "b": [{"id": 1, "similarity": 0.7}, {"id": 2, "similarity": 0.65}],
        "c": [{"id": 3, "similarity": 0.5}, {"id": 1, "similarity": 0.2}],
        "off-1": [{"id": 4, "similarity": 0.2}],
        "off-2": [{"id": 4, "similarity": 0.61}],
    }
    out = eval_retrieval.evaluate(rows, lambda q, k: results[q], threshold=0.6, top_k=3)
    assert out["gated"] == {
        "hit_at_1": 0.3333,
        "hit_at_3": 0.6667,
        "mrr_at_3": 0.5,
        "in_scope_false_rejection_rate": 0.3333,
        "off_topic_rejection_rate": 0.5,
    }
    assert out["raw"] == {"hit_at_1": 0.6667, "hit_at_3": 1.0, "mrr_at_3": 0.8333}
    assert out["misses"]["gated_not_top1"] == ["b", "c"]
    assert out["misses"]["raw_not_top1"] == ["b"]
    assert out["misses"]["off_topic_passed_gate"] == ["off-2"]


def test_main_writes_results_outside_the_repo(tmp_path, monkeypatch):
    fake = {"gated": {"hit_at_1": 1, "hit_at_3": 1, "mrr_at_3": 1, "in_scope_false_rejection_rate": 0,
                      "off_topic_rejection_rate": 1},
            "raw": {"hit_at_1": 1, "hit_at_3": 1, "mrr_at_3": 1},
            "in_scope_questions": 1, "off_topic_questions": 0, "indexed_chunks": 12, "threshold": 0.6,
            "per_question": []}
    monkeypatch.setattr(eval_retrieval, "run_on_fresh_index", lambda rows: dict(fake))
    out = tmp_path / "eval.json"
    assert eval_retrieval.main(["--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["indexed_chunks"] == 12
    with pytest.raises(SystemExit):
        eval_retrieval.main(["--out", str(ROOT / "eval.json")])


# --------------------------------------------------------------------------
# Real index: no regression below the measured values
# --------------------------------------------------------------------------

@pytest.mark.chroma
def test_real_retrieval_metrics_do_not_regress():
    result = eval_retrieval.run_on_fresh_index(eval_retrieval.load_dataset())
    assert result["indexed_chunks"] == len(FAQS)
    assert result["threshold"] == 0.6
    for (group, name), floor in MEASURED_FLOORS.items():
        assert result[group][name] >= floor, f"{group} {name} = {result[group][name]} < {floor}"
    assert result["gated"]["in_scope_false_rejection_rate"] <= MEASURED_MAX_FALSE_REJECTION
