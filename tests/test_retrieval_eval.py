"""
Retrieval evaluation (scripts/eval_retrieval.py) on two sets:
  dev      data/eval/retrieval_eval.jsonl    (30 in-scope + 7 off-topic; used to choose the threshold)
  heldout  data/eval/retrieval_heldout.jsonl (15 in-scope + 10 off-topic; written before any
           measurement and used only to check the chosen threshold)

The dataset checks and the metric arithmetic run everywhere. The real-index
test (marker `chroma`) fails if any metric gets worse than the values measured
on 2026-11-08 at the default threshold 0.45 (floors are the measured values
rounded down to 2 decimals; ceilings rounded up):

  dev      gated hit@1 = hit@3 = MRR@3 = 0.6667, false rejection 0.3333,
           off-topic rejection 1.00 (off-topic max similarity 0.3598)
           raw hit@1 = hit@3 = MRR@3 = 1.00
  heldout  gated hit@1 0.80, hit@3 0.8667, MRR@3 0.8333, false rejection 0.1333,
           off-topic rejection 0.60 (off-topic max similarity 0.648)
           raw hit@1 0.9333, hit@3 1.00, MRR@3 0.9667

Known limit, kept visible here on purpose: 4 of the 10 held-out off-topic
questions pass the 0.45 gate (one, 0.648, would also pass 0.60).
"""

import difflib
import importlib.util
import json
import re
from pathlib import Path

import pytest

from app.core.config import settings

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("eval_retrieval", ROOT / "scripts" / "eval_retrieval.py")
eval_retrieval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_retrieval)

FAQS = json.loads((ROOT / "app" / "mock_data" / "faqs.json").read_text(encoding="utf-8"))

FLOORS = {
    "dev": {
        ("gated", "hit_at_1"): 0.66,
        ("gated", "hit_at_3"): 0.66,
        ("gated", "mrr_at_3"): 0.66,
        ("gated", "off_topic_rejection_rate"): 1.00,
        ("raw", "hit_at_1"): 1.00,
        ("raw", "hit_at_3"): 1.00,
        ("raw", "mrr_at_3"): 1.00,
    },
    "heldout": {
        ("gated", "hit_at_1"): 0.80,
        ("gated", "hit_at_3"): 0.86,
        ("gated", "mrr_at_3"): 0.83,
        ("gated", "off_topic_rejection_rate"): 0.60,
        ("raw", "hit_at_1"): 0.93,
        ("raw", "hit_at_3"): 1.00,
        ("raw", "mrr_at_3"): 0.96,
    },
}
CEILINGS = {
    "dev": {"in_scope_false_rejection_rate": 0.34, "off_topic_max_similarity": 0.36},
    "heldout": {"in_scope_false_rejection_rate": 0.14, "off_topic_max_similarity": 0.65},
}
MIN_SIZES = {"dev": (25, 5), "heldout": (15, 10)}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _sets():
    return {name: eval_retrieval.load_dataset(path) for name, path in eval_retrieval.DATASETS.items()}


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["dev", "heldout"])
def test_dataset_size_and_labels(name):
    rows = _sets()[name]
    in_scope = [r for r in rows if r["expected_id"] is not None]
    off_topic = [r for r in rows if r["expected_id"] is None]
    min_in, min_off = MIN_SIZES[name]
    assert len(in_scope) >= min_in
    assert len(off_topic) >= min_off
    faq_ids = {faq["id"] for faq in FAQS}
    assert {r["expected_id"] for r in in_scope} == faq_ids, "every FAQ has questions"
    assert len({r["q"] for r in rows}) == len(rows), "duplicate question"


def test_questions_do_not_copy_faq_wording():
    faq_questions = [_normalize(faq["question"]) for faq in FAQS]
    for rows in _sets().values():
        for row in rows:
            q = _normalize(row["q"])
            assert q not in faq_questions, row["q"]
            for faq_q in faq_questions:
                ratio = difflib.SequenceMatcher(None, q, faq_q).ratio()
                assert ratio < 0.85, f"{row['q']!r} is too close to {faq_q!r} ({ratio:.2f})"


def test_heldout_does_not_copy_dev_questions():
    sets = _sets()
    dev = [_normalize(r["q"]) for r in sets["dev"]]
    for row in sets["heldout"]:
        q = _normalize(row["q"])
        assert q not in dev, row["q"]
        for dev_q in dev:
            ratio = difflib.SequenceMatcher(None, q, dev_q).ratio()
            assert ratio < 0.85, f"{row['q']!r} is too close to dev question {dev_q!r} ({ratio:.2f})"


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
    assert out["off_topic_max_similarity"] == 0.61
    assert out["misses"]["gated_not_top1"] == ["b", "c"]
    assert out["misses"]["raw_not_top1"] == ["b"]
    assert out["misses"]["off_topic_passed_gate"] == ["off-2"]


def _fake_set_result():
    return {
        "gated": {"hit_at_1": 1, "hit_at_3": 1, "mrr_at_3": 1, "in_scope_false_rejection_rate": 0,
                  "off_topic_rejection_rate": 1},
        "raw": {"hit_at_1": 1, "hit_at_3": 1, "mrr_at_3": 1},
        "in_scope_questions": 1, "off_topic_questions": 0, "threshold": 0.45,
        "off_topic_max_similarity": None, "per_question": [],
    }


def test_main_writes_both_sets_outside_the_repo(tmp_path, monkeypatch):
    seen = {}

    def fake_run(datasets, threshold=None):
        seen["names"], seen["threshold"] = sorted(datasets), threshold
        return {"dev": _fake_set_result(), "heldout": _fake_set_result(), "indexed_chunks": 12}

    monkeypatch.setattr(eval_retrieval, "run_on_fresh_index", fake_run)
    out = tmp_path / "eval.json"
    assert eval_retrieval.main(["--out", str(out), "--threshold", "0.6"]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["indexed_chunks"] == 12 and set(saved) >= {"dev", "heldout"}
    assert seen == {"names": ["dev", "heldout"], "threshold": 0.6}
    with pytest.raises(SystemExit):
        eval_retrieval.main(["--out", str(ROOT / "eval.json")])
    with pytest.raises(SystemExit):
        eval_retrieval.main(["--out", str(out), "--threshold", "1.5"])


# --------------------------------------------------------------------------
# Real index: no regression below the measured values at the default 0.45
# --------------------------------------------------------------------------

@pytest.mark.chroma
def test_real_retrieval_metrics_do_not_regress():
    assert settings.rag_similarity_threshold == 0.45
    result = eval_retrieval.run_on_fresh_index(_sets())
    assert result["indexed_chunks"] == len(FAQS)
    for name in ("dev", "heldout"):
        metrics = result[name]
        assert metrics["threshold"] == 0.45
        for (group, key), floor in FLOORS[name].items():
            assert metrics[group][key] >= floor, f"{name} {group} {key} = {metrics[group][key]} < {floor}"
        ceilings = CEILINGS[name]
        assert metrics["gated"]["in_scope_false_rejection_rate"] <= ceilings["in_scope_false_rejection_rate"]
        assert metrics["off_topic_max_similarity"] <= ceilings["off_topic_max_similarity"]
