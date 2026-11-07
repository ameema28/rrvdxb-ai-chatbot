# Retrieval evaluation sets

| File | Questions | Purpose |
|---|---|---|
| `retrieval_eval.jsonl` (dev) | 30 in-scope + 7 off-topic | Used to **choose** the retrieval threshold |
| `retrieval_heldout.jsonl` (held-out) | 15 in-scope + 10 off-topic, several near-topic (other shops, cities, unrelated deliveries) | Written **before any measurement** and not edited after; used only to **check** the chosen threshold |

Each row is `{"q": ..., "expected_id": <FAQ id> | null}` (`null` = off-topic, should match no FAQ).
In-scope questions are paraphrases written as customers might ask; tests check that none copies a FAQ question (and that the held-out set does not copy the dev set).

Run: `python scripts/eval_retrieval.py --out <path outside the repo>` (no LLM, no API key; builds a fresh index in a temp directory). `--threshold` overrides the threshold for comparison.

## Threshold: 0.45 (`RAG_SIMILARITY_THRESHOLD`)

Measured 2026-11-08, top-3 retrieval, all-MiniLM-L6-v2, 12 FAQ chunks. "Gated" = only hits at or above the threshold count, as in the chat flow.

| Set | Threshold | Gated hit@1 | hit@3 | MRR@3 | In-scope passing | Off-topic rejected | Off-topic max similarity |
|---|---|---|---|---|---|---|---|
| dev | 0.60 (old) | 0.40 | 0.40 | 0.40 | 12/30 | 7/7 | 0.360 |
| dev | **0.45** | 0.67 | 0.67 | 0.67 | 20/30 | 7/7 | 0.360 |
| held-out | 0.60 (old) | 0.33 | 0.33 | 0.33 | 5/15 | 9/10 | 0.648 |
| held-out | **0.45** | 0.80 | 0.87 | 0.83 | 13/15 | 6/10 | 0.648 |

Raw ranking (no threshold): dev hit@1 1.00; held-out hit@1 0.93 (one miss: "Do you ship within Sharjah?" ranks the KSA FAQ above the UAE FAQ).

**Known leaks.** At 0.45, 4 of the 10 held-out off-topic questions pass the gate:
- "Does Amazon deliver to Riyadh?" 0.648 (KSA shipping FAQ). This one also passed the old 0.60.
- "How much is a flight from Karachi to Lahore?" 0.583 (Pakistan FAQ)
- "How long does Noon take to refund an order?" 0.455 (returns FAQ)
- "Is the Dubai metro running late tonight?" 0.454 (UAE FAQ)

Questions that name a city, a delivery or a refund score like real customer questions, so no single similarity threshold separates them here. As a second line of defence the reply system prompt has a general scope rule (answer only about RRVDXB; decline other companies, travel, transport and unrelated topics). A test checks the rule is in the prompt on every reply path; **whether the live model follows it is not measured**.

The sets are small (45 in-scope, 17 off-topic questions in total); treat the numbers as indicative.
