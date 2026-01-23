# Evaluation Metrics

7 separate metrics (no weighted composites). See full docs: [docs/specs/evaluation.md](../../docs/specs/evaluation.md)

## The 7 Metrics

| Metric | What It Measures | Good Target |
|--------|------------------|-------------|
| **AUPRC** | Ranking quality of verdicts | 85%+ |
| **Evidence Precision** | % claimed evidences exist in GT | 80%+ |
| **Evidence Recall** | % GT evidences found by method | 60%+ |
| **Snippet Validity** | % quotes match source text | 95%+ |
| **Judgement Accuracy** | % correct field values | 70%+ |
| **Score Accuracy** | Score matches GT (V2/V3 only) | 90%+ |
| **Verdict Consistency** | Evidence+rule → verdict | 90%+ |

## Quick Reference

```python
from addm.eval import compute_evaluation_metrics

metrics = compute_evaluation_metrics(
    results=results,
    gt_data=gt_data,
    gt_verdicts=gt_verdicts,
    method="amos",
    reviews_data=reviews_data,  # For snippet validation
    policy_id="G1_allergy_V2",  # For score accuracy
)

# All 7 metrics
print(metrics["auprc"])
print(metrics["evidence_precision"])
print(metrics["evidence_recall"])
print(metrics["snippet_validity"])
print(metrics["judgement_accuracy"])
print(metrics["score_accuracy"])      # None for V0/V1
print(metrics["verdict_consistency"])
```

## Console Output

```
EVALUATION METRICS (7 total)
┌─────────────────────┬─────────┬─────────────────────────────┐
│ Metric              │ Score   │ Notes                       │
├─────────────────────┼─────────┼─────────────────────────────┤
│ AUPRC               │ 85.3%   │ (ranking quality)           │
│ Evidence Precision  │ 72.0%   │ (12/15 claimed exist in GT) │
│ Evidence Recall     │ 60.0%   │ (12/20 GT evidences found)  │
│ Snippet Validity    │ 95.0%   │ (19/20 quotes match source) │
│ Judgement Accuracy  │ 68.5%   │ (field correctness)         │
│ Score Accuracy      │ 90.0%   │ (9/10 scores match GT)      │
│ Verdict Consistency │ 91.2%   │ (evidence+rule→verdict)     │
└─────────────────────┴─────────┴─────────────────────────────┘
```

## Evidence Fields by Policy Group

| Group | Field | Values |
|-------|-------|--------|
| G1 | incident_severity | mild, moderate, severe |
| G2 | date_outcome | positive, negative |
| G3 | quality_for_price | excellent, good, poor, terrible |
| G4 | attentiveness | excellent, good, poor, terrible |
| G5 | service_degradation | minor, moderate, severe |
| G6 | memorability | memorable, remarkable, forgettable |

## Files

| File | Purpose |
|------|---------|
| `src/addm/eval/metrics.py` | AUPRC, Score Accuracy, Verdict Consistency |
| `src/addm/eval/intermediate_metrics.py` | Evidence Precision/Recall, Snippet Validity, Judgement Accuracy |
| `src/addm/eval/constants.py` | Scoring rules (V2 policy), EVIDENCE_FIELDS |
| `src/addm/eval/__init__.py` | Public exports |

---

**Full documentation:** [docs/specs/evaluation.md](../../docs/specs/evaluation.md)
