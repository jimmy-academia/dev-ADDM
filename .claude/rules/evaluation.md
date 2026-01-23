# Evaluation Metrics

6 separate metrics (no weighted composites). See full docs: [docs/specs/evaluation.md](../../docs/specs/evaluation.md)

## The 6 Metrics

| Metric | What It Measures | Good Target |
|--------|------------------|-------------|
| **AUPRC** | Ranking quality of verdicts | 85%+ |
| **Incident Precision** | % claimed incidents exist in GT | 80%+ |
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

# All 6 metrics
print(metrics["auprc"])
print(metrics["incident_precision"])
print(metrics["snippet_validity"])
print(metrics["judgement_accuracy"])
print(metrics["score_accuracy"])      # None for V0/V1
print(metrics["verdict_consistency"])
```

## Console Output

```
EVALUATION METRICS (6 total)
┌─────────────────────┬─────────┬─────────────────────────────┐
│ Metric              │ Score   │ Notes                       │
├─────────────────────┼─────────┼─────────────────────────────┤
│ AUPRC               │ 85.3%   │ (ranking quality)           │
│ Incident Precision  │ 72.0%   │ (12/15 claimed exist in GT) │
│ Snippet Validity    │ 95.0%   │ (19/20 quotes match source) │
│ Judgement Accuracy  │ 68.5%   │ (field correctness)         │
│ Score Accuracy      │ 90.0%   │ (9/10 scores match GT)      │
│ Verdict Consistency │ 91.2%   │ (evidence+rule→verdict)     │
└─────────────────────┴─────────┴─────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `src/addm/eval/metrics.py` | AUPRC, Score Accuracy, Verdict Consistency |
| `src/addm/eval/intermediate_metrics.py` | Incident Precision, Snippet Validity, Judgement Accuracy |
| `src/addm/eval/constants.py` | Scoring rules (V2 policy) |
| `src/addm/eval/__init__.py` | Public exports |

---

**Full documentation:** [docs/specs/evaluation.md](../../docs/specs/evaluation.md)
