# Evaluation Metrics

7 separate metrics (no weighted composites). See full docs: [docs/specs/evaluation.md](../../docs/specs/evaluation.md)

## The 7 Metrics

| Metric | What It Measures | Good Target |
|--------|------------------|-------------|
| **AUPRC** | Ranking quality (are risky items ranked higher?) | 85%+ |
| **Macro F1** | Classification quality (are predictions correct?) | 60%+ |
| **Evidence Precision** | % claimed evidences exist in GT | 80%+ |
| **Evidence Recall** | % GT evidences found by method | 60%+ |
| **Snippet Validity** | % quotes match source text | 95%+ |
| **Judgement Accuracy** | % correct field values | 70%+ |
| **Verdict Consistency** | Evidence+rule → verdict | 90%+ |

## AUPRC vs Macro F1

**AUPRC** measures **ranking quality**: "Are high-risk items ranked above low-risk items?"
- Can be high (even 100%) if the *order* is correct, even with wrong classifications
- Example: GT=[Low,High], Pred=[High,Critical], Scores=[1,2] → AUPRC=1.0 but Accuracy=0%
- Use case: triage/prioritization workflows

**Macro F1** measures **classification quality**: "Are the predictions actually correct?"
- Averages F1 across all classes (Low, High, Critical)
- Resistant to class imbalance (penalizes missing classes equally)
- Use case: decision-making where correct labels matter

## Quick Reference

```python
from addm.eval import compute_evaluation_metrics

metrics = compute_evaluation_metrics(
    results=results,
    gt_data=gt_data,
    gt_verdicts=gt_verdicts,
    method="amos",
    reviews_data=reviews_data,  # For snippet validation
    policy_id="G1_allergy_V2",
)

# All 7 metrics
print(metrics["auprc"])
print(metrics["macro_f1"])
print(metrics["evidence_precision"])
print(metrics["evidence_recall"])
print(metrics["snippet_validity"])
print(metrics["judgement_accuracy"])
print(metrics["verdict_consistency"])
```

## Console Output

```
EVALUATION METRICS (7 total)
┌─────────────────────┬─────────┬─────────────────────────────┐
│ Metric              │ Score   │ Notes                       │
├─────────────────────┼─────────┼─────────────────────────────┤
│ AUPRC               │ 85.3%   │ (ranking quality)           │
│ Macro F1            │ 72.0%   │ (classification quality)    │
│ Evidence Precision  │ 72.0%   │ (12/15 claimed exist in GT) │
│ Evidence Recall     │ 60.0%   │ (12/20 GT evidences found)  │
│ Snippet Validity    │ 95.0%   │ (19/20 quotes match source) │
│ Judgement Accuracy  │ 68.5%   │ (field correctness)         │
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
| `src/addm/eval/metrics.py` | AUPRC, Macro F1, Verdict Consistency |
| `src/addm/eval/intermediate_metrics.py` | Evidence Precision/Recall, Snippet Validity, Judgement Accuracy |
| `src/addm/eval/constants.py` | Scoring rules, EVIDENCE_FIELDS |
| `src/addm/eval/__init__.py` | Public exports |

---

**Full documentation:** [docs/specs/evaluation.md](../../docs/specs/evaluation.md)
