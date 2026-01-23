# Evaluation Metrics

Simplified evaluation system with **6 separate, interpretable metrics** - no weighted composites.

## Design Philosophy

The old system used weighted composite scores (Process Score = 35% incident precision + 30% severity accuracy + ...). This had problems:
- Arbitrary weights obscured what was actually being measured
- A single "Process Score" hid important distinctions
- Hard to diagnose what went wrong

**New approach:** 6 separate metrics, each measuring one thing clearly.

---

## The Six Metrics

| Tier | Metric | What It Measures |
|------|--------|------------------|
| **Final Quality** | AUPRC | Ranking quality of verdicts |
| **Evidence Quality** | Incident Precision | Are claimed incidents real? |
| **Evidence Quality** | Snippet Validity | Are quoted texts real? |
| **Reasoning Quality** | Judgement Accuracy | Are field values correct? |
| **Reasoning Quality** | Score Accuracy | Is computed score correct? (V2/V3) |
| **Reasoning Quality** | Verdict Consistency | Does evidence → verdict logic hold? |

---

## Tier 1: Final Quality

### AUPRC (Ordinal Area Under Precision-Recall Curve)

**Purpose:** Measures how well the method ranks restaurants by risk level.

**Computation:**
- Binary AUPRC for two thresholds:
  - **≥High**: High+Critical vs Low
  - **≥Critical**: Critical vs rest
- Final: Mean of both AUPRC values

**Why ordinal?** Standard AUPRC treats all errors equally. Ordinal AUPRC respects severity ordering - confusing High/Critical is less wrong than confusing Low/Critical.

**Location:** `src/addm/eval/metrics.py::compute_ordinal_auprc()`

```python
from addm.eval import compute_ordinal_auprc
import numpy as np

y_true = np.array([0, 0, 1, 2])  # Low, Low, High, Critical
y_scores = np.array([1.0, 2.0, 6.0, 10.0])

result = compute_ordinal_auprc(y_true, y_scores)
# result["ordinal_auprc"] = 0.85
```

---

## Tier 2: Evidence Quality

### Incident Precision

**Purpose:** Are the claimed incidents actually in the ground truth?

**Formula:** `|claimed ∩ GT| / |claimed|` (by review_id)

**Interpretation:**
- 100% = Every claimed incident exists in GT
- 80% = 20% of claims are fabricated/wrong review IDs
- N/A = Method made no incident claims (can't compute precision)

**Location:** `src/addm/eval/intermediate_metrics.py::compute_evidence_validity()`

### Snippet Validity

**Purpose:** Are the quoted text snippets actually in the source reviews?

**Formula:** `|valid snippets| / |total snippets|`

**Validation:** Each `evidence.snippet` is checked as substring of `reviews[evidence.review_id].text`.

**Interpretation:**
- 100% = All quotes are real
- <100% = Some quotes are fabricated/hallucinated
- N/A = No snippets provided or no review data available

**Note:** Requires `reviews_data` parameter to be passed to compute.

---

## Tier 3: Reasoning Quality

### Judgement Accuracy

**Purpose:** For matched incidents, are the field values correct?

**How it works:**
1. Match method's evidences to GT incidents by `review_id`
2. For each match, compare method's `field`/`judgement` to GT's `severity_field`/`severity_value`
3. Also checks modifier fields (assurance, staff response) if present

**Policy-agnostic:** Works for any policy group (G1-G6) using GT's `severity_field`/`severity_value`.

**Example:**
- GT: `{review_id: "abc", severity_field: "incident_severity", severity_value: "moderate"}`
- Method: `{review_id: "abc", field: "INCIDENT_SEVERITY", judgement: "moderate incident"}`
- Result: ✓ Match (normalized comparison)

**Location:** `src/addm/eval/intermediate_metrics.py::compute_judgement_accuracy()`

### Score Accuracy (V2/V3 only)

**Purpose:** Does the method's computed score match ground truth?

**Formula:** `|method_score - gt_score| < 0.01` (exact match within float tolerance)

**Only evaluated for:**
- V2 policies (point-based scoring)
- V3 policies (point-based with recency weighting)

**Skipped for:**
- V0/V1 policies (condition-based, no numeric scores)

**Fields compared:**
- Method: `parsed.justification.scoring_trace.total_score`
- GT: `score`

**Location:** `src/addm/eval/metrics.py::compute_score_accuracy()`

### Verdict Consistency (Enhanced)

**Purpose:** Does the method's internal reasoning support its claimed verdict?

**Two checks:**
1. **Evidence → Verdict:** Recompute verdict from method's claimed evidences using V2 scoring rules. Does it match claimed verdict?
2. **Rule → Verdict:** If method claims `triggered_rule`, does that rule imply the claimed verdict?

**Consistent only if BOTH pass** (rule check skipped if no rule claimed).

**Rule mapping:**
| triggered_rule | Implied Verdict |
|----------------|-----------------|
| `CRITICAL` | Critical Risk |
| `HIGH` | High Risk |
| `LOW` | Low Risk |

**Why enhanced?** Old version only checked evidence→verdict. New version also validates that claimed triggered_rule is consistent.

**Location:** `src/addm/eval/metrics.py::compute_verdict_consistency_enhanced()`

---

## Scoring Rules Reference (V2 Policy)

Used for Verdict Consistency and Score Accuracy validation:

### Severity Base Points

| Severity | Points |
|----------|--------|
| Mild | 2 |
| Moderate | 5 |
| Severe | 15 |

### Modifier Bonuses

| Modifier | Points |
|----------|--------|
| False assurance | +5 |
| Dismissive staff | +3 |

### Verdict Thresholds

| Verdict | Score Threshold |
|---------|-----------------|
| Critical Risk | ≥8 points |
| High Risk | ≥4 points |
| Low Risk | <4 points |

**Example scoring:**
- 1 moderate (5) + false assurance (+5) = 10 → Critical Risk
- 2 mild (2×2 = 4) → High Risk
- 1 mild (2) → Low Risk

**Constants:** `src/addm/eval/constants.py`

---

## Console Output

```
EVALUATION METRICS (6 total)
┌─────────────────────┬─────────┬────────────────────────────────┐
│ Metric              │ Score   │ Notes                          │
├─────────────────────┼─────────┼────────────────────────────────┤
│ AUPRC               │ 85.3%   │ (ranking quality)              │
│ Incident Precision  │ 72.0%   │ (12/15 claimed exist in GT)    │
│ Snippet Validity    │ 95.0%   │ (19/20 quotes match source)    │
│ Judgement Accuracy  │ 68.5%   │ (field correctness)            │
│ Score Accuracy      │ 90.0%   │ (9/10 scores match GT)         │
│ Verdict Consistency │ 91.2%   │ (evidence+rule→verdict)        │
└─────────────────────┴─────────┴────────────────────────────────┘
```

---

## results.json Structure

```json
{
  "run_id": "G1_allergy_V2",
  "policy_id": "G1_allergy_V2",
  "method": "amos",
  "model": "gpt-5-nano",
  "k": 50,
  "n": 100,

  "evaluation_metrics": {
    "auprc": 0.853,
    "incident_precision": 0.72,
    "snippet_validity": 0.95,
    "judgement_accuracy": 0.685,
    "score_accuracy": 0.90,
    "verdict_consistency": 0.912,
    "n_samples": 100,
    "n_structured": 95,
    "n_with_gt": 100
  },

  "evaluation_metrics_full": {
    "auprc": 0.853,
    "incident_precision": 0.72,
    "snippet_validity": 0.95,
    "judgement_accuracy": 0.685,
    "score_accuracy": 0.90,
    "verdict_consistency": 0.912,
    "details": {
      "auprc_metrics": {
        "auprc_ge_high": 0.83,
        "auprc_ge_critical": 0.88,
        "ordinal_auprc": 0.853
      },
      "incident_details": {
        "total_claimed": 15,
        "total_matched": 12,
        "total_snippets": 20,
        "valid_snippets": 19
      },
      "judgement_details": {
        "correct": 24,
        "total": 35,
        "per_field": {...}
      },
      "score_details": {
        "correct": 9,
        "total": 10,
        "per_sample": [...]
      },
      "consistency_details": {
        "consistent": 82,
        "total": 90,
        "skipped_no_evidence": 5,
        "per_sample": [...]
      }
    }
  },

  "results": [...]
}
```

---

## API Usage

### Main Entry Point

```python
from addm.eval import compute_evaluation_metrics

metrics = compute_evaluation_metrics(
    results=method_outputs,      # List of method result dicts
    gt_data=gt_with_incidents,   # {biz_id: {verdict, score, incidents}}
    gt_verdicts=gt_verdicts,     # {biz_id: verdict}
    method="amos",               # Method name
    reviews_data=reviews_data,   # Optional: for snippet validation
    policy_id="G1_allergy_V2",   # Optional: for V2/V3 score check
)

# Access individual metrics
print(f"AUPRC: {metrics['auprc']:.1%}")
print(f"Incident Precision: {metrics['incident_precision']:.1%}")
print(f"Snippet Validity: {metrics['snippet_validity']:.1%}")
print(f"Judgement Accuracy: {metrics['judgement_accuracy']:.1%}")
print(f"Score Accuracy: {metrics['score_accuracy']:.1%}")
print(f"Verdict Consistency: {metrics['verdict_consistency']:.1%}")
```

### Individual Metrics

```python
from addm.eval import (
    compute_ordinal_auprc,
    compute_evidence_validity,
    compute_judgement_accuracy,
    compute_score_accuracy,
    compute_verdict_consistency_enhanced,
)

# Each can be computed independently
score_acc, score_details = compute_score_accuracy(results, gt_data, policy_id)
consistency, cons_details = compute_verdict_consistency_enhanced(results, policy_id)
```

---

## Interpreting Results

### Healthy Results
```
AUPRC               85%+    Method ranks restaurants well
Incident Precision  80%+    Most claims are real incidents
Snippet Validity    95%+    Quotes are accurate
Judgement Accuracy  70%+    Classifications are mostly correct
Score Accuracy      90%+    Point calculations are accurate
Verdict Consistency 90%+    Internal logic is sound
```

### Diagnosing Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Low AUPRC | Wrong verdicts | Check GT accuracy, method prompts |
| Low Incident Precision | Fabricated incidents | Improve evidence extraction |
| Low Snippet Validity | Hallucinated quotes | Add source verification |
| Low Judgement Accuracy | Wrong severity labels | Review classification prompts |
| Low Score Accuracy | Math errors | Check scoring trace logic |
| Low Verdict Consistency | Logic bugs | Audit verdict derivation |

---

## Migration from Old System

### Removed Metrics

| Old | Replacement |
|-----|-------------|
| Process Score | Split into Incident Precision, Judgement Accuracy, Snippet Validity |
| Consistency Score | Now Verdict Consistency (enhanced with triggered_rule) |
| Accuracy | Removed (misleading with class imbalance) |
| Incident Recall | Removed (partial extraction is fine) |

### Legacy Support

Old `unified_scores` still computed for backward compatibility:
```json
{
  "unified_scores": {
    "auprc": 0.85,
    "process_score": 82.5,
    "consistency_score": 100.0
  }
}
```

Use `evaluation_metrics` for new code.

---

## Files

| File | Purpose |
|------|---------|
| `src/addm/eval/metrics.py` | AUPRC, Score Accuracy, Verdict Consistency (enhanced) |
| `src/addm/eval/intermediate_metrics.py` | Incident Precision, Snippet Validity, Judgement Accuracy |
| `src/addm/eval/constants.py` | Scoring rules (SEVERITY_BASE_POINTS, etc.) |
| `src/addm/eval/unified_metrics.py` | Legacy 3-score system (deprecated) |
| `src/addm/eval/__init__.py` | Public exports |

---

## Known Limitations

1. **Verdict Consistency** uses hardcoded V2 scoring rules. Works for G1, may not be accurate for G2-G6 with different scoring schemes.
   - Future: Load policy-specific rules from YAML

2. **Score Accuracy** only for V2/V3. Returns `None` for V0/V1 condition-based policies.

3. **Snippet Validity** requires `reviews_data`. Returns `None` if not provided.

---

## Related Documentation

- [Ground Truth Pipeline](ground_truth.md) - GT generation and human overrides
- [Methods](../../.claude/rules/methods.md) - Method specifications
- [Output Schema](output_system.md) - Structured output format
