# Evaluation Metrics

Simplified evaluation system with **7 separate, interpretable metrics** - no weighted composites.

## Design Philosophy

The old system used weighted composite scores (Process Score = 35% incident precision + 30% severity accuracy + ...). This had problems:
- Arbitrary weights obscured what was actually being measured
- A single "Process Score" hid important distinctions
- Hard to diagnose what went wrong

**New approach:** 7 separate metrics, each measuring one thing clearly.

---

## The Seven Metrics

| Tier | Metric | What It Measures |
|------|--------|------------------|
| **Final Quality** | AUPRC | Ranking quality of verdicts |
| **Final Quality** | Macro F1 | Classification quality (correct predictions) |
| **Evidence Quality** | Evidence Precision | Are claimed evidences real? |
| **Evidence Quality** | Evidence Recall | Were GT evidences found? |
| **Evidence Quality** | Snippet Validity | Are quoted texts real? |
| **Reasoning Quality** | Judgement Accuracy | Are field values correct? |
| **Reasoning Quality** | Verdict Consistency | Does evidence → verdict logic hold? |

---

## Tier 1: Final Quality

### AUPRC (Ordinal Area Under Precision-Recall Curve)

**Purpose:** Measures how well the method **ranks** restaurants by risk level.

**What it measures:** "Are high-risk items ranked above low-risk items?"

**Computation:**
- Binary AUPRC for two thresholds:
  - **≥High**: High+Critical vs Low
  - **≥Critical**: Critical vs rest
- Final: Mean of both AUPRC values

**Why ordinal?** Standard AUPRC treats all errors equally. Ordinal AUPRC respects severity ordering - confusing High/Critical is less wrong than confusing Low/Critical.

**Important caveat:** AUPRC can be perfect (1.0) even with wrong predictions, as long as the ranking order is correct:

```python
# Example where AUPRC=1.0 but all predictions are wrong
GT verdicts:   [Low,  Low,  High, High]
Predictions:   [High, High, Critical, Critical]  # ALL WRONG!
Risk scores:   [1,    2,    3,    4]

AUPRC = 1.0   # Perfect ranking - Highs scored above Lows
Accuracy = 0% # All predictions wrong
```

**Use case:** Triage/prioritization workflows where you need to handle high-risk items first.

**Location:** `src/addm/eval/metrics.py::compute_ordinal_auprc()`

### Macro F1

**Purpose:** Measures how well the method **classifies** restaurants correctly.

**What it measures:** "Are the predictions actually correct?"

**Computation:**
- Compute F1 score for each class (Low, High, Critical)
- Average the F1 scores (macro averaging)

**Why macro?**
- Treats all classes equally, regardless of frequency
- Resistant to class imbalance (won't be inflated by predicting the majority class)
- Penalizes missing entire classes

**Use case:** Decision-making scenarios where the correct label matters, not just relative ordering.

**Location:** `src/addm/eval/metrics.py::compute_evaluation_metrics()` (uses sklearn.metrics.f1_score)

### AUPRC vs Macro F1: When to Use Each

| Scenario | Use | Why |
|----------|-----|-----|
| "Prioritize high-risk for manual review" | AUPRC | Order matters, labels are soft targets |
| "Make automated decisions based on verdict" | Macro F1 | Correct labels matter |
| "Some classes rare but important" | Macro F1 | Won't be dominated by majority class |
| "Benchmark ranking algorithms" | AUPRC | Standard for ranking evaluation |

---

## Tier 2: Evidence Quality

### Evidence Precision

**Purpose:** Are the claimed evidences actually in the ground truth?

**Formula:** `|claimed ∩ GT| / |claimed|` (by review_id)

**Interpretation:**
- 100% = Every claimed evidence exists in GT
- 80% = 20% of claims are fabricated/wrong review IDs
- N/A = Method made no evidence claims (can't compute precision)

**Location:** `src/addm/eval/intermediate_metrics.py::compute_evidence_validity()`

### Evidence Recall

**Purpose:** Did the method find all the evidences in the ground truth?

**Formula:** `|claimed ∩ GT| / |GT|` (by review_id)

**Interpretation:**
- 100% = Method found every GT evidence
- 50% = Method missed half of the GT evidences
- N/A = No GT evidences for this sample

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

### Verdict Consistency (Enhanced)

**Purpose:** Does the method's internal reasoning support its claimed verdict?

**What it checks:** If method claims `triggered_rule: "SCORE >= 5 → Critical Risk"`, does its verdict match "Critical Risk"?

**Consistent if:** The verdict extracted from triggered_rule matches the claimed verdict.

**Rule extraction examples:**
| triggered_rule | Extracted Verdict |
|----------------|------------------|
| `SCORE >= 5 → Critical Risk` | Critical Risk |
| `N_NEGATIVE >= 1 → Not Recommended` | Not Recommended |
| `else → Low Risk` | Low Risk |

**Location:** `src/addm/eval/metrics.py::compute_verdict_consistency_enhanced()`

---

## Scoring Rules Reference (V2 Policy)

Used for verdict derivation in ground truth:

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
EVALUATION METRICS (7 total)
┌─────────────────────┬─────────┬────────────────────────────────┐
│ Metric              │ Score   │ Notes                          │
├─────────────────────┼─────────┼────────────────────────────────┤
│ AUPRC               │ 85.3%   │ (ranking quality)              │
│ Macro F1            │ 72.0%   │ (classification quality)       │
│ Evidence Precision  │ 72.0%   │ (12/15 claimed exist in GT)    │
│ Evidence Recall     │ 60.0%   │ (12/20 GT evidences found)     │
│ Snippet Validity    │ 95.0%   │ (19/20 quotes match source)    │
│ Judgement Accuracy  │ 68.5%   │ (field correctness)            │
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
    "macro_f1": 0.72,
    "evidence_precision": 0.72,
    "evidence_recall": 0.60,
    "snippet_validity": 0.95,
    "judgement_accuracy": 0.685,
    "verdict_consistency": 0.912,
    "n_samples": 100,
    "n_structured": 95,
    "n_with_gt": 100
  },

  "evaluation_metrics_full": {
    "auprc": 0.853,
    "macro_f1": 0.72,
    "evidence_precision": 0.72,
    "evidence_recall": 0.60,
    "snippet_validity": 0.95,
    "judgement_accuracy": 0.685,
    "verdict_consistency": 0.912,
    "details": {
      "auprc_metrics": {
        "auprc_ge_high": 0.83,
        "auprc_ge_critical": 0.88,
        "ordinal_auprc": 0.853
      },
      "evidence_details": {
        "total_claimed": 15,
        "total_matched": 12,
        "total_gt_evidence": 20,
        "total_snippets": 20,
        "valid_snippets": 19
      },
      "judgement_details": {
        "correct": 24,
        "total": 35,
        "per_field": {...}
      },
      "consistency_details": {
        "consistent": 82,
        "total": 90,
        "skipped_no_rule": 5,
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
    policy_id="G1_allergy_V2",   # Optional: for verdict normalization
)

# Access individual metrics
print(f"AUPRC: {metrics['auprc']:.1%}")
print(f"Macro F1: {metrics['macro_f1']:.1%}")
print(f"Evidence Precision: {metrics['evidence_precision']:.1%}")
print(f"Evidence Recall: {metrics['evidence_recall']:.1%}")
print(f"Snippet Validity: {metrics['snippet_validity']:.1%}")
print(f"Judgement Accuracy: {metrics['judgement_accuracy']:.1%}")
print(f"Verdict Consistency: {metrics['verdict_consistency']:.1%}")
```

### Individual Metrics

```python
from addm.eval import (
    compute_ordinal_auprc,
    compute_evidence_validity,
    compute_judgement_accuracy,
    compute_verdict_consistency_enhanced,
)
from sklearn.metrics import f1_score

# AUPRC
auprc_result = compute_ordinal_auprc(y_true_ordinal, y_scores)

# Macro F1
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

# Evidence metrics
evidence_details = compute_evidence_validity(results, gt_data, reviews_data)

# Consistency
consistency, cons_details = compute_verdict_consistency_enhanced(results, policy_id)
```

---

## Interpreting Results

### Healthy Results
```
AUPRC               85%+    Method ranks restaurants well
Macro F1            60%+    Method classifies correctly
Evidence Precision  80%+    Most claims are real evidences
Evidence Recall     60%+    Method finds key evidences (partial OK)
Snippet Validity    95%+    Quotes are accurate
Judgement Accuracy  70%+    Classifications are mostly correct
Verdict Consistency 90%+    Internal logic is sound
```

### Diagnosing Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| High AUPRC, Low Macro F1 | Good ranking but wrong labels | Check classification threshold/calibration |
| Low AUPRC, High Macro F1 | Good labels but bad scores | Check risk_score computation |
| Low Evidence Precision | Fabricated evidences | Improve evidence extraction |
| Low Evidence Recall | Missing evidences | Increase retrieval coverage |
| Low Snippet Validity | Hallucinated quotes | Add source verification |
| Low Judgement Accuracy | Wrong severity labels | Review classification prompts |
| Low Verdict Consistency | Logic bugs | Audit verdict derivation |

---

## Migration from Old System

### Removed Metrics

| Old | Replacement |
|-----|-------------|
| Process Score | Split into Evidence Precision, Judgement Accuracy, Snippet Validity |
| Consistency Score | Now Verdict Consistency (enhanced with triggered_rule) |
| Accuracy | Replaced by Macro F1 (resistant to class imbalance) |
| Incident Recall | Renamed to Evidence Recall |
| Score Accuracy | Removed (V2/V3-specific, not generalizable) |

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
| `src/addm/eval/metrics.py` | AUPRC, Macro F1, Verdict Consistency |
| `src/addm/eval/intermediate_metrics.py` | Evidence Precision/Recall, Snippet Validity, Judgement Accuracy |
| `src/addm/eval/constants.py` | Scoring rules (SEVERITY_BASE_POINTS, etc.) |
| `src/addm/eval/unified_metrics.py` | Legacy 3-score system (deprecated) |
| `src/addm/eval/__init__.py` | Public exports |

---

## Known Limitations

1. **Verdict Consistency** uses hardcoded V2 scoring rules. Works for G1, may not be accurate for G2-G6 with different scoring schemes.
   - Future: Load policy-specific rules from YAML

2. **Snippet Validity** requires `reviews_data`. Returns `None` if not provided.

---

## Related Documentation

- [Ground Truth Pipeline](ground_truth.md) - GT generation and human overrides
- [Methods](../../.claude/rules/methods.md) - Method specifications
- [Output Schema](output_system.md) - Structured output format
