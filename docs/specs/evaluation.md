# Evaluation Metrics

Overview of the unified 3-score evaluation system for ADDM methods.

## The Three Scores

Every method run produces three scores (0-100%). Target: **>75% for each**.

| Score | What It Measures | Location |
|-------|------------------|----------|
| **AUPRC** | Ranking quality of final verdicts | `src/addm/eval/metrics.py` |
| **Process** | Quality of intermediate reasoning steps | `src/addm/eval/unified_metrics.py` |
| **Consistency** | Alignment between evidence and verdict | `src/addm/eval/unified_metrics.py` |

---

## 1. AUPRC (Ranking Quality)

**Purpose:** Measures how well the method ranks restaurants by risk level.

**Computation:**
- Ordinal AUPRC averaging two binary tasks:
  - **AUPRC ≥High**: High+Critical vs Low (threshold=1)
  - **AUPRC ≥Critical**: Critical vs rest (threshold=2)
- Final score: Mean of both AUPRC values

**Location:** `src/addm/eval/metrics.py::compute_ordinal_auprc()`

**Why ordinal AUPRC?**
- Standard AUPRC treats all misclassifications equally
- Ordinal AUPRC respects the severity ordering (Low < High < Critical)
- A method that confuses High/Critical is less wrong than one confusing Low/Critical

**Example:**
```python
from addm.eval.metrics import compute_ordinal_auprc
import numpy as np

y_true = np.array([0, 0, 1, 2])  # Low, Low, High, Critical
y_scores = np.array([1.0, 2.0, 6.0, 10.0])  # Risk scores

result = compute_ordinal_auprc(y_true, y_scores)
# result["ordinal_auprc"] = mean of auprc_ge_high and auprc_ge_critical
```

---

## 2. Process Score (Reasoning Quality)

**Purpose:** Evaluates the quality of intermediate reasoning steps, not just final verdicts.

**Computation:** Weighted average of five components:

| Component | Weight | Description |
|-----------|--------|-------------|
| `incident_precision` | 35% | Found correct incidents (claimed ∩ GT / claimed) |
| `severity_accuracy` | 30% | Correct severity classification for matched incidents |
| `modifier_accuracy` | 15% | Correct modifier detection (False assurance, Dismissive staff) |
| `verdict_support_rate` | 15% | Does claimed evidence support the claimed verdict? |
| `snippet_validity` | 5% | Are quoted snippets actually in the review text? |

**Note:** If `snippet_validity` is null (AMOS doesn't use snippets), the weight is redistributed proportionally among other components.

**Location:** `src/addm/eval/unified_metrics.py::compute_process_score()`

**Three-stage evaluation:**
1. **Stage 1 - Evidence Validity:** Are claimed evidence items actually valid?
2. **Stage 2 - Classification Accuracy:** For valid matches, are classifications correct?
3. **Stage 3 - Verdict Support:** Does claimed evidence support the verdict?

**Location:** `src/addm/eval/intermediate_metrics.py::compute_intermediate_metrics()`

---

## 3. Consistency Score (Verdict-Evidence Alignment)

**Purpose:** Checks if the method's claimed verdict matches what would be computed from its claimed evidence.

**Computation:**
1. Extract method's claimed evidences
2. Recompute what verdict SHOULD be using V2 policy scoring rules
3. Compare to method's claimed verdict
4. Score = (consistent / total) × 100

**Location:** `src/addm/eval/unified_metrics.py::compute_consistency_score()`

**Why this matters:**
- A method might find correct incidents but compute the wrong verdict
- This score catches "math errors" in the method's scoring logic
- 100% consistency means the method's internal logic is sound

---

## Scoring Rules (V2 Policy)

The evaluation uses V2 policy scoring rules for consistency checks:

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

**Example:**
- 1 moderate incident (5 pts) + false assurance (+5 pts) = 10 pts → Critical Risk
- 2 mild incidents (2×2 = 4 pts) → High Risk
- 1 mild incident (2 pts) → Low Risk

---

## results.json Structure

Each run produces a `results.json` file with this structure:

```json
{
  "run_id": "G1_allergy_V2",
  "policy_id": "G1_allergy_V2",
  "domain": "yelp",
  "method": "amos",
  "model": "gpt-5-nano",
  "k": 50,
  "n": 100,
  "timestamp": "20260119_004007",

  "accuracy": 0.97,
  "correct": 97,
  "total": 100,

  "auprc": {
    "auprc_ge_high": 0.73,
    "auprc_ge_critical": 0.87,
    "ordinal_auprc": 0.80,
    "n_samples": 100
  },

  "unified_scores": {
    "auprc": 0.80,
    "process_score": 86.15,
    "consistency_score": 100.0
  },

  "unified_metrics_full": {
    "auprc": 0.80,
    "process_score": 86.15,
    "process_components": {
      "incident_precision": 1.0,
      "severity_accuracy": 0.57,
      "modifier_accuracy": 1.0,
      "verdict_support_rate": 0.98,
      "snippet_validity": null,
      "weighted_sum": 0.82,
      "total_weight": 0.95
    },
    "consistency_score": 100.0,
    "consistency_details": {
      "consistent": 5,
      "total": 5,
      "per_sample": [...]
    },
    "intermediate_metrics": {...}
  },

  "results": [...]  // Per-sample results array
}
```

### Key Fields

| Field | Description |
|-------|-------------|
| `unified_scores` | The 3 main scores (AUPRC, Process, Consistency) |
| `process_components` | Breakdown of Process Score components |
| `consistency_details` | Per-sample consistency analysis |
| `accuracy` | Simple verdict accuracy (correct/total) |
| `results` | Array of per-sample method outputs |

---

## Console Output

When running `run_baseline.py`, you'll see output like:

```
═══════════════════════════════════════════════════════════════════
 RUN: G1_allergy_V2 | Method: amos | K=50 | N=100
═══════════════════════════════════════════════════════════════════

 UNIFIED SCORES
 ┌──────────────────┬─────────┬────────┐
 │ Metric           │ Score   │ Status │
 ├──────────────────┼─────────┼────────┤
 │ AUPRC            │ 79.8%   │ ✓      │
 │ Process          │ 86.2%   │ ✓      │
 │ Consistency      │ 100.0%  │ ✓      │
 └──────────────────┴─────────┴────────┘

 PROCESS COMPONENTS
 ┌─────────────────────┬─────────┬────────┐
 │ Component           │ Score   │ Weight │
 ├─────────────────────┼─────────┼────────┤
 │ Incident Precision  │ 100.0%  │ 35%    │
 │ Severity Accuracy   │ 57.1%   │ 30%    │
 │ Modifier Accuracy   │ 100.0%  │ 15%    │
 │ Verdict Support     │ 98.0%   │ 15%    │
 │ Snippet Validity    │ N/A     │ 5%     │
 └─────────────────────┴─────────┴────────┘
```

Use `show_results.py` CLI to generate formatted summaries:

```bash
# Show latest benchmark results
.venv/bin/python -m addm.eval.cli.show_results

# Filter by method
.venv/bin/python -m addm.eval.cli.show_results --method amos

# Show specific dev run
.venv/bin/python -m addm.eval.cli.show_results --dev results/dev/20260119_004007_G1_allergy_V2/
```

---

## Files

| File | Purpose |
|------|---------|
| `src/addm/eval/unified_metrics.py` | Main 3-score computation, AMOS normalization |
| `src/addm/eval/intermediate_metrics.py` | Process score components (evidence validity, classification, verdict support) |
| `src/addm/eval/metrics.py` | AUPRC computation, primitive accuracy |
| `src/addm/eval/cli/show_results.py` | CLI for formatted result display |

---

## API Usage

```python
from addm.eval.unified_metrics import compute_unified_metrics

# Main entry point
metrics = compute_unified_metrics(
    results=method_outputs,      # List of method result dicts
    gt_data=gt_with_incidents,   # {business_id: {verdict, incidents, ...}}
    gt_verdicts=gt_verdicts,     # {business_id: verdict}
    method="amos",               # Method name
    reviews_data=reviews_data,   # Optional: for snippet validation
)

# Access scores
print(f"AUPRC: {metrics['auprc']:.1%}")
print(f"Process: {metrics['process_score']:.1f}%")
print(f"Consistency: {metrics['consistency_score']:.1f}%")
```

---

## Interpreting Results

### Good Results (>75% all three)
- AUPRC 80%, Process 86%, Consistency 100%
- Method ranks restaurants well, finds correct incidents, internal logic is sound

### Process Score Issues
- **Low incident_precision:** Method finding non-existent incidents
- **Low severity_accuracy:** Misclassifying incident severity
- **Low verdict_support_rate:** Evidence doesn't justify verdict

### Consistency Issues
- **<100% consistency:** Method has scoring bugs
- Check `consistency_details.per_sample` for specific failures

### AUPRC Issues
- **Low auprc_ge_high:** Confusing Low/High boundary
- **Low auprc_ge_critical:** Confusing High/Critical boundary
- May indicate need for human overrides to correct GT

---

## Related Documentation

- `docs/specs/ground_truth.md` - GT generation and human override system
- `docs/BASELINES.md` - Method specifications (direct, rag, rlm, amos)
- `docs/tasks/TAXONOMY.md` - 72 task definitions
