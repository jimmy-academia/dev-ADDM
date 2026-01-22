# Ground Truth Generation

This document describes the policy-based ground truth (GT) generation system for ADDM benchmark tasks.

## Overview

Ground truth generation follows a two-step flow:

1. **Step 1: Extract** - Extract L0 judgments from reviews using multi-model ensemble
2. **Step 2: Compute** - Apply policy scoring rules to produce verdicts

This separation allows:
- Shared L0 judgments across policy variants (V0-V3)
- Multi-model aggregation for robustness
- Incremental extraction with quota tracking

## Multi-Model Strategy

Since batch mode is ~50% cheaper, we use savings for robustness via multi-model ensemble.

### Default Config (Fast)

Used for initial extraction runs:

| Model | Runs | Weight | Total Weight |
|-------|------|--------|--------------|
| gpt-5-mini | 1 | 2 | 2 |
| gpt-5-nano | 3 | 1 | 3 |
| **Total** | 4 | - | **5** |

### High-Quality Config

For production ground truth (more expensive, more robust):

| Model | Runs | Weight | Total Weight |
|-------|------|--------|--------------|
| gpt-5.1 | 1 | 3 | 3 |
| gpt-5-mini | 3 | 2 | 6 |
| gpt-5-nano | 5 | 1 | 5 |
| **Total** | 9 | - | **14** |

Use `--models "gpt-5.1:1,gpt-5-mini:3,gpt-5-nano:5"` to specify high-quality config.

### Weighted Majority Voting

For each L0 field, the aggregated value is determined by weighted majority vote:

```python
aggregated[field] = max(votes.items(), key=lambda x: x[1])
confidence[field] = winning_weight / total_weight
```

Example: For `incident_severity`:
- gpt-5.1 says "moderate" (weight 3)
- gpt-5-mini run1 says "mild" (weight 2)
- gpt-5-mini run2 says "moderate" (weight 2)
- gpt-5-mini run3 says "moderate" (weight 2)
- gpt-5-nano run1-5 all say "moderate" (weight 5)

Result: "moderate" wins with weight 12/14 = 0.857 confidence

## Cache Structure

Policy-based extraction uses a dual cache for raw and aggregated results:

```json
{
  "_metadata": {
    "G1_allergy": {
      "term_hash": "a1b2c3d4e5f6g7h8",
      "created_at": "2024-01-18T10:00:00",
      "model_config": {"gpt-5.1": 1, "gpt-5-mini": 3, "gpt-5-nano": 5}
    }
  },
  "raw": {
    "G1_allergy::review123::gpt-5.1::run1": {...judgment...},
    "G1_allergy::review123::gpt-5-mini::run1": {...judgment...},
    ...
  },
  "aggregated": {
    "G1_allergy::review123": {...aggregated_judgment...}
  }
}
```

### Term Version Tracking

L0 schema is derived from policy term references. A SHA-256 hash of the schema is stored in metadata.

On subsequent runs:
- Compare current term hash with cached hash
- If mismatch: warn user, offer to invalidate with `--invalidate`
- Prevents stale cache from producing incorrect GT

## CLI Usage

### Step 1: Extract L0 Judgments

```bash
# Extract for a topic (shared by V0-V3 policies)
.venv/bin/python -m addm.tasks.cli.extract \
    --topic G1_allergy --k 50 --mode batch

# Or derive topic from policy
.venv/bin/python -m addm.tasks.cli.extract \
    --policy G1_allergy_V2 --k 50 --mode batch
```

This submits a batch with 9 requests per review (multi-model) and installs a cron job to poll for completion.

### Step 2: Compute Ground Truth

```bash
# Single policy
.venv/bin/python -m addm.tasks.cli.compute_gt \
    --policy G1_allergy_V2 --k 50

# Multiple policies (same topic, shared judgments)
.venv/bin/python -m addm.tasks.cli.compute_gt \
    --policy G1_allergy_V0,G1_allergy_V1,G1_allergy_V2,G1_allergy_V3 --k 50
```

## Output Format

Ground truth files are saved to:
```
data/answers/{domain}/{policy_id}_K{k}_groundtruth.json
```

Example output:
```json
{
  "policy_id": "G1_allergy_V2",
  "topic": "G1_allergy",
  "domain": "yelp",
  "k": 50,
  "computed_at": "2024-01-18T12:00:00",
  "has_scoring": true,
  "restaurants": {
    "biz123": {
      "name": "Thai Kitchen",
      "categories": "Thai, Asian",
      "n_reviews": 50,
      "n_judgments": 50,
      "ground_truth": {
        "verdict": "Critical Risk",
        "score": 12,
        "n_incidents": 3,
        "incidents": [
          {"review_id": "rev1", "severity": "severe", "base_points": 15, "modifiers": [], "total_points": 15},
          ...
        ],
        "cuisine_modifier": true,
        "policy_id": "G1_allergy_V2"
      }
    }
  },
  "summary": {
    "total_restaurants": 100,
    "verdict_distribution": {"Low Risk": 60, "High Risk": 30, "Critical Risk": 10},
    "missing_judgments": 0
  }
}
```

## Policy Scoring (V2+)

Policies with scoring systems (V2+) use point-based GT computation:

1. **Base points**: Per incident severity
   - Mild: 2 points
   - Moderate: 5 points
   - Severe: 15 points

2. **Modifiers**: Applied to incidents
   - False assurance: +5 points
   - Dismissive staff: +3 points
   - High-risk cuisine: +2 points (restaurant-level)

3. **Thresholds**: Score → Verdict
   - Score ≥ 8: Critical Risk
   - Score ≥ 4: High Risk
   - Score < 4: Low Risk

## Qualitative GT (V0/V1)

Policies without scoring use rule-based evaluation:

1. Evaluate structured conditions in order (precedence ladder)
2. First matching rule determines verdict
3. Default rule applies if no others match

## Incremental Extraction

The system supports incremental extraction:

- Tracks which (model, run) combinations are cached per review
- On re-run, only extracts missing runs
- Aggregates when all quotas satisfied

Example:
```bash
# First run: extracts all 9 runs per review
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 50 --mode batch

# Batch fails partway through...

# Second run: only extracts missing runs
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 50 --mode batch
```

## Human Override System

LLM extraction sometimes makes errors (e.g., misclassifying positive reviews as incidents). The override system allows human corrections without modifying the raw cache.

### Override Level

Overrides apply at the **aggregated judgment level**:
- One override affects all policy variants (V0-V3) for that topic
- One override affects all K values (25, 50, 100, 200)
- Original LLM outputs preserved in raw cache for analysis

### Override File

Location: `data/answers/yelp/judgment_overrides.json`

Format:
```json
{
  "_comment": "Human corrections to LLM judgment errors.",
  "created_at": "2026-01-19",
  "last_updated": "2026-01-19",
  "G1_allergy": [
    {
      "review_id": "abc123",
      "business_id": "biz456",
      "original": {"incident_severity": "severe"},
      "corrected": {"incident_severity": "none"},
      "reason": "Positive review - customer praised allergy handling, no reaction occurred"
    }
  ]
}
```

### How It Works

1. `compute_gt.py` calls `load_overrides(topic)` at startup
2. Returns dict mapping `review_id` → override spec
3. `apply_overrides()` modifies judgment before scoring loop
4. Overridden judgments get `_override_applied: true` flag

### Adding Overrides

1. Identify incorrect judgments via evaluation failures
2. Check raw cache to see original LLM outputs
3. Add entry to `judgment_overrides.json` with:
   - `review_id` (required)
   - `business_id` (for documentation)
   - `original` (what LLM said, for audit)
   - `corrected` (field values to override)
   - `reason` (why correction is needed)
4. Regenerate GT: `.venv/bin/python -m addm.tasks.cli.compute_gt --policy G1_allergy_V2 --k 50`

## Files

| File | Description |
|------|-------------|
| `src/addm/tasks/policy_gt.py` | Core GT computation, override functions |
| `src/addm/tasks/extraction.py` | JudgmentCache, PolicyJudgmentCache |
| `src/addm/tasks/cli/extract.py` | Extraction CLI |
| `src/addm/tasks/cli/compute_gt.py` | GT computation CLI |
| `data/answers/yelp/judgement_cache.json` | Raw + aggregated L0 judgments |
| `data/answers/yelp/judgment_overrides.json` | Human corrections to LLM errors |

## Legacy Task Mode

The legacy `--task` flag still works for backwards compatibility:

```bash
# Extract (single model, uses task-to-topic mapping)
.venv/bin/python -m addm.tasks.cli.extract --task G1a --k 50

# Compute GT (uses task-to-policy mapping)
.venv/bin/python -m addm.tasks.cli.compute_gt --task G1a --k 50
```

Legacy mode:
- Maps task IDs (G1a) to topics (G1_allergy) via hardcoded mapping
- Uses single model extraction (no multi-model ensemble)
- Maintained for backward compatibility with existing scripts
- **Note**: Policy-based mode (`--policy`, `--topic`) is recommended for new work
