# Ground Truth Generation

## Two-Step Flow

### Step 1: Extract L0 Judgments

Multi-model ensemble extraction (default: gpt-5-mini:1, gpt-5-nano:3)

```bash
# Single topic
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 200 --mode 24hrbatch

# All topics
.venv/bin/python -m addm.tasks.cli.extract --k 200 --mode 24hrbatch

# Custom model config
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 200 \
    --models "gpt-5-nano:5,gpt-5-mini:3,gpt-5.1:1"
```

### Step 2: Compute Ground Truth

```bash
# Single policy
.venv/bin/python -m addm.tasks.cli.compute_gt --policy G1_allergy_V2 --k 200

# Multiple policies
.venv/bin/python -m addm.tasks.cli.compute_gt \
    --policy G1_allergy_V0,G1_allergy_V1,G1_allergy_V2,G1_allergy_V3 --k 200

# All 72 policies
.venv/bin/python -m addm.tasks.cli.compute_gt --k 200
```

## Human Override System

Overrides correct LLM judgment errors at the aggregated level.

**File**: `data/answers/yelp/judgment_overrides.json`

**Format**:
```json
{
  "G1_allergy": [
    {
      "review_id": "abc123",
      "corrected": {"incident_severity": "none"},
      "reason": "False positive - no actual allergy incident"
    }
  ]
}
```

**Behavior**:
- Auto-loaded during `compute_gt.py` (logged if found)
- One override applies to all variants (V0-V3) and K values for the topic

## Key Files

| File | Purpose |
|------|---------|
| `src/addm/tasks/policy_gt.py` | Aggregation, scoring, override logic |
| `src/addm/tasks/extraction.py` | PolicyJudgmentCache (raw + aggregated) |
| `data/answers/yelp/judgement_cache.json` | Cached L0 judgments |
| `data/answers/yelp/judgment_overrides.json` | Human corrections |
| `data/answers/yelp/{policy}_K{k}_groundtruth.json` | GT outputs |

See `docs/specs/ground_truth.md` for full details.
