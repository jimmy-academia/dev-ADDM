# Future: High-Quality Ground Truth Configuration

## Status: Deferred (cost)

## Overview

The original multi-model GT extraction strategy for maximum robustness.

## Original Configuration

```python
MODEL_WEIGHTS = {
    "gpt-5.1": 3,      # Highest quality, highest weight
    "gpt-5-mini": 2,   # Medium quality
    "gpt-5-nano": 1,   # Lowest quality, lowest weight
}

REQUIRED_RUNS = {
    "gpt-5.1": 1,      # 1 run
    "gpt-5-mini": 3,   # 3 runs
    "gpt-5-nano": 5,   # 5 runs
}

TOTAL_WEIGHT = 14  # 1×3 + 3×2 + 5×1 = 14
```

## Cost Estimate

- 18 topics × 100 restaurants × 200 reviews × 9 runs = 3.24M requests
- Estimated cost: ~$500

## Current Configuration (Cost-Optimized)

```python
MODEL_WEIGHTS = {
    "gpt-5-mini": 2,
    "gpt-5-nano": 1,
}

REQUIRED_RUNS = {
    "gpt-5-mini": 1,
    "gpt-5-nano": 3,
}

TOTAL_WEIGHT = 5  # 1×2 + 3×1 = 5
```

- 18 topics × 100 restaurants × 200 reviews × 4 runs = 1.44M requests
- Estimated cost: ~$50-80

## When to Use High-Quality Config

Consider upgrading to high-quality config when:
1. Publishing final benchmark results
2. Need highest confidence in GT labels
3. Budget allows for ~$500 extraction cost

## How to Switch

Update constants in:
- `src/addm/tasks/extraction.py`: REQUIRED_RUNS
- `src/addm/tasks/policy_gt.py`: MODEL_WEIGHTS, REQUIRED_RUNS, TOTAL_WEIGHT

Then re-run extraction (cache will detect quota change and extract additional runs).
