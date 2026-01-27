# Session: T1 P3 ALL Logic Fix

**Date**: 2026-01-27
**Status**: Complete
**Topic**: Fix T1P3 to properly test ALL logic with non-zero GT distribution

## Summary

Fixed T1P3 (ALL logic variant) which was failing due to:
1. `EXTRACT_VERDICTS_PROMPT` only showed ANY logic examples
2. P3 rules were too complex/strict, causing 0 High/Critical at some K values

## Changes Made

### 1. Updated EXTRACT_VERDICTS_PROMPT (`src/addm/methods/amos/phase1_prompts.py`)

Added ALL logic example and detection rule:
```yaml
# Example with ALL logic (AND conditions - ALL must be true simultaneously)
- verdict: Verdict2
  logic: ALL
  conditions:
    - field: INCIDENT_SEVERITY
      values: [severe]
      min_count: 1
    - field: INCIDENT_SEVERITY
      values: [moderate]
      min_count: 1
```

Added rule #7:
```
7. LOGIC DETECTION:
   - If text says "if **all** of the following are true", "AND", or "both...and" → use `logic: ALL`
   - If text says "if **any** of the following is true", "OR", or "either...or" → use `logic: ANY`
   - Default to `logic: ANY` if not specified
```

### 2. Redesigned T1P3 Rules (`src/addm/query/policies/T1/P3.yaml`)

**Pattern**: Copy P1's threshold + add confirmatory condition from sample restaurants

Analyzed P1 K25 samples:
- **Critical (Cafe Blue Moose)**: 3 moderate incidents, all have `staff_response=dismissive`
- **High (Front Street Cafe)**: 1 moderate incident, has `assurance_claim=true`

New P3 rules:
```yaml
# Critical: P1's threshold (2+ moderate) AND staff was dismissive
- verdict: "Critical Risk"
  logic: ALL
  conditions:
    - text: "2 or more moderate firsthand incidents are reported"
    - text: "staff response was dismissive when the allergy concern was raised"

# High: P1's threshold (1+ mild/moderate) AND assurance was given
- verdict: "High Risk"
  logic: ALL
  conditions:
    - text: "1 or more mild or moderate firsthand incidents are reported"
    - text: "an assurance of safety was given (staff claimed the food was safe)"
```

### 3. Regenerated GT and Samples

Final T1P3 distribution:
| K | Low | High | Critical |
|---|-----|------|----------|
| 25 | 98 | 1 | 1 |
| 50 | 98 | 1 | 1 |
| 100 | 96 | 3 | 1 |
| 200 | 93 | 6 | 1 |

## Files Modified

- `src/addm/methods/amos/phase1_prompts.py` - Added ALL logic example
- `src/addm/query/policies/T1/P3.yaml` - Redesigned ALL logic rules
- `data/answers/yelp/T1P3_K*_groundtruth.json` - Regenerated
- `data/answers/yelp/verdict_sample_ids.json` - Updated samples
- `.claude/rules/benchmark.md` - Documented T1 status and P3 pattern

## Next Steps

**Tomorrow**: Apply same pattern to T2-T5 P3 policies:
1. Check T{N}P1 sample restaurants
2. Find what additional property Critical/High samples have
3. Add as condition 2 for T{N}P3
4. Regenerate GT and verify non-zero distribution at all K

## Key Insight

For P3 ALL logic to work:
- Condition 1: Same threshold as P1 (ensures semantic equivalence)
- Condition 2: Property that P1's sample restaurants already have (ensures same restaurants trigger)

This tests ALL logic parsing without changing which restaurants are Critical/High.
