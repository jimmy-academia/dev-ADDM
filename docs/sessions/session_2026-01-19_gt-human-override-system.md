# Session Log: GT Human Override System

**Date**: 2026-01-19
**Status**: completed

## Summary

Implemented a systematic GT Human Override System that allows human corrections to LLM judgment errors. Overrides are stored in a JSON file and applied at the aggregated judgment level during GT computation. Also fixed the cuisine modifier bug that applied +2 points even when no incidents existed.

## Decisions Made

- **Override Level**: Aggregated judgments (not raw, not GT output) - one override applies to all policy variants (V0-V3) and K values
- **File Format**: Single JSON file `data/answers/yelp/judgment_overrides.json` for all topics
- **Override applies before scoring**: Human corrections are applied in `compute_gt_from_policy_scoring()` before the scoring loop

## Work Completed

1. Created `data/answers/yelp/judgment_overrides.json` with 2 G1_allergy corrections:
   - `a-ptEPsf-ea_7tpK2yoimw` (Chop Steakhouse): severity "severe" → "none" (positive review)
   - `roT4J43wWrc5gxQCWJ5fzQ` (South Coast Deli): severity "mild" → "none" (no symptoms)

2. Added to `src/addm/tasks/policy_gt.py`:
   - `load_overrides(topic)` - loads overrides from JSON, indexed by review_id
   - `apply_overrides(judgment, overrides)` - applies corrections, adds `_override_applied` flag
   - Updated function signatures to accept `overrides` parameter

3. Fixed cuisine modifier bug (line 419):
   ```python
   # Before: if is_high_risk_cuisine(categories):
   # After:  if is_high_risk_cuisine(categories) and len(incidents) > 0:
   ```

4. Updated `src/addm/tasks/cli/compute_gt.py`:
   - Imports `load_overrides`
   - Loads and logs override count
   - Passes overrides to GT computation

5. Regenerated `G1_allergy_V2_K50_groundtruth.json` - clean output, no inline `_gt_adjustment` fields

6. Deleted `docs/gt_adjustments_G1_allergy_V2.md` (content migrated to override file)

## Current State

Override system is fully functional. All 6 implementation tasks completed:
- Override file created
- Override functions added
- Cuisine modifier bug fixed
- CLI integration done
- GT regenerated and verified
- Old adjustment file deleted

## Next Steps

1. Run AMOS evaluation to verify >75% on all three metrics:
   ```bash
   .venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 100 --method amos --dev --k 50
   ```

2. If more judgment errors found, add to `judgment_overrides.json`

3. Regenerate GT for other K values (25, 100, 200) if needed:
   ```bash
   .venv/bin/python -m addm.tasks.cli.compute_gt --topic G1_allergy
   ```

## Open Questions

None - implementation is complete

## Key Files

- `data/answers/yelp/judgment_overrides.json` - **NEW**: Human overrides for all topics
- `src/addm/tasks/policy_gt.py` - Override functions + cuisine modifier fix (lines 28-78, 419)
- `src/addm/tasks/cli/compute_gt.py` - CLI integration (lines 29-30, 180-183, 234)
- `data/answers/yelp/G1_allergy_V2_K50_groundtruth.json` - Regenerated GT (clean)

## Context & Background

This system addresses the issue of LLM extraction errors in ground truth:
- LLMs sometimes misclassify positive reviews as incidents
- LLMs sometimes detect "allergic reaction" keywords without actual symptoms
- The override system allows human correction without polluting raw cache

Override file format:
```json
{
  "G1_allergy": [
    {
      "review_id": "...",
      "business_id": "...",
      "original": {"incident_severity": "severe"},
      "corrected": {"incident_severity": "none"},
      "reason": "..."
    }
  ]
}
```
