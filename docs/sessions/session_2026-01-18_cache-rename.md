# Session Log: Cache File Rename

**Date**: 2026-01-18
**Status**: completed

## Summary

Renamed `policy_cache.json` to `judgement_cache.json` per user preference. Also confirmed that G1_allergy extraction is fully complete (all model quotas met).

## Decisions Made

- Renamed cache file: `policy_cache.json` → `judgement_cache.json`
- Renamed function: `_get_policy_cache_path()` → `_get_judgement_cache_path()`
- Spelling: "judgement" (British) used consistently

## Current State

G1_allergy extraction is 100% complete:
- gpt-5-mini: 18440/18440
- gpt-5-nano: 55320/55320
- Aggregated: 18440 entries

GT files exist for all V0-V3 variants.

## Next Steps

1. Continue with Phase I as per ROADMAP.md
2. Implement baselines
3. AMOS development

## Key Files Modified

- `data/tasks/yelp/judgement_cache.json` - renamed from policy_cache.json
- `src/addm/tasks/cli/extract.py` - updated function name and calls
- `src/addm/tasks/cli/compute_gt.py` - updated function name and call
- `.claude/CLAUDE.md` - updated documentation references
- `.gitignore` - updated pattern

## Context & Background

This session was a brief continuation to rename the cache file. The previous session (`session_2026-01-18_extraction-bugfixes.md`) fixed critical bugs in the extraction pipeline:
- Fixed `needs_extraction()` to check for SPECIFIC run numbers (not just count)
- Added `--topic` flag to `compute_gt.py`
- Removed slow pre-scan in ondemand mode
