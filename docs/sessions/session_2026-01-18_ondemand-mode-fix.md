# Session Log: Ondemand Mode Logging Improvements

**Date**: 2026-01-18
**Status**: completed

## Summary

Improved the `--mode ondemand` extraction to provide better visibility into what models are being extracted. The code logic was already correct (iterating through all model/run pairs), but lacked clear feedback about progress per model.

## Decisions Made

- Keep existing multi-model iteration logic (it was correct)
- Add pre-extraction summary showing exactly what's needed per model
- Add per-model success/error tracking during extraction
- Add final summary showing results breakdown and final cache status

## Current State

Changes complete in `src/addm/tasks/cli/extract.py` (lines 712-855). The ondemand mode now shows:

1. **Before extraction**: Count of runs needed per model
2. **During extraction**: Status per restaurant (unchanged)
3. **After extraction**:
   - Per-model success/error breakdown
   - Final cache status comparing to quotas

## Next Steps

1. Run actual extraction to complete the 2 missing gpt-5-nano runs:
   ```bash
   .venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --mode ondemand
   ```
2. Continue with Phase I: aggregate G1_allergy GT (as per ROADMAP.md)

## Open Questions

None - the fix is complete.

## Key Files

- `src/addm/tasks/cli/extract.py` - Modified ondemand mode (lines 712-855)

## Context & Background

User reported that `--mode ondemand` "didn't finish all model quota requirement." Investigation showed the code logic was correct - it iterates through all (model, run) pairs from `needs_extraction()`. The issue was lack of visibility - user couldn't see which models were being processed.

The fix adds detailed logging:
- Pre-extraction: "Need to extract N reviews: gpt-5-nano: X runs, gpt-5-mini: Y runs"
- Post-extraction: "Extraction results by model: gpt-5-nano: X/Y success"
- Final cache status verification

Current cache status (as of this session):
- gpt-5-mini: 18440/18440 complete
- gpt-5-nano: 55318/55320 (2 runs still needed)
