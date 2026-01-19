# Session Log: Batch Extraction Fixes

**Date**: 2026-01-18 04:30
**Status**: in-progress

## Summary

Fixed OpenAI Batch API "mismatched_model" error by implementing model-based batch splitting. Cleaned up failed batch artifacts, improved cron notifications, and updated gitignore. Session ready for G1_allergy extraction run.

## Decisions Made

- **Split batches by model first**: OpenAI Batch API requires single model per batch. Created `_split_by_model_then_size()` to group requests by model before chunking
- **Gitignore batch artifacts**: Added `batch_manifest_*.json` and `policy_cache.json` to .gitignore (temporary, machine-specific)
- **Platform-specific cron warning**: Only show "cron output may not appear in terminal" warning on Linux, not Mac
- **Deleted invalid GT files**: K=25 groundtruth files had `verdict: None` (never ran with proper LLM calls) - deleted them
- **Orphaned data/topic_selection/**: User manually deleted (not referenced in codebase)

## Current State

- Batch extraction is ready to run for G1_allergy topic
- All prior failed batch artifacts cleaned up (manifest deleted, stale cron removed)
- Cache is empty and ready for fresh extraction
- Split-by-model logic implemented and tested

## Next Steps

1. **Run G1_allergy extraction**:
   ```bash
   .venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy
   ```
   - Cost estimate: ~$3.32 (with 50% Batch API discount)
   - Creates 3 batches: 1 for gpt-5-mini, 2 for gpt-5-nano
   - Cron job will poll every 5 minutes

2. After batch completes, run GT computation:
   ```bash
   .venv/bin/python -m addm.tasks.cli.compute_gt --policy G1_allergy_V0,G1_allergy_V1,G1_allergy_V2,G1_allergy_V3 --k 50
   ```

3. Verify results before scaling to all 18 topics

## Open Questions

- None at this point - ready to execute

## Key Files

- `src/addm/tasks/cli/extract.py` - Added `_split_by_model_then_size()` and `_print_cron_installed()`
- `src/addm/tasks/policy_gt.py` - Fixed `build_l0_schema_from_topic()` to use union of V0-V3 terms
- `.gitignore` - Added batch_manifest and policy_cache patterns
- `data/tasks/yelp/policy_cache.json` - Cache for L0 judgments (currently empty)

## Context & Background

### Multi-Model Extraction Strategy
- gpt-5-mini: 1 run per review (weight 2)
- gpt-5-nano: 3 runs per review (weight 1)
- Total: 4 runs per review, aggregated via weighted majority vote

### OpenAI Batch API Constraint
Each batch must contain requests for a single model only. The `_split_by_model_then_size()` function now:
1. Groups all requests by model
2. Then splits each model's requests into chunks of MAX_BATCH_SIZE (40,000)
3. Returns list of (model, items) tuples for batch submission

### Cost Breakdown (G1_allergy, K=50)
- 4610 reviews to process
- gpt-5-mini: 4610 requests @ $0.30/1M + $1.20/1M tokens
- gpt-5-nano: 13830 requests @ $0.075/1M + $0.30/1M tokens
- Batch discount: 50%
- Total: ~$3.32

### Plan File Reference
Full implementation plan at: `/Users/jimmyyeh/.claude/plans/spicy-yawning-pike.md`
