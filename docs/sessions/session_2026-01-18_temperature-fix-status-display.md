# Session Log: Temperature Fix & Status Display

**Date**: 2026-01-18 ~14:00
**Status**: completed

## Summary

Fixed the temperature=0 issue that caused all gpt-5-mini batch requests to fail. Removed temperature parameter from all LLM calls (models use default 1.0). Added per-model cache status display and fixed `--dry-run` to work in batch mode.

## Decisions Made

- **Remove temperature entirely**: Don't pass temperature parameter to any model - all use default 1.0. Some models (gpt-5-mini, gpt-5-nano) don't support temperature=0.
- **Cache is source of truth**: Check cache for missing extractions, not batch failure counts
- **Use `output` singleton**: Converted all `print()` to `output.info/warn/success/print/status`
- **Status display**: Show per-model cache status immediately after loading cache

## Changes Made

### Temperature Removal
- `src/addm/llm.py` - Removed temperature from config and API calls
- `src/addm/llm_batch.py` - Removed temperature from `build_chat_batch_item`
- `src/addm/tasks/cli/extract.py` - Removed 6 temperature=0.0 calls
- `src/addm/tasks/cli/run_baseline.py` - Removed 2 temperature=0.0 calls
- `src/addm/runner.py` - Removed temperature from configure()
- `src/addm/cli.py` - Marked --temperature as ignored

### Status Display
- Added `count_raw_by_model()` method to `PolicyJudgmentCache`
- Added per-model cache status output after loading cache:
  ```
  ℹ Cache status by model:
    gpt-5-mini: 0/18440 (need 18440)
    gpt-5-nano: 55318/55320 ✓ complete
  ```

### Dry-Run Fix
- `--dry-run` now works in batch mode (skips batch submission)

### Output Cleanup
- Converted all `print()` calls to `output.()` in extract.py

## Current State

- gpt-5-nano: 55,318/55,320 complete (just 2 missing)
- gpt-5-mini: 0/18,440 (all missing due to temperature=0 failure)
- Ready to re-run extraction to submit gpt-5-mini batch

## Next Steps

1. Submit the missing extractions:
   ```bash
   .venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --mode batch
   ```
2. After batch completes (~24hrs), aggregation will auto-run
3. Then compute ground truth:
   ```bash
   .venv/bin/python -m addm.tasks.cli.compute_gt --policy G1_allergy_V2 --k 200
   ```

## Key Files

- `src/addm/tasks/cli/extract.py` - Main extraction CLI (status display, dry-run)
- `src/addm/tasks/extraction.py` - PolicyJudgmentCache (count_raw_by_model)
- `src/addm/llm.py` - LLM service (temperature removed)
- `src/addm/llm_batch.py` - Batch API (temperature removed)

## Context & Background

- Root cause: OpenAI gpt-5-mini model doesn't support `temperature=0`
- All 18,440 batch requests failed with: "Unsupported value: 'temperature' does not support 0 with this model"
- Fix: Remove temperature entirely, let models use their default (1.0)
