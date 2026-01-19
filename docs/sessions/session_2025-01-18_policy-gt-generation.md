# Session Log: Policy-Based Ground Truth Generation

**Date**: 2025-01-18 ~15:00
**Status**: in-progress

## Summary

Implemented the full policy-based ground truth generation system with multi-model extraction, weighted majority voting, and two-step flow (extract → compute_gt). Tested both ondemand and 24hrbatch modes successfully with small samples.

## Decisions Made

- **Two-step flow**: `extract.py` for L0 judgments → `compute_gt.py` for verdict computation
- **Multi-model ensemble**: gpt-5.1 (1 run, weight 3) + gpt-5-mini (3 runs, weight 2) + gpt-5-nano (5 runs, weight 1) = 9 runs per review, total weight 14
- **Weighted majority voting**: Aggregate 9 extractions per field using model weights
- **Term version tracking**: SHA-256 hash of L0 schema stored in cache metadata; warns on mismatch
- **Dual cache**: Raw (per model/run) + Aggregated (final voted result) in `policy_cache.json`
- **Default K=200, mode=24hrbatch**: Changed from K=50, ondemand
- **`--all` flag**: Defaults to extracting all 18 topics (G1-G6 × 3 topics each)
- **`--models` flag**: Allows custom config like `gpt-5-nano:1` for testing

## Current State

- All code implemented and tested
- Small test passed:
  - ondemand: G1_allergy, 2 restaurants, gpt-5-nano:1 → 50 reviews extracted
  - 24hrbatch: G1_dietary, 1 restaurant, gpt-5-nano:1 → batch submitted, completed, processed, cron cleaned
- compute_gt tested: Generated GT files for V0-V3 policies

**Ready for full extraction** but user ended session before running.

## Next Steps

1. Run full extraction (when ready):
   ```bash
   # Full multi-model (3.24M requests)
   .venv/bin/python -m addm.tasks.cli.extract

   # Or single model for testing (360K requests)
   .venv/bin/python -m addm.tasks.cli.extract --models "gpt-5-nano:1"
   ```

2. After batch completes, compute GT:
   ```bash
   .venv/bin/python -m addm.tasks.cli.compute_gt
   ```

3. Verify GT distributions across policies

## Open Questions

- Which model config to use for production? Full multi-model (expensive but robust) vs gpt-5-nano only (cheaper but less reliable)?
- Cost estimate needed for 3.24M vs 360K requests

## Key Files

### Created
- `src/addm/tasks/policy_gt.py` - Core GT computation, aggregation, term loading
- `docs/specs/ground_truth.md` - Documentation

### Modified
- `src/addm/tasks/extraction.py` - Added `PolicyJudgmentCache`, `REQUIRED_RUNS`
- `src/addm/tasks/cli/extract.py` - Added `--topic`, `--policy`, `--all`, `--models` flags; multi-model batch support
- `src/addm/tasks/cli/compute_gt.py` - Added `--policy`, `--all` flags; policy-based scoring

## Context & Background

- 18 topics total: G1 (allergy, dietary, hygiene), G2 (romance, business, group), G3 (price_worth, hidden_costs, time_value), G4 (server, kitchen, environment), G5 (capacity, execution, consistency), G6 (uniqueness, comparison, loyalty)
- 72 policies total: 18 topics × 4 variants (V0, V1, V2, V3)
- V0/V1: Qualitative decision rules
- V2/V3: Point-based scoring system with thresholds
- Cache location: `data/tasks/yelp/policy_cache.json`
- GT output: `data/tasks/yelp/{policy_id}_K{k}_groundtruth.json`
