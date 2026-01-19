# Session Log: Extraction Pipeline Bugfixes

**Date**: 2026-01-18
**Status**: completed

## Summary

Fixed critical bugs in the ondemand extraction mode and improved CLI usability. The main bug was that `needs_extraction()` assumed runs were sequential (1,2,3) but some reviews had run2,run3 missing run1, causing infinite re-extraction of already-existing runs.

## Decisions Made

- **Fixed `needs_extraction()` logic** - Now checks for SPECIFIC run numbers, not just count
- **Added `--topic` flag to compute_gt.py** - Computes all V0-V3 variants for a topic
- **Improved ondemand mode output** - Shows what's being extracted and when cache is saved
- **Removed slow pre-scan** - Was redundant; cache status already shown before ondemand starts

## Current State

All bugs fixed. User can now run:

```bash
# Extract (will find missing run1 for 2 reviews)
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --mode ondemand

# Compute GT for all V0-V3 variants
.venv/bin/python -m addm.tasks.cli.compute_gt --topic G1_allergy
```

GT has been computed - files exist:
- `data/tasks/yelp/G1_allergy_V0_K200_groundtruth.json`
- `data/tasks/yelp/G1_allergy_V1_K200_groundtruth.json`
- `data/tasks/yelp/G1_allergy_V2_K200_groundtruth.json`
- `data/tasks/yelp/G1_allergy_V3_K200_groundtruth.json`

## Bugs Fixed

### 1. `needs_extraction()` assumed sequential runs
**Problem**: If a review had run2 and run3 but was missing run1, the old code:
- Counted 2 runs
- Assumed runs 1,2 existed
- Requested run 3 (which already existed)
- Kept overwriting run3, never adding run1

**Fix**: Added `get_cached_runs()` that returns the SET of run numbers, then check for specific missing runs.

**Files**: `src/addm/tasks/extraction.py` (lines 156-174, 256-265)

### 2. Slow pre-scan in ondemand mode
**Problem**: Added a pre-scan to count needed extractions that was redundant (cache status already shown) and took >1 minute for 18k reviews.

**Fix**: Removed the scan, use the already-computed cache status to check if all complete.

**File**: `src/addm/tasks/cli/extract.py` (lines 716-730)

### 3. Misleading ondemand mode message
**Problem**: Said "for testing" which implied it wasn't production-ready.

**Fix**: Changed to "immediate, no batch discount" which accurately describes the mode.

## Next Steps

1. Run extraction to complete the 2 missing run1 entries:
   ```bash
   .venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --mode ondemand
   ```

2. Continue with Phase I: implement baselines, then AMOS development

## Key Files Modified

- `src/addm/tasks/extraction.py` - Added `get_cached_runs()`, fixed `needs_extraction()`
- `src/addm/tasks/cli/extract.py` - Improved ondemand mode, better logging
- `src/addm/tasks/cli/compute_gt.py` - Added `--topic` flag

## Context & Background

The extraction pipeline uses a multi-model ensemble (gpt-5-mini:1, gpt-5-nano:3) for GT generation. Two reviews were stuck with run2,run3 but missing run1, causing the completion check to repeatedly overwrite run3 instead of adding run1.

Cache status before fix:
- gpt-5-mini: 18440/18440 complete
- gpt-5-nano: 55318/55320 (need 2) - these 2 were the missing run1 entries
