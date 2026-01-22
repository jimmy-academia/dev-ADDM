# [2026-01-22] GT Pipeline Recovery & Extract Fixes 🔄

**What**: Fixed corrupted judgment cache, recovered batch data from OpenAI, added progress/ETA to extract, optimized aggregation speed.

**Impact**: Cache repaired (1.32M raw entries), recovery manifest created from 32 OpenAI batches, aggregation now O(n) instead of O(n²).

**Files**:
- `src/addm/tasks/cli/extract.py` - Added `--aggregate` flag, progress/ETA, O(n) index for aggregation, removed auto-delete of manifests
- `src/addm/tasks/extraction.py` - Added `get_raw_review_ids()` method
- `scripts/run_gt_pipeline.sh` - Added `--aggregate` option for recovery mode
- `data/answers/yelp/batch_manifest_all_topics_recovery.json` - Recovery manifest with 32 batch IDs

**Next**:
- Run `bash scripts/run_gt_pipeline.sh` to complete GT generation
- After GT verified, manually delete manifest: `rm data/answers/yelp/batch_manifest_all_topics_recovery.json`
- Consider adding progress bar (rich) instead of text updates

**Status**: Uncommitted: 8 files | Running: extract.py (in progress)

**Context**: Cache got corrupted during interrupted aggregation. Recovered by: (1) repairing truncated JSON, (2) fetching batch IDs from OpenAI API, (3) creating recovery manifest.
