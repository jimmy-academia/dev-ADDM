# [2026-01-24] Fix EVIDENCE_FIELDS Mismatch ✅

**What**: Fixed eval metrics showing 0% evidence precision - EVIDENCE_FIELDS in constants.py was mismatched with Term Library definitions.

**Impact**: Evaluation metrics (evidence precision/recall, judgment accuracy) now work correctly. Created automation script to prevent future drift.

**Files**:
- `src/addm/eval/constants.py` - Fixed EVIDENCE_FIELDS to match term libraries
- `scripts/generate_evidence_fields.py` - NEW: Auto-generates EVIDENCE_FIELDS from term libraries
- `data/answers/yelp/*_K50_groundtruth.json` - Regenerated with correct incidents
- `src/addm/utils/results_manager.py` - Dev runs now add suffix on collision (_1, _2)
- `src/addm/methods/amos/phase1_prompts.py` - Fixed default verdict rule (rules 5 & 6)
- `src/addm/utils/debug_logger.py` - Per-restaurant debug logging

**Root Cause Found**:
```
Term Library → defines field + values + default
constants.py → had wrong field names / wrong values
compute_gt   → couldn't find matching values → incidents: []
Evaluation   → compared AMOS vs empty GT → 0% precision
```

**Next**:
- Run `python scripts/generate_evidence_fields.py --check` before releases
- Consider adding this check to CI
- Regenerate GT for other K values (25, 100, 200) if needed

**Status**: Uncommitted: 87 files (mostly regenerated GT files)
