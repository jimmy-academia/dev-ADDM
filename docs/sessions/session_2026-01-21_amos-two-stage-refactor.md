# [2026-01-21] AMOS Two-Stage Retrieval Refactor ✅

**What**: Implemented and refactored AMOS two-stage retrieval. Changed from opt-in `--thorough` to default behavior with `--filter-mode` for Stage 1 selection.

**Impact**: AMOS now processes ALL reviews by default (thorough sweep always on). Stage 1 filter mode controls early-exit behavior.

**Files**:
- `src/addm/methods/amos/config.py` - Added `FilterMode` enum, removed adaptive/hybrid/thorough_sweep
- `src/addm/methods/amos/phase2.py` - New `_filter_by_mode()`, `execute()` with two-stage flow
- `src/addm/methods/amos/__init__.py` - Simplified `AMOSMethod.__init__` with `filter_mode` param
- `src/addm/methods/amos/search/embeddings.py` - Added `retrieve_by_embedding()` method
- `src/addm/tasks/cli/run_experiment.py` - Replaced old flags with `--filter-mode`
- `.claude/rules/methods.md` - Updated AMOS documentation

**Next**:
- Improve AMOS to achieve 90+ scores (AUPRC, Process, Consistency)
- Test `--filter-mode embedding` and `--filter-mode hybrid`
- Investigate extraction quality vs ground truth alignment

**Status**: Clean (all changes uncommitted but complete)

**Context**: Current scores are low (AUPRC: 0%, Process: 7.9%) despite processing all reviews. Issue is likely extraction/aggregation quality, not retrieval coverage.
