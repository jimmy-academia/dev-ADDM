# [2026-01-20] AMOS Adaptive Mode Fix ✅

**What**: Fixed adaptive mode verdict issue where `_compute_case()` couldn't evaluate full expressions like `"SCORE >= 8"` from Formula Seed.

**Impact**: Both Parallel and Adaptive modes now work with 100% accuracy on K=50 (n=20).

**Files**:
- `src/addm/methods/amos/phase2.py` - Fixed `_compute_case()` to use `SafeExpressionExecutor` for full expression evaluation
- `src/addm/methods/amos/phase1.py` - Has iterative fix approach (3 retries) for malformed expressions
- `src/addm/tasks/cli/run_baseline.py` - Captures strategy metrics in results

**Next**:
- Test K=200 to verify early stopping saves tokens with more reviews
- Compare token costs between Parallel vs Adaptive modes
- Consider updating early_verdict_expr in formula seed prompt to include Low Risk case

**Status**: Working tree clean

**Context**: AMOS Enhancement plan implemented with LLM-driven search strategy (priority_expr, stopping_condition, early_verdict_expr). Two modes: Parallel (default, fast) vs Adaptive (batch processing with early stopping).
