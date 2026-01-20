# [2026-01-20] Unified Structured Output for All Methods ✅

**What**: Implemented unified structured output format (`output_schema.txt`) across all methods (direct, rag, amos, rlm).

**Impact**: All methods now output consistent format with `verdict`, `evidences`, `justification`, enabling unified evaluation metrics.

**Files**:
- `src/addm/methods/amos/phase2.py` - Added `_build_standard_output()`, `get_standard_output()` transformation
- `src/addm/methods/amos.py` - Added `system_prompt` param, returns `parsed` field
- `src/addm/methods/direct.py` - Added `system_prompt` param to constructor
- `src/addm/methods/rlm.py` - Updated `RLM_QUERY_TEMPLATE` with full schema
- `src/addm/tasks/cli/run_experiment.py` - Passes `system_prompt` to AMOS

**Next**:
- RLM needs work - code execution makes exact JSON format hard to guarantee
- Test with more samples to verify evaluation metrics work correctly

**Status**: Uncommitted: 7 files (includes unrelated renames)

**Context**: Plan was to ensure all methods produce same output for fair comparison. AMOS transforms internal `_extractions` → standard `evidences` format.
