# [2026-01-22] Manifest-Based Batch Support ✅

**What**: Implemented manifest-based batch tracking for run_experiment.py - re-run same command to check status instead of needing --batch-id.

**Impact**: Simplified batch workflow. Manifest auto-saves after submission, auto-deletes after processing. Token usage now extracted from batch responses.

**Files**:
- `src/addm/tasks/cli/run_experiment.py` - Added manifest functions, batch usage extraction
- `src/addm/methods/cot.py` - New CoT baseline method
- `src/addm/methods/react.py` - New ReACT baseline method
- `src/addm/methods/__init__.py` - Registered CoT/ReACT in method registry

**Next**:
- Test manifest workflow end-to-end with actual batch job
- Verify token tracking appears correctly in results.json

**Status**: Committed and pushed (88357c9)

**Context**: Manifest stored at `results/{method}/{policy}_K{k}/batch_manifest.json`. Contains batch_id, method, policy_id, k, n, model, created_at.
