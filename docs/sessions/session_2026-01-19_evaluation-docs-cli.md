# [2026-01-19] Evaluation Documentation & CLI Tool ✅

**What**: Created comprehensive evaluation docs and Rich-based CLI tool for displaying formatted result summaries.

**Files**:
- `docs/specs/evaluation.md` - 3-score system (AUPRC, Process, Consistency)
- `src/addm/eval/cli/show_results.py` - Result display CLI with comparison tables

**Features**:
- Multiple modes: single result, comparison table, benchmark vs dev
- Auto-fallback to latest dev results
- Filters: `--method`, `--policy`, `--latest N`

**Next**: Run AMOS evaluation, use CLI to verify >75% accuracy on all 3 scores

**Status**: Complete, ready for use
