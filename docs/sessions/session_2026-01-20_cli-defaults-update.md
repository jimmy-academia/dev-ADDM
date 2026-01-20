# [2026-01-20] CLI Defaults Update ✅

**What**: Updated CLI commands to have better defaults - `generate.py` now generates all 72 policies by default, `run_baseline.py` now uses benchmark mode by default.

**Files**:
- `src/addm/query/cli/generate.py` - Removed `--all` flag, default is all-generate
- `src/addm/tasks/cli/run_baseline.py` - Removed `--benchmark` flag, benchmark is default
- `.claude/rules/data-pipeline.md` - Fixed query command syntax
- `.claude/rules/cli-commands.md` - Updated flags table, added quota docs

**Next**:
- Test generate.py default behavior on fresh run
- Verify benchmark quota works as expected with new default

**Status**: Clean working tree (changes already committed)

**Context**: Part of documentation audit fixes. Previously required explicit `--all` or `--benchmark` flags; now these are the defaults with `--dev` as the opt-out for development work.
