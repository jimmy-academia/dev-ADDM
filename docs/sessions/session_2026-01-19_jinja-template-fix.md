# [2026-01-19] Jinja Template Condition Rendering Fix ✅

**What**: Fixed bug in `verdict_rules.jinja2` where structured conditions rendered as raw dicts instead of natural language text.

**Impact**: 32 V0/V1 prompt files were showing `{'text': '...'}` instead of the actual condition text. All regenerated.

**Files**:
- `src/addm/query/generators/templates/verdict_rules.jinja2` - Added `{% if condition is mapping %}` check
- `data/query/yelp/*_V0_prompt.txt` - 17 files regenerated
- `data/query/yelp/*_V1_prompt.txt` - 15 files regenerated

**Next**:
- Commit the template fix and regenerated prompts
- Note: CLI has separate bug where `--output` creates nested directories

**Status**: Uncommitted: 33 files (1 template + 32 prompts)
