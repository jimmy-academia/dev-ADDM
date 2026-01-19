# Session Log: ROADMAP Audit and Update

**Date**: 2026-01-18
**Status**: completed

## Summary

Performed documentation audit at user request to verify A1 (GT Completion) status. Confirmed A1 is complete (as of Jan 18, 20:59) but ROADMAP.md was outdated. Updated ROADMAP to mark A1 complete and reprioritized timeline to implement AMOS (A4) before experiment infrastructure (A2).

## Decisions Made

- **A1 is COMPLETE**: All Phase I tasks done (aggregation + GT computation for V0-V3)
- **AMOS prioritized**: Implement draft AMOS (Sun-Mon) before designing experiment infrastructure (Tue)
- **New timeline order**: A1 → A4 draft → A2 → A3 → A4 tuning → A5
- **Rationale**: Building AMOS first will inform what experiment infrastructure needs to support

## Current State

### Verified A1 Completion Evidence
- ✅ Raw judgments: 73,760 entries (18,440 reviews × 4 model runs)
- ✅ Aggregated to consensus L0: 18,440 reviews
- ✅ GT computed: 4 files (V0-V3), 100 restaurants each
- ✅ Committed: Jan 18, 20:59 (commits `3350bbc`, `4de7deb`)
- ✅ Cache file exists: `judgement_cache.json` (33MB)

### ROADMAP Updates Made
- Marked A1 Phase I as "✅ COMPLETE" with checked boxes
- Moved "G1_allergy extraction and aggregation" to "Completed (Phase I)" section
- Updated Quick Reference: A1 status → "✅ Complete" (Jan 18)
- Reordered Week 1 schedule: AMOS (Sun-Mon) before A2 (Tue)
- Updated critical path diagram to show new order
- Updated A4 status to "In Progress" (from "Not Started")
- Committed changes: `abef521` - "docs: update ROADMAP - A1 complete, AMOS prioritized before A2"

### Uncommitted Work
User has uncommitted changes (see git status):
- Modified: GT files, method registry, CLI scripts
- New files: GT variants for K=25/50/100, RAG baseline draft, dev results
- Note: `policy_cache.json` appears as untracked (should be in .gitignore as `judgement_cache.json`)

## Next Steps

1. **Tomorrow (Sun Jan 19)**: Start A4 AMOS draft implementation
2. **Paper work**: Begin B2 Related Work draft (informs baseline selection)
3. **Optional**: Spot-check GT quality (5-10 samples from V0-V3 files)
4. **Cleanup**: Review uncommitted changes, commit or discard as appropriate

## Key Files Modified This Session

- `docs/ROADMAP.md` - Updated A1 status, reordered timeline, marked AMOS in progress
- `docs/sessions/session_2026-01-18_*.md` - Read previous session logs for context

## Key Files Referenced

- `data/tasks/yelp/judgement_cache.json` - L0 judgment cache (33MB)
- `data/tasks/yelp/G1_allergy_V{0-3}_K200_groundtruth.json` - GT outputs
- `src/addm/tasks/extraction.py` - PolicyJudgmentCache implementation
- `src/addm/tasks/cli/extract.py` - Multi-model extraction CLI
- `src/addm/tasks/cli/compute_gt.py` - GT computation CLI

## Context & Background

### Audit Process
1. Checked GT file existence and structure
2. Verified judgement cache contents (73,760 raw + 18,440 aggregated)
3. Confirmed git history (commits on Jan 18)
4. Verified CLI flags (`--topic` exists in both extract.py and compute_gt.py)
5. Identified 3 ROADMAP discrepancies (outdated status, unchecked boxes, vague state)

### Timeline Philosophy Change
Original plan: A1 → A2 → A3 → A4 → A5
New plan: A1 → **A4 draft** → A2 → A3 → A4 tuning → A5

Implementing AMOS early will reveal:
- What intermediate results need tracking (for ablations)
- Usage/latency/cost patterns to capture
- Multi-run variance requirements
- Phase 1/Phase 2 separation needs

This informs A2 (experiment infrastructure design) better than designing infrastructure in isolation.

### Project Context
- Phase I: Validate pipeline on G1_allergy before Phase II scaling
- Timeline: 14 days to goal (Feb 1), 21 days to hard deadline (Feb 8)
- Critical checkpoint: AMOS must beat baselines on G1_allergy by Sat 25
- Related Work (B2) should happen early (Sun 19) to inform baseline selection

## Related Sessions

- `session_2026-01-18_cache-rename.md` - Cache file renamed, G1_allergy confirmed complete
- `session_2026-01-18_extraction-bugfixes.md` - Fixed extraction bugs, computed GT
- `session_2026-01-18_ondemand-mode-fix.md` - Improved extraction logging
- `session_2026-01-18_hello-roadmap-integration.md` - Integrated roadmap into /hello
- `session_2026-01-18_batch-pipeline-refactor.md` - Batch pipeline work
