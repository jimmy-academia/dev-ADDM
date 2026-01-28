# Session Log: Fix Introduction Execution Model

**Date**: 2026-01-18
**Status**: completed

## Summary

Fixed the paper introduction to accurately describe AMOS's 2-phase execution model instead of the outdated 5-stage model. Removed references to "semantic grouping" (not implemented) and added missing description of multi-model ensemble ground truth generation.

## Decisions Made

- **Remove "semantic grouping" entirely** - this stage was described but never implemented
- **Use Phase 1/Phase 2 terminology consistently** - matches actual AMOS implementation
- **Add multi-model ground truth description** - was missing from introduction
- **Update introduction plan (1p_introduction.md)** - paragraphs 6-9 now reflect 2-phase framework

## Current State

**Completed**:
- Updated `paper/sections/1_introduction.tex`:
  - Lines 16-19: Replaced 5-stage model with 2-phase framework description
  - Line 20: Updated allergy example to use Phase 1/2 terminology
  - Line 31: Added multi-model ensemble ground truth description
- Updated `paper/plan/1p_introduction.md`:
  - Paragraphs 6-9: Now describe Phase 1 (compile-time), Phase 2 (runtime), and multi-model consensus

**Verification**:
- ✅ No remaining "L0/L1/L2/FL" references in introduction
- ✅ No remaining "semantic grouping" references
- ✅ Consistent Phase 1/Phase 2 terminology throughout
- ✅ Introduction now accurately reflects actual AMOS implementation

## Next Steps

1. **Review updated introduction** - read through to ensure flow is natural
2. **Alignment check** - verify introduction is consistent with:
   - Problem formulation (Section 3) - should map Phase 1/2 to L0/L1/L2/FL layers
   - Method section (when written) - should explain Phase 1/2 implementation details
3. **Continue paper work** - other sections may need similar accuracy fixes

## Key Changes

### Old (Lines 16-22):
```tex
We instantiate ADDM in a fixed multi-stage data aggregation execution model:
(i) per-review semantic extraction (map),
(ii) review-level logic composition,
(iii) optional semantic grouping over multi-review evidence,  ← NOT IMPLEMENTED
(iv) item-level conditional aggregation (reduce),
and (v) final policy evaluation.
```

### New (Lines 16-19):
```tex
We instantiate ADDM through a two-phase execution framework:
Phase~1 (compile-time) interprets the agenda specification once per decision type,
producing an Agenda Spec---a structured program containing filter keywords,
extraction field schemas, and deterministic aggregation rules.
Phase~2 (runtime) executes the Agenda Spec for each entity: filtering relevant
reviews, extracting primitives via parallel LLM calls, computing aggregates
deterministically, and applying the final decision policy.
```

### Added (Line 31):
```tex
Ground truth generation follows a two-step protocol:
(i)~multi-model ensemble extraction, where each review is processed by
multiple LLMs with different capability/cost tradeoffs and aggregated
via weighted majority voting to produce consensus primitives, and
(ii)~deterministic policy execution that computes aggregates and verdicts
from consensus primitives following the same rules as the evaluated methods.
```

## Key Files

- `paper/sections/1_introduction.tex` - Main introduction section (updated)
- `paper/plan/1p_introduction.md` - Introduction writing plan (updated paragraphs 6-9)
- `src/addm/methods/amos/phase1.py` - Actual Phase 1 implementation (reference)
- `src/addm/methods/amos/phase2.py` - Actual Phase 2 implementation (reference)
- `src/addm/tasks/policy_gt.py` - Multi-model ensemble GT generation (reference)

## Context & Background

The introduction previously described a "5-stage execution model" that didn't match the actual AMOS implementation:
- Claimed stage (iii): "optional semantic grouping over multi-review evidence" - **never implemented**
- Problem formulation (Section 3) uses L0/L1/L2/FL layered model (conceptually correct)
- Actual AMOS uses 2-phase approach: Phase 1 (agenda → Agenda Spec), Phase 2 (Agenda Spec → verdict)
- Ground truth uses multi-model ensemble (gpt-5-nano×3, gpt-5-mini×1) with weighted voting - **not mentioned in old version**

The plan from exit plan mode provided detailed analysis and specific text replacements which were all successfully implemented.
