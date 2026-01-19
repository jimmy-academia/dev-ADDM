# Session Log: Paper Introduction First Draft

**Date**: 2026-01-18 23:23
**Status**: completed

## Summary

Completed the first draft of the paper introduction section, transforming it from 70% complete (with placeholders) to a polished draft ready for downstream sections. Applied critical corrections based on user feedback to fix terminology, method descriptions, and positioning. Added concrete examples, expanded previews, and defined key claims for Discussion section.

## Decisions Made

- **Emphasize versatility**: Frame ADDM as general-purpose framework across multiple domains (allergy safety, service quality, competitive positioning), not just domain-specific
- **Two-phase framework description**: AMOS = Query-Focused Compilation (Phase 1) + Context Operation (Phase 2)
- **Observable steps**: Central contribution is explicit intermediate values for auditability and ablation
- **Key claims defined**: 5 testable claims for Discussion section to validate
- **Positioning strategy**: Contrast with semantic operator work (UNIFY, PALIMPZEST) by emphasizing dynamic agenda adaptation + observable steps

## Critical Corrections Applied

Based on user feedback, corrected several misconceptions:

1. **Removed legacy terminology**: Eliminated L0/L1/L2 language (old system). New structure is V0/V1 (Overview→Definitions→Verdict Rules) and V2/V3 (Overview→Definitions→**Scoring System**→Verdict Rules)

2. **Corrected method description**: AMOS is NOT the multi-model ensemble (that's GT generation). AMOS is:
   - Phase 1: Query-Focused Compilation (agenda → Formula Seed with filter/extract/compute)
   - Phase 2: Context Operation (interpret Formula Seed against reviews)

3. **Emphasized versatility**: Not just "allergy safety inspector" - show multiple personas, domains, and aggregation logic types

4. **Clarified framework contribution**: Highlight is dynamic adaptation to agenda specifications using LLMs to produce "key ingredients" from documents

5. **Moved details to appendices**: Kept introduction high-level - layered greedy selection, 72 tasks breakdown, multi-model ensemble, L0/L1/L2 implementation all deferred to appendices

## Changes Made to `paper/sections/1_introduction.tex`

### 1. Added Concrete Motivating Example (line 12)
```latex
Decision-makers across domains---from allergy safety assessors evaluating
restaurant risk, to business owners diagnosing service quality issues, to
investors analyzing competitive positioning---face a common challenge: they
must extract structured evidence from customer reviews, apply domain-specific
policies, and justify their verdicts with explicit criteria...
```

**Purpose**: Make problem tangible, emphasize versatility across domains

### 2. Expanded Method Preview (lines 28-33)
Three sub-sections:
- **Framework Overview**: Two-phase design (compile agenda → execute on data)
- **Observable Steps**: Explicit intermediate values at each stage
- **Evaluation Protocol**: Multi-dimensional metrics (verdict correctness, extraction quality, efficiency)

**Key insight**: Separates agenda understanding (compile-time) from evidence processing (runtime), enabling caching and inspection

### 3. Expanded Experiment Preview (line 35)
- 100 restaurants curated to stress rare-signal prioritization, conflicting evidence, temporal drift
- 18 decision topics × 4 policy variants (V0-V3 testing compositional reasoning)
- Baselines: full-context prompting, RAG, code-execution search
- Multi-dimensional metrics

### 4. Strengthened Positioning (line 7)
Revised gap statement to explicitly contrast with semantic operator work:
```latex
Yet a missing piece remains in practice: a framework that (i)~\emph{dynamically
adapts} to diverse decision agendas, (ii)~executes at the level of real-world
entities (one item with many reviews), and (iii)~produces \emph{auditable
decision artifacts} with observable intermediate steps...
```

### 5. Defined Key Claims for Discussion (line 46)
5 testable claims:
1. AMOS adapts its processing logic to diverse agenda specifications
2. Observable intermediate steps improve auditability and enable ablation analysis
3. Policy complexity (V0-V3 progression) reveals compositional reasoning limits in LLMs
4. Rare-but-critical signals require explicit prioritization mechanisms beyond frequency-based retrieval
5. Two-phase design reduces redundant agenda interpretation costs across samples

### 6. Polish Pass
- Removed extra blank lines for consistency
- Verified smooth paragraph transitions
- Ensured consistent terminology (ADDM, AMOS, agenda, Formula Seed)

## Current State

Introduction section is **complete** (first draft):
- ✅ No placeholders remain ("... method...", "...experiment..." removed)
- ✅ All planned content added
- ✅ Stays within ~2 page budget (~45 lines)
- ✅ Smooth paragraph flow maintained
- ✅ Ready to support downstream sections (Related Work, Discussion, Experiments)

## Next Steps

1. **B2: Related Work** (next in dependency chain from roadmap)
   - Use Introduction's gap analysis for positioning
   - Cover 6 groups: ABSA/Opinion Mining, Text Summarization, RAG & Retrieval, Semantic Operators, LLM Reasoning, Multi-Agent Systems
   - Keep main paper concise (~0.5 page), expand in Appendix B

2. **Future polish** (after experiments complete):
   - Update contributions list to match final results (currently marked pending in roadmap)
   - Ensure claims in Discussion section address all 5 defined claims

3. **Validation**:
   - After Method section (B4) is written, verify method preview is consistent
   - After Experiments (B5) is written, verify benchmark preview is accurate

## Key Files Modified

- `paper/sections/1_introduction.tex` - Main introduction content (6 edits)
- `docs/ROADMAP.md` - Updated B1 status to "✅ COMPLETE (First Draft)"

## Key Files Referenced

- `paper/plan/1p_introduction.md` - Paragraph-level plan (read-only)
- `docs/architecture.md` - System overview for technical details
- `docs/specs/ground_truth.md` - GT generation approach
- `docs/tasks/TAXONOMY.md` - 72 tasks structure
- `src/addm/query/policies/G1/allergy/V2.yaml` - Policy example

## Background Processes

⚠️ Two Python processes are running:
- AMOS baseline run (gpt-5-nano, n=10, quiet mode)
- Direct baseline run (gpt-5, K=25, skip=64, n=1)

These appear to be experiment runs and can be left running or killed as needed.

## Git Status

Uncommitted changes:
- `M data/formula_seeds/G1_allergy_V2.json` - Formula seed updates
- `M docs/ROADMAP.md` - Updated B1 status
- `m paper` - LaTeX changes (submodule marker)
- `M src/addm/tasks/cli/run_baseline.py` - CLI updates

User should commit when ready.

## Roadmap Context

- **Phase**: Phase I - G1_allergy Pipeline Validation
- **Timeline**: 14 days to goal (Feb 1), 21 days to hard deadline (Feb 8)
- **Today's Focus** (Jan 18):
  - ✅ A1: Aggregate G1_allergy raw judgments → consensus L0 (COMPLETE)
  - ✅ B1: Polish Introduction, define key claims for Discussion (COMPLETE)

**Next priority**: B2 (Related Work) follows in dependency chain

## Notes

- Introduction successfully balances high-level clarity with technical precision
- All placeholders removed - section is ready for review
- Key claims provide clear roadmap for Discussion section to validate
- Emphasis on versatility (multiple domains) positions ADDM as general framework
- Two-phase design and observable steps are central to framework contribution
