# Session Log: Intermediate Process Evaluation Implementation

**Date**: 2026-01-18 22:45
**Status**: completed

## Summary

Implemented a multi-stage intermediate process evaluation framework for ADDM that evaluates the quality of reasoning steps, not just final verdicts. The framework assesses evidence validity, L0 classification accuracy, and verdict support validity.

## Decisions Made

- Use "sufficiency mode" - methods can early-stop once threshold is met, so we evaluate whether claimed evidence justifies the verdict (not requiring ALL incidents to be found)
- Added `scoring_trace` field to output schema for transparent score tracking
- RAG method now uses structured output schema (same as direct method) when system_prompt provided
- Snippet validation uses substring matching with whitespace normalization fallback

## Implementation Completed

### Phase 1: RAG Method Fix (`src/addm/methods/rag.py`)
- Added `system_prompt` parameter to `RAGMethod.__init__`
- Added `_build_messages()` helper method
- Updated all 3 LLM call sites to use structured output when available

### Phase 2: Intermediate Metrics Module (`src/addm/eval/intermediate_metrics.py`)
- `compute_evidence_validity()` - Stage 1: incident precision + snippet validation
- `compute_classification_accuracy()` - Stage 2: severity/modifier accuracy on matched evidence
- `compute_verdict_support()` - Stage 3: whether claimed evidence supports verdict
- `compute_intermediate_metrics()` - Main entry combining all stages
- Helper functions: `load_gt_with_incidents()`, `build_reviews_data()`

### Phase 3: Pipeline Integration (`src/addm/tasks/cli/run_baseline.py`)
- Pass `system_prompt` to RAG method instantiation
- Compute intermediate metrics after AUPRC when structured output available
- Display summary table and add `intermediate_metrics` to output JSON

### Phase 4: Output Schema Update (`src/addm/query/prompts/output_schema.txt`)
- Added `scoring_trace` field with `total_score` and `breakdown` array
- Emphasized verbatim snippet requirement (must be substring of review)
- Added guideline for including scoring_trace with point-based scoring

## Current State

Implementation complete and tested:
- All imports work correctly
- Unit test of intermediate metrics produces expected results:
  - incident_precision, snippet_validity, severity_accuracy, verdict_support_rate
- Ready for real-world testing with `--policy G1_allergy_V2 -n 5 --dev`

## Next Steps

1. Run baseline with structured output to verify real metrics:
   `.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 5 --dev`
2. Verify `parsed.evidences` and `parsed.justification.scoring_trace` populated in results
3. Inspect metric breakdown in output JSON
4. Consider adding visualization/reporting for intermediate metrics

## Key Files

- `src/addm/eval/intermediate_metrics.py` - **NEW**: Multi-stage metrics module (17KB)
- `src/addm/eval/__init__.py` - Updated exports
- `src/addm/methods/rag.py` - Added system_prompt support
- `src/addm/tasks/cli/run_baseline.py` - Pipeline integration
- `src/addm/query/prompts/output_schema.txt` - Added scoring_trace

## Metrics Computed

| Metric | Description |
|--------|-------------|
| `incident_precision` | claimed ∩ GT / claimed |
| `snippet_validity` | valid snippets / total snippets |
| `severity_accuracy` | correct severity on matched evidence |
| `modifier_accuracy` | correct modifiers on matched evidence |
| `verdict_support_rate` | methods where evidence justifies verdict |
| `score_consistency` | claimed score matches computed score |

## Context & Background

- The plan was provided in the user's request with detailed specs
- GT structure already had `incidents` array with review_id, severity, modifiers
- Output schema already had evidences/justification structure, just needed scoring_trace
- Uses "sufficiency mode" since methods can early-stop once threshold is met
