# Session Log: AMOS Method Implementation

**Date**: 2026-01-18 22:20
**Status**: completed

## Summary

Implemented the AMOS (Agenda-Driven Mining with Observable Steps) method - a two-phase approach where Phase 1 "compiles" the agenda into an Agenda Spec (executable JSON spec), and Phase 2 interprets it against restaurant data. The method is now functional and tested with 100% accuracy on test samples.

## Decisions Made

- **Architecture**: Two-phase design with LLM-generated Agenda Spec cached per policy
- **Caching**: Agenda Specs cached at `data/formula_seeds/{policy_id}.json` with hash-based invalidation
- **Per-extraction scoring**: Case operations on extraction fields (like INCIDENT_SEVERITY) are applied per-extraction and summed
- **JSON robustness**: Added multiple fallback strategies for parsing LLM-generated JSON (trailing commas, comments, balanced brace detection)

## Current State

AMOS method is fully implemented and working:
- Phase 1 generates Agenda Spec with filter keywords, extraction fields, and compute operations
- Phase 2 filters reviews, extracts signals via parallel LLM calls, and computes final verdict
- Tested successfully: 5/5 accuracy on G1_allergy_V2 K=25

## Next Steps

1. Run AMOS on larger sample set (n=20+) to get meaningful AUPRC metrics
2. Compare AMOS token cost vs direct baseline
3. Consider regenerating Agenda Spec with better model if extraction quality is insufficient
4. Test on other policies (V0, V1, V3) to verify generalization

## Open Questions

- Should Agenda Spec generation use a stronger model (gpt-5-mini) for better schema quality?
- How to handle policies where the LLM-generated schema doesn't match the actual policy intent?

## Key Files

### Created
- `src/addm/methods/amos/__init__.py` - Module exports
- `src/addm/methods/amos/phase1.py` - Agenda Spec generation (~170 lines)
- `src/addm/methods/amos/phase2.py` - Interpreter (~520 lines)
- `src/addm/methods/amos_method.py` - Main AMOSMethod class (~100 lines)
- `data/formula_seeds/G1_allergy_V2.json` - Cached Agenda Spec

### Modified
- `src/addm/methods/__init__.py` - Registered AMOSMethod
- `src/addm/tasks/cli/run_baseline.py` - Added `--method amos` and `--regenerate-seed` flags

## Context & Background

### CLI Usage
```bash
# Basic AMOS run
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 5 --method amos --dev

# Force regenerate Agenda Spec
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 5 --method amos --dev --regenerate-seed
```

### Agenda Spec Structure
The generated Agenda Spec for G1_allergy_V2 includes:
- **Filter**: 44 allergy-related keywords
- **Extract**: 7 fields (INCIDENT_SEVERITY, ACCOUNT_TYPE, ASSURANCE_OF_SAFETY, STAFF_RESPONSE, CUISINE_RISK, ALLERGEN_MENTION, FIRSTHAND_INCIDENT_PRESENT)
- **Compute**: 6 operations (counts, case-based scoring, final verdict)
- **Output**: VERDICT, SCORE, N_INCIDENTS

### Observable Output
Each result includes intermediate values for debugging:
```json
{
  "VERDICT": "Low Risk",
  "SCORE": 0,
  "_extractions": [...],
  "_namespace": {"N_INCIDENTS": 0, "INCIDENT_POINTS": 0, ...},
  "_filter_stats": {"total_reviews": 25, "filtered_reviews": 1, ...}
}
```
