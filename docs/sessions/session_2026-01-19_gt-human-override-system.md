# [2026-01-19] GT Human Override System ✅

**What**: Implemented human override system for correcting LLM judgment errors. Overrides apply at aggregated judgment level (before scoring), one override covers all policy variants (V0-V3) and K values.

**Files**:
- `data/answers/yelp/judgment_overrides.json` - NEW: Override storage (2 G1_allergy corrections)
- `src/addm/tasks/policy_gt.py` - Override functions + cuisine modifier bug fix
- `src/addm/tasks/cli/compute_gt.py` - CLI integration

**Fixes**:
- Cuisine modifier bug: Only apply +2 points when incidents exist
- 2 G1_allergy corrections: Chop Steakhouse (severe→none), South Coast Deli (mild→none)

**Next**: Run AMOS evaluation on G1_allergy_V2 (K=50, N=100) → verify >75% accuracy on all 3 scores

**Status**: Complete, all 16 G1_allergy GT files ready
