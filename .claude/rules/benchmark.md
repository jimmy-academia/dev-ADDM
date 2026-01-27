# Benchmark: T* System (35 Policies)

## Structure

**5 Tiers × 7 Variants = 35 Policies**

| Tier | Topic | Source | Status |
|------|-------|--------|--------|
| T1 | Allergy Safety | F1: 0.64 | ✅ Ready (all P1-P7 complete) |
| T2 | Price Worth | F1: 0.56 | ⏳ P3 needs fix (use T1P3 pattern) |
| T3 | Environment | F1: 0.56 | ⏳ P3 needs fix |
| T4 | Execution | F1: 0.56 | ⏳ P3 needs fix |
| T5 | Server | F1: 0.56 | ⏳ P3 needs fix |

## Policy Variants (P1-P7)

| Variant | Type | Logic | Description |
|---------|------|-------|-------------|
| **P1** | Base rules | ANY | Simple thresholds (standard markdown) |
| **P2** | Extended rules | ANY | Compound conditions (standard markdown) |
| **P3** | ALL logic | ALL | P1 thresholds + confirmatory condition (tests AND logic) |
| **P4** | Reorder v1 | ANY | Verdicts before Definitions |
| **P5** | Reorder v2 | ANY | Interleaved structure |
| **P6** | XML | ANY | XML-structured format |
| **P7** | Prose | ANY | Flowing narrative format |

**Format variants (P4-P7)** test OBSERVE step's format-agnostic parsing. Same decision
logic as P1, different NL presentation. All share P1's ground truth.

**P3 (ALL logic)** tests LLM's ability to apply AND logic correctly. Design pattern:
1. Copy P1's severity threshold as condition 1
2. Find what property P1's sample restaurants have (e.g., staff_response=dismissive)
3. Add that as condition 2 (ensures same samples trigger the rule)

## Policy ID Format

`T{tier}P{variant}` - no underscore

Examples:
- `T1P1` (tier 1, variant 1)
- `T3P5` (tier 3, variant 5)

## File Locations

- Policy definitions: `src/addm/query/policies/T{1-5}/P{1-7}.yaml`
- Ground truth: Uses G* counterpart (T1P1 → G1_allergy_V1)
- Generated prompts: `data/query/yelp/T{tier}P{variant}_K{k}_prompt.txt`

## G* to T* Transition

**Old system (G*)**: 18 topics × 4 variants = 72 policies (G1_allergy_V1, G1_allergy_V2, ...)
**New system (T*)**: 5 tiers × 7 variants = 35 policies (T1P1, T1P2, ..., T5P7)

The T* system simplifies the benchmark while adding format/logic variants:
- Fewer topics (5 vs 18) - focuses on diverse policy types
- More variants (7 vs 4) - tests format-agnostic parsing and logic handling
- Cleaner naming (T1P3 vs G1_allergy_V3)

## Ground Truth

T* policies generate their own GT files (e.g., `T1P3_K25_groundtruth.json`).

| T* Variant | GT Notes |
|------------|----------|
| P1, P4-P7 | Share same GT (format variants don't change logic) |
| P2 | Own GT (extended conditions) |
| P3 | Own GT (ALL logic with confirmatory conditions) |

**Tier → G* topic mapping** (for raw L0 extractions):
- T1 → G1_allergy
- T2 → G3_price_worth
- T3 → G4_environment
- T4 → G5_execution
- T5 → G6_server

## CLI Examples

```bash
# Single policy
.venv/bin/python -m addm.tasks.cli.run_experiment --policy T1P1 -n 5 --dev

# Multiple policies (comma-separated)
.venv/bin/python -m addm.tasks.cli.run_experiment --policy T1P1,T1P6 --method amos --dev

# All variants for a tier (7 policies)
.venv/bin/python -m addm.tasks.cli.run_experiment --tier T1 --dev

# Multiple tiers (comma-separated, 14 policies)
.venv/bin/python -m addm.tasks.cli.run_experiment --tier T1,T2 --dev

# All 35 policies (omit --policy and --tier)
.venv/bin/python -m addm.tasks.cli.run_experiment --dev
```
