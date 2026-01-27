# Benchmark: T* System (35 Policies)

## Structure

**5 Tiers × 7 Variants = 35 Policies**

| Tier | Topic | Source |
|------|-------|--------|
| T1 | Allergy Safety | F1: 0.64 |
| T2 | Price Worth | F1: 0.56 |
| T3 | Environment | F1: 0.56 |
| T4 | Execution | F1: 0.56 |
| T5 | Server | F1: 0.56 |

## Policy Variants (P1-P7)

| Variant | Type | Description |
|---------|------|-------------|
| **P1** | Base rules | Simple thresholds (standard markdown) |
| **P2** | Extended rules | Compound conditions (standard markdown) |
| **P3** | TBD rules | Rule variation to be defined |
| **P4** | Reorder v1 | Verdicts before Definitions |
| **P5** | Reorder v2 | Interleaved structure |
| **P6** | XML | XML-structured format |
| **P7** | Prose | Flowing narrative format |

**Format variants (P4-P7)** test OBSERVE step's format-agnostic parsing. Same decision
logic as P1, different NL presentation. All share P1's ground truth.

## Policy ID Format

`T{tier}P{variant}` - no underscore

Examples:
- `T1P1` (tier 1, variant 1)
- `T3P5` (tier 3, variant 5)

## File Locations

- Policy definitions: `src/addm/query/policies/T{1-5}/P{1-7}.yaml`
- Ground truth: Uses G* counterpart (T1P1 → G1_allergy_V1)
- Generated prompts: `data/query/yelp/T{tier}P{variant}_K{k}_prompt.txt`

## Ground Truth Mapping

T* policies use G* GT files from previous benchmark:

| T* Variant | G* GT Source |
|------------|--------------|
| P1, P3-P7 | V1 (base rules) |
| P2 | V2 (extended rules) |

Tier → G* topic mapping:
- T1 → G1_allergy
- T2 → G3_price_worth
- T3 → G4_environment
- T4 → G5_execution
- T5 → G4_server

## CLI Examples

```bash
# Single policy
.venv/bin/python -m addm.tasks.cli.run_experiment --policy T1P1 -n 5 --dev

# All variants for a tier
.venv/bin/python -m addm.tasks.cli.run_experiment --tier T1 --dev

# Multiple policies
.venv/bin/python -m addm.tasks.cli.run_experiment --policy T1P1,T1P6 --method amos --dev

# All 35 policies
.venv/bin/python -m addm.tasks.cli.run_experiment --dev
```
