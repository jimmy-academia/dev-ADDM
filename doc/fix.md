# AMOS Fix Log

## Session 2026-01-25

### Fix 1: YAML Parsing Destroying Enum Values

**Issue**: Formula Seed had corrupted enum values like "v", "a", "l", "u", "e" instead of actual values like "severe", "moderate", "mild", "none".

**Root Cause**: In `phase1_helpers.py:parse_yaml_safely()`, the "last resort" fallback was triggered when YAML parsing failed. This fallback replaced ALL values with the literal string `"value"`, destroying data.

**Why Parsing Failed**: The LLM generates `\'` to escape single quotes inside single-quoted YAML strings (e.g., `'someone else\'s experience'`). But YAML single-quoted strings don't support backslash escaping - this syntax is invalid YAML.

**Fix**: Added a new parsing step in `parse_yaml_safely()` that:
1. Detects lines with `\'` inside single-quoted strings
2. Converts them to double-quoted strings with proper escaping
3. Replaces `\'` with just `'` (valid in double quotes)

**File**: `src/addm/methods/amos/phase1_helpers.py`

**Result**: G1_allergy_V1 went from 33% accuracy to 100% accuracy.

---

### Fix 2: Boolean vs String Type Mismatch in Policy IR

**Issue**: G1_hygiene policies use `illness_reported: true` (boolean) in filters, but L0 judgments use `illness_reported: "true"` (string). The GT computation's comparison fails because `True != "true"`.

**Root Cause**: YAML parses `true` as boolean, but the term library defines `illness_reported` with string values `"true"`, `"false"`, `"unknown"`.

**Fix**: Changed policy IR from `illness_reported: true` to `illness_reported: "true"` in:
- `src/addm/query/policies/G1/hygiene/V1.yaml`
- `src/addm/query/policies/G1/hygiene/V2.yaml`

**Files**: Both V1 and V2 hygiene policy files.

---

### Fix 3: G2_group Invalid Enum Value + Ambiguous Query Wording

**Issue**: G2_group_V1 had 0% accuracy. Two problems:

**Problem A: Invalid Enum Value**
Policy IR used `group_outcome: positive` but term library only has: disaster, mixed, success, great. The value "positive" doesn't exist.

**Fix A**: Changed `group_outcome: positive` to `group_outcome: [success, great]` in:
- `src/addm/query/policies/G2/group/V1.yaml`
- `src/addm/query/policies/G2/group/V2.yaml`

**Problem B: Ambiguous Query Wording**
Query said "long waits despite reservations" but policy means `wait_time: excessive`. AMOS interpreted "long" literally as `WAIT_TIME: long` instead of `excessive`.

Also, "poorly handled" was mapping to `GROUP_OUTCOME: disaster` instead of `ACCOMMODATION_QUALITY: poor`.

**Fix B**: Updated condition text in Policy IR to be more explicit:
- "groups being turned away or poorly handled" → "poor accommodation quality (turned away or poorly handled)"
- "long waits despite reservations" → "excessive wait times despite reservations"

**Files**:
- `src/addm/query/policies/G2/group/V1.yaml`
- `src/addm/query/policies/G2/group/V2.yaml`

**Commands**: After fixes, regenerated queries and GT:
```bash
.venv/bin/python -m addm.query.cli.generate --policy G2/group/V1 --k 25
.venv/bin/python -m addm.query.cli.generate --policy G2/group/V1 --k 50
.venv/bin/python -m addm.query.cli.generate --policy G2/group/V1 --k 100
.venv/bin/python -m addm.query.cli.generate --policy G2/group/V1 --k 200
.venv/bin/python -m addm.tasks.cli.compute_gt --policy G2_group_V1 --k 50
```

---

## V1 Testing Progress

### Summary
- **Passing (≥66%)**: 10/18 (55.6%)
- **Failing (<66%)**: 8/18 (44.4%)

| Policy | Attempt 1 | Attempt 2 | Attempt 3 | Status |
|--------|-----------|-----------|-----------|--------|
| G1_allergy_V1 | 100% | - | - | ✓ PASS |
| G1_hygiene_V1 | 33% → 67% (after Fix 2) | - | - | ✓ PASS |
| G1_dietary_V1 | 67% | - | - | ✓ PASS |
| G2_group_V1 | 0% → 33% (after Fix 3) | 33% | - | SKIP (complex) |
| G2_romance_V1 | 67% | - | - | ✓ PASS |
| G2_business_V1 | 33% | 67% | 67% | ✓ PASS |
| G3_price_worth_V1 | 67% | - | - | ✓ PASS |
| G3_hidden_costs_V1 | 33% | 33% | 33% | ✗ FAIL |
| G3_time_value_V1 | 33% | 33% | 33% | ✗ FAIL |
| G4_server_V1 | 67% | - | - | ✓ PASS |
| G4_kitchen_V1 | 33% | 33% | 33% | ✗ FAIL |
| G4_environment_V1 | 67% | - | - | ✓ PASS |
| G5_capacity_V1 | 33% | 0% | 33% | ✗ FAIL |
| G5_execution_V1 | 67% | - | - | ✓ PASS |
| G5_consistency_V1 | 67% | - | - | ✓ PASS |
| G6_uniqueness_V1 | 33% | 33% | 33% | ✗ FAIL |
| G6_comparison_V1 | 33% | 33% | 33% | ✗ FAIL |
| G6_loyalty_V1 | 33% | 33% | 33% | ✗ FAIL |

### Observations
- **G1 (Customer Safety)**: 3/3 passing - AMOS handles safety policies well
- **G2 (Customer Experience)**: 2/3 passing (group policy has extraction complexity)
- **G3 (Customer Value)**: 1/3 passing - hidden_costs and time_value need investigation
- **G4 (Owner Operations)**: 2/3 passing - kitchen policy needs investigation
- **G5 (Owner Performance)**: 2/3 passing - capacity policy inconsistent
- **G6 (Owner Strategy)**: 0/3 passing - ALL G6 policies failing at 33%
