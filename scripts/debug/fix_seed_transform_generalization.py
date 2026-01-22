#!/usr/bin/env python3
"""Documentation of seed_transform.py fixes for AMOS generalization.

## Summary

Two bugs fixed in `src/addm/methods/amos/seed_transform.py`:

### Bug 1: G5 crash - Boolean in verdict labels (FIXED)

**Problem**: `normalize_verdict_label()` called `.lower()` on non-string values.
G5 policies had case rules with boolean "then" values like `{"when": "...", "then": true}`.

**Error**:
    AttributeError: 'bool' object has no attribute 'lower'
    at seed_transform.py:66

**Fix**: Added type check at start of `normalize_verdict_label()`:
    if not isinstance(label, str):
        return label

**Result**: G5_capacity_V0 now runs and returns correct verdict "Excellent"


### Bug 2: G2/G5 verdict labels not mapped (FIXED)

**Problem**: `VERDICT_LABEL_MAP` only contained risk-based verdicts (Critical/High/Low Risk).
G2 policies use "Recommended"/"Not Recommended", G5 uses "Excellent"/"Satisfactory"/"Needs Improvement".

**Error**: Verdicts returned as numeric `0` instead of string labels.

**Fix**: Expanded `VERDICT_LABEL_MAP` to include:
    - G2: "Recommended", "Not Recommended"
    - G5: "Excellent", "Satisfactory", "Needs Improvement"

**Result**: G2_group_V1 now returns string labels ("Not Recommended" instead of `0`)


## Test Results After Fix

| Policy | Before Fix | After Fix |
|--------|------------|-----------|
| G5_capacity_V0 | CRASH | ✓ Excellent (100% accuracy) |
| G2_group_V1 | verdict: `0` | verdict: "Not Recommended" (string) |

Note: G2_group_V1 still has 0% accuracy - the verdict is wrong ("Not Recommended" vs GT "Recommended").
This is a **scoring logic issue**, not the type bug we fixed. The type bug is fixed.


## Git Diff Summary

```diff
# 1. Added Union to imports
-from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
+from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

# 2. Expanded VERDICT_LABEL_MAP with G2/G5 verdicts
+    # === Recommendation-based verdicts (G2) ===
+    "recommended": "Recommended",
+    "not recommended": "Not Recommended",
+    ...
+    # === Performance-based verdicts (G5) ===
+    "excellent": "Excellent",
+    "satisfactory": "Satisfactory",
+    ...

# 3. Fixed normalize_verdict_label() to handle non-strings
-def normalize_verdict_label(label: str) -> str:
+def normalize_verdict_label(label: Union[str, Any]) -> Union[str, Any]:
+    # Handle non-string types (booleans, numbers, None)
+    if not isinstance(label, str):
+        return label
```


## Handoff Instructions

The fixes are already applied to `seed_transform.py`. To merge:

1. Review the diff: `git diff src/addm/methods/amos/seed_transform.py`
2. Stage and commit: `git add src/addm/methods/amos/seed_transform.py`
3. Commit message suggestion:
   ```
   fix(amos): handle non-string verdict labels and add G2/G5 mappings

   - normalize_verdict_label() now handles bool/int/None (fixes G5 crash)
   - VERDICT_LABEL_MAP includes Recommended/Not Recommended (G2)
   - VERDICT_LABEL_MAP includes Excellent/Satisfactory/Needs Improvement (G5)
   ```


## Remaining Issues (Not Fixed)

1. **G2 scoring accuracy**: G2_group_V1 returns wrong verdict. Needs investigation of
   G2-specific scoring logic (different from risk-based policies).

2. **G3/G1 recall**: Low recall on G3_hidden_costs and G1_hygiene policies.
   Not a code bug - the sweep is missing relevant reviews.
"""

# Test verification code
def verify_fixes():
    """Quick verification that the fixes work."""
    import sys
    from pathlib import Path

    # Add project to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root / "src"))

    from addm.methods.amos.seed_transform import normalize_verdict_label, VERDICT_LABEL_MAP

    print("Testing normalize_verdict_label():")
    test_cases = [
        # Risk-based (G1, G3, G4)
        ("critical", "Critical Risk"),
        ("High Risk", "High Risk"),
        # G2 verdicts
        ("recommended", "Recommended"),
        ("not recommended", "Not Recommended"),
        # G5 verdicts
        ("excellent", "Excellent"),
        ("needs improvement", "Needs Improvement"),
        # Non-string types (should pass through)
        (True, True),
        (False, False),
        (0, 0),
        (None, None),
    ]

    all_passed = True
    for input_val, expected in test_cases:
        result = normalize_verdict_label(input_val)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"  {status} {input_val!r} → {result!r} (expected {expected!r})")

    print(f"\nVERDICT_LABEL_MAP entries: {len(VERDICT_LABEL_MAP)}")
    print(f"Has G2 verdicts: {'Recommended' in VERDICT_LABEL_MAP.values()}")
    print(f"Has G5 verdicts: {'Excellent' in VERDICT_LABEL_MAP.values()}")

    return all_passed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Run verification tests")
    args = parser.parse_args()

    if args.verify:
        success = verify_fixes()
        exit(0 if success else 1)
    else:
        print(__doc__)
