# Phase2.py Case Sensitivity Fix

## Problem

Formula Seeds use uppercase field names (e.g., `INCIDENT_SEVERITY`, `ACCOUNT_TYPE`) while LLM extractions return lowercase keys (e.g., `incident_severity`, `account_type`). The following methods do exact key matching, causing conditions to fail silently:

- `_matches_condition()`
- `_matches_single_condition()`
- `_compute_count()`
- `_compute_sum()`
- `_eval_sql_case_expr()`
- `_eval_single_case()`
- `_apply_case_to_extraction()`

## Solution

Add a helper method `_get_field_value()` that tries multiple case variants, then use it everywhere field values are accessed from extractions.

## Files

- **Fix module**: `src/addm/methods/amos/phase2_case_fix.py` (subclass with fixed methods)
- **Test script**: `scripts/debug/test_case_sensitivity_fix.py` (verifies fix works)

## Changes to Apply to phase2.py

### 1. Add helper method after `self._executor = SafeExpressionExecutor()` in `__init__`:

```python
    def _get_field_value(self, extraction: Dict[str, Any], field_name: str) -> Any:
        """Get field value from extraction, handling case-insensitive lookup.

        Formula Seeds use uppercase field names (e.g., INCIDENT_SEVERITY) while
        LLM extractions often return lowercase (e.g., incident_severity). This
        helper tries multiple case variants.

        Args:
            extraction: Extraction dict with field values
            field_name: Field name to look up

        Returns:
            Field value if found, None otherwise
        """
        # Try exact match first
        if field_name in extraction:
            return extraction[field_name]
        # Try lowercase
        lower_name = field_name.lower()
        if lower_name in extraction:
            return extraction[lower_name]
        # Try uppercase
        upper_name = field_name.upper()
        if upper_name in extraction:
            return extraction[upper_name]
        return None
```

### 2. Update `_matches_condition()` - replace `extraction.get(field)` with `self._get_field_value(extraction, field)`:

Lines to change:
- Line 719: `actual = extraction.get(field)` -> `actual = self._get_field_value(extraction, field)`
- Line 760: `actual = extraction.get(field)` -> `actual = self._get_field_value(extraction, field)`

### 3. Update `_matches_single_condition()`:

- Line 786: `actual = extraction.get(field)` -> `actual = self._get_field_value(extraction, field)`

### 4. Update `_eval_single_case()`:

Replace lines 863-867:
```python
        for field, value, then_value in matches:
            actual_value = extraction.get(field, extraction.get(field.upper(), None))
            if actual_value is None:
                # Try lowercase field name
                actual_value = extraction.get(field.lower(), "")
```

With:
```python
        for field, value, then_value in matches:
            actual_value = self._get_field_value(extraction, field)
            if actual_value is None:
                actual_value = ""
```

Replace lines 874-877:
```python
        for field, values_str, then_value in in_matches:
            actual_value = extraction.get(field, extraction.get(field.upper(), None))
            if actual_value is None:
                actual_value = extraction.get(field.lower(), "")
```

With:
```python
        for field, values_str, then_value in in_matches:
            actual_value = self._get_field_value(extraction, field)
            if actual_value is None:
                actual_value = ""
```

### 5. Update `_apply_case_to_extraction()`:

Replace line 946:
```python
        source_value = extraction.get(source, "none")
```

With:
```python
        source_value = self._get_field_value(extraction, source)
        if source_value is None:
            source_value = "none"
```

Also update the string comparison at line 978 to be case-insensitive:
```python
                if str(source_value).lower() == str(when).lower():
```

## Verification

Run the test script:
```bash
.venv/bin/python scripts/debug/test_case_sensitivity_fix.py
```

Expected output:
```
[PASS] _get_field_value works correctly
[PASS] _matches_condition works correctly
[PASS] _compute_count works correctly
[PASS] _eval_sql_case_expr works correctly
[PASS] _apply_case_to_extraction works correctly
[CONFIRMED] Original FormulaSeedInterpreter has case sensitivity bug
```

## Immediate Usage

Import the fixed class instead of the original:

```python
# Instead of:
from addm.methods.amos.phase2 import FormulaSeedInterpreter

# Use:
from addm.methods.amos.phase2_case_fix import FormulaSeedInterpreterFixed as FormulaSeedInterpreter
```
