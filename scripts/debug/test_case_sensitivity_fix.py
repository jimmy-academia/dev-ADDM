#!/usr/bin/env python
"""Test script to verify case sensitivity fix in phase2.py.

This tests that FormulaSeedInterpreterFixed correctly handles case mismatches
between Formula Seed field names (uppercase) and LLM extraction keys (lowercase).

Usage:
    .venv/bin/python scripts/debug/test_case_sensitivity_fix.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))


def test_get_field_value():
    """Test _get_field_value helper method."""
    from addm.methods.amos.phase2_case_fix import FormulaSeedInterpreterFixed

    # Minimal seed for instantiation
    seed = {"filter": {}, "extract": {"fields": []}, "compute": [], "output": []}

    class MockLLM:
        pass

    interpreter = FormulaSeedInterpreterFixed(seed=seed, llm=MockLLM())

    # Test extraction with lowercase keys (typical LLM output)
    extraction = {
        "incident_severity": "severe",
        "account_type": "regular",
        "review_id": "123",
    }

    # These should all work (case-insensitive)
    assert interpreter._get_field_value(extraction, "incident_severity") == "severe"
    assert interpreter._get_field_value(extraction, "INCIDENT_SEVERITY") == "severe"
    assert interpreter._get_field_value(extraction, "Incident_Severity") == "severe"  # lowercase fallback
    assert interpreter._get_field_value(extraction, "account_type") == "regular"
    assert interpreter._get_field_value(extraction, "ACCOUNT_TYPE") == "regular"

    # Non-existent field
    assert interpreter._get_field_value(extraction, "non_existent") is None

    print("[PASS] _get_field_value works correctly")


def test_matches_condition():
    """Test _matches_condition with case-insensitive field lookups."""
    from addm.methods.amos.phase2_case_fix import FormulaSeedInterpreterFixed

    seed = {"filter": {}, "extract": {"fields": []}, "compute": [], "output": []}

    class MockLLM:
        pass

    interpreter = FormulaSeedInterpreterFixed(seed=seed, llm=MockLLM())

    # Extraction with lowercase keys
    extraction = {
        "incident_severity": "severe",
        "account_type": "premium",
    }

    # Condition with uppercase field (Formula Seed style)
    condition_upper = {"INCIDENT_SEVERITY": "severe"}
    assert interpreter._matches_condition(extraction, condition_upper) is True

    # Condition with lowercase field
    condition_lower = {"incident_severity": "severe"}
    assert interpreter._matches_condition(extraction, condition_lower) is True

    # Condition with field/equals syntax
    condition_fe = {"field": "INCIDENT_SEVERITY", "equals": "severe"}
    assert interpreter._matches_condition(extraction, condition_fe) is True

    # List matching
    condition_list = {"INCIDENT_SEVERITY": ["severe", "moderate"]}
    assert interpreter._matches_condition(extraction, condition_list) is True

    # Wrong value should fail
    condition_wrong = {"INCIDENT_SEVERITY": "mild"}
    assert interpreter._matches_condition(extraction, condition_wrong) is False

    print("[PASS] _matches_condition works correctly")


def test_compute_count():
    """Test _compute_count with case-insensitive conditions."""
    from addm.methods.amos.phase2_case_fix import FormulaSeedInterpreterFixed

    seed = {"filter": {}, "extract": {"fields": []}, "compute": [], "output": []}

    class MockLLM:
        pass

    interpreter = FormulaSeedInterpreterFixed(seed=seed, llm=MockLLM())

    # Set up extractions with lowercase keys
    interpreter._extractions = [
        {"incident_severity": "severe", "review_id": "1"},
        {"incident_severity": "moderate", "review_id": "2"},
        {"incident_severity": "severe", "review_id": "3"},
    ]

    # Count with uppercase field name (Formula Seed style)
    op_def = {
        "name": "N_SEVERE",
        "op": "count",
        "where": {"INCIDENT_SEVERITY": "severe"},
    }
    assert interpreter._compute_count(op_def) == 2

    # Count with list condition
    op_def_list = {
        "name": "N_BAD",
        "op": "count",
        "where": {"INCIDENT_SEVERITY": ["severe", "moderate"]},
    }
    assert interpreter._compute_count(op_def_list) == 3

    print("[PASS] _compute_count works correctly")


def test_eval_sql_case():
    """Test SQL CASE expression evaluation with case-insensitive field lookups."""
    from addm.methods.amos.phase2_case_fix import FormulaSeedInterpreterFixed

    seed = {"filter": {}, "extract": {"fields": []}, "compute": [], "output": []}

    class MockLLM:
        pass

    interpreter = FormulaSeedInterpreterFixed(seed=seed, llm=MockLLM())

    # Extraction with lowercase keys
    extraction = {"incident_severity": "severe"}

    # SQL CASE with uppercase field (Formula Seed style)
    expr = "CASE WHEN INCIDENT_SEVERITY = 'severe' THEN 15 WHEN INCIDENT_SEVERITY = 'moderate' THEN 8 ELSE 0 END"
    assert interpreter._eval_sql_case_expr(expr, extraction) == 15.0

    # Test moderate
    extraction_mod = {"incident_severity": "moderate"}
    assert interpreter._eval_sql_case_expr(expr, extraction_mod) == 8.0

    # Test ELSE case
    extraction_none = {"incident_severity": "none"}
    assert interpreter._eval_sql_case_expr(expr, extraction_none) == 0.0

    print("[PASS] _eval_sql_case_expr works correctly")


def test_apply_case_to_extraction():
    """Test case rule application with case-insensitive field lookups."""
    from addm.methods.amos.phase2_case_fix import FormulaSeedInterpreterFixed

    seed = {"filter": {}, "extract": {"fields": []}, "compute": [], "output": []}

    class MockLLM:
        pass

    interpreter = FormulaSeedInterpreterFixed(seed=seed, llm=MockLLM())

    # Extraction with lowercase keys
    extraction = {"incident_severity": "severe"}

    # Rules using uppercase source (Formula Seed style)
    rules = [
        {"when": "severe", "then": 15},
        {"when": "moderate", "then": 8},
        {"when": "mild", "then": 3},
        {"else": 0},
    ]

    # Apply with uppercase source name
    result = interpreter._apply_case_to_extraction(extraction, "INCIDENT_SEVERITY", rules)
    assert result == 15

    # Test moderate
    extraction_mod = {"incident_severity": "moderate"}
    result_mod = interpreter._apply_case_to_extraction(extraction_mod, "INCIDENT_SEVERITY", rules)
    assert result_mod == 8

    print("[PASS] _apply_case_to_extraction works correctly")


def test_original_still_broken():
    """Verify that the original FormulaSeedInterpreter still has the case sensitivity bug."""
    from addm.methods.amos.phase2 import FormulaSeedInterpreter

    seed = {"filter": {}, "extract": {"fields": []}, "compute": [], "output": []}

    class MockLLM:
        pass

    interpreter = FormulaSeedInterpreter(seed=seed, llm=MockLLM())

    # Set up extractions with lowercase keys
    interpreter._extractions = [
        {"incident_severity": "severe", "review_id": "1"},
        {"incident_severity": "moderate", "review_id": "2"},
    ]

    # This WILL FAIL in original (returns 0 instead of 1) because of case mismatch
    op_def = {
        "name": "N_SEVERE",
        "op": "count",
        "where": {"INCIDENT_SEVERITY": "severe"},  # Uppercase in condition
    }

    count = interpreter._compute_count(op_def)
    if count == 0:
        print("[CONFIRMED] Original FormulaSeedInterpreter has case sensitivity bug (count=0 when it should be 1)")
    else:
        print(f"[UNEXPECTED] Original returned count={count}, bug may have been fixed elsewhere")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Testing case sensitivity fix for phase2.py")
    print("=" * 60)
    print()

    test_get_field_value()
    test_matches_condition()
    test_compute_count()
    test_eval_sql_case()
    test_apply_case_to_extraction()

    print()
    print("-" * 60)
    print("Verifying original has the bug:")
    print("-" * 60)
    test_original_still_broken()

    print()
    print("=" * 60)
    print("All tests passed! The fix is working correctly.")
    print()
    print("To use the fix, import from phase2_case_fix:")
    print("  from addm.methods.amos.phase2_case_fix import FormulaSeedInterpreterFixed")
    print()
    print("Or apply the _get_field_value helper and updated methods to phase2.py directly.")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
