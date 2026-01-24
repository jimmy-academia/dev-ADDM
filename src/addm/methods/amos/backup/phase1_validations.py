"""Phase 1 Full Validation Functions (Archived).

Contains comprehensive validation functions that were replaced by minimal
structural validation in phase1_helpers.py.

These functions perform detailed semantic validation:
- Expression syntax checking
- Enum value consistency
- Field reference validation
- Full Formula Seed structure validation

Kept for reference and potential future use if detailed validation is needed.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def validate_expression(expr: str, field_name: str) -> Optional[str]:
    """Validate a Python expression is syntactically correct and semantically meaningful.

    Args:
        expr: Python expression string
        field_name: Name of the field (for error messages)

    Returns:
        Error message if invalid, None if valid
    """
    if not expr or not isinstance(expr, str):
        return None  # Empty is ok

    # Check for syntax errors
    try:
        compile(expr, f"<{field_name}>", "eval")
    except SyntaxError as e:
        return f"{field_name}: SyntaxError - {e.msg}"

    # Semantic check: detect expressions that are just string literals containing Python code
    stripped = expr.strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        inner = stripped[1:-1]
        if any(keyword in inner for keyword in [' if ', ' else ', ' and ', ' or ']):
            return (
                f"{field_name}: Expression appears to be a string literal containing Python code. "
                f"Remove the outer quotes. Got: {expr[:50]}..."
            )

    if stripped.startswith("'") and stripped.endswith("'"):
        inner = stripped[1:-1]
        if any(keyword in inner for keyword in [' if ', ' else ', ' and ', ' or ']):
            return (
                f"{field_name}: Expression appears to be a string literal containing Python code. "
                f"Remove the outer quotes. Got: {expr[:50]}..."
            )

    return None


def validate_enum_consistency(seed: Dict[str, Any]) -> List[str]:
    """Validate that compute.where values match extract.fields enum values.

    Catches mismatches like compute using "Moderate" when extract defines "Moderate incident".
    This is a warning-level validation (returns warnings, not blocking errors).

    Args:
        seed: Formula Seed dict

    Returns:
        List of warning messages about enum inconsistencies
    """
    warnings = []

    # Build map of field -> valid enum values (lowercased for comparison)
    enum_values: Dict[str, set] = {}
    for field in seed.get("extract", {}).get("fields", []):
        if field.get("type") == "enum":
            name = field.get("name", "")
            values = field.get("values", {})
            if isinstance(values, dict):
                # Store both original and lowercase for matching
                enum_values[name.upper()] = set(v.lower() for v in values.keys())

    # Check compute.where conditions
    for op in seed.get("compute", []):
        op_name = op.get("name", "unknown")
        where = op.get("where", {})

        if not isinstance(where, dict):
            continue

        for field, expected in where.items():
            # Skip logical operators and field/equals keys
            if field in ("and", "or", "field", "equals", "not_equals"):
                continue

            field_upper = field.upper()
            if field_upper not in enum_values:
                continue  # Not an enum field or unknown field

            valid_values = enum_values[field_upper]

            # Check each expected value
            expected_list = expected if isinstance(expected, list) else [expected]
            for e in expected_list:
                if not isinstance(e, str):
                    continue
                e_lower = e.lower()
                # Check if value exists exactly or is a substring/superstring of valid values
                exact_match = e_lower in valid_values
                partial_match = any(e_lower in v or v in e_lower for v in valid_values)

                if not exact_match and partial_match:
                    # This is a potential mismatch that fuzzy matching will handle,
                    # but we should warn about it
                    warnings.append(
                        f"compute.{op_name}: where value '{e}' is a partial match for {field} "
                        f"enum values: {sorted(valid_values)}. Consider using exact enum keys."
                    )
                elif not exact_match and not partial_match:
                    warnings.append(
                        f"compute.{op_name}: where value '{e}' not found in {field} "
                        f"enum values: {sorted(valid_values)}"
                    )

    return warnings


def validate_field_references(seed: Dict[str, Any]) -> List[str]:
    """Validate that compute operations reference valid extraction fields.

    Also validates that where clauses only reference enum fields (not string fields).

    Args:
        seed: Formula Seed dict

    Returns:
        List of field reference errors
    """
    errors = []

    # Get extraction field names and their types
    extraction_fields = set()
    field_types = {}  # field_name.upper() -> "enum" or "string"
    for field in seed.get("extract", {}).get("fields", []):
        field_name = field.get("name", "")
        field_type = field.get("type", "string")
        if field_name:
            extraction_fields.add(field_name.upper())
            extraction_fields.add(field_name)
            field_types[field_name.upper()] = field_type

    # Get computed value names
    computed_names = set()
    for i, op in enumerate(seed.get("compute", [])):
        if not isinstance(op, dict):
            errors.append(
                f"compute[{i}]: Expected dict but got {type(op).__name__}: {str(op)[:100]}"
            )
            continue
        name = op.get("name", "")
        if name:
            computed_names.add(name.upper())
            computed_names.add(name)

    # Pattern to find field references in CASE WHEN expressions
    case_when_pattern = re.compile(r"WHEN\s+(\w+)\s*=", re.IGNORECASE)

    for i, op in enumerate(seed.get("compute", [])):
        if not isinstance(op, dict):
            continue
        op_name = op.get("name", "unknown")
        op_type = op.get("op", "")

        # Check CASE WHEN expressions in sum operations
        if op_type == "sum":
            expr = op.get("expr", "")
            if expr.upper().startswith("CASE"):
                matches = case_when_pattern.findall(expr)
                for field_ref in matches:
                    if field_ref.upper() not in extraction_fields:
                        errors.append(
                            f"compute.{op_name}: CASE expression references unknown field '{field_ref}'. "
                            f"Available extraction fields: {sorted(extraction_fields)}"
                        )

        # Check where conditions
        where = op.get("where", {})
        if isinstance(where, dict):
            for field_ref in where.keys():
                if field_ref in ("and", "or", "field", "equals", "not_equals"):
                    continue
                field_upper = field_ref.upper()
                if field_upper not in extraction_fields:
                    errors.append(
                        f"compute.{op_name}: where condition references unknown field '{field_ref}'. "
                        f"Available extraction fields: {sorted(extraction_fields)}"
                    )
                elif field_types.get(field_upper) == "string":
                    # Where clauses can only filter by enum fields, not string fields
                    errors.append(
                        f"compute.{op_name}: where condition references string field '{field_ref}'. "
                        f"Count/sum where clauses can only filter by enum fields. "
                        f"To count by this criterion, create an enum field instead."
                    )

        # Check case operations that reference extraction fields
        if op_type == "case":
            source = op.get("source", "")
            if source and source.upper() not in extraction_fields and source.upper() not in computed_names:
                later_computed = set()
                found = False
                for later_op in seed.get("compute", []):
                    if not isinstance(later_op, dict):
                        continue
                    if later_op.get("name") == op_name:
                        found = True
                    if found:
                        later_name = later_op.get("name", "")
                        if later_name:
                            later_computed.add(later_name.upper())

                if source.upper() not in later_computed:
                    errors.append(
                        f"compute.{op_name}: case source '{source}' not found in extraction fields or computed values"
                    )

    return errors


def validate_formula_seed_full(seed: Dict[str, Any]) -> List[str]:
    """Full validation of Formula Seed structure, expressions, and field references.

    This is the comprehensive validation that checks:
    - Required keys and structure
    - Field name uniqueness
    - Enum value format
    - outcome_field and none_values
    - VERDICT operation
    - Field references in compute operations
    - Enum consistency

    Args:
        seed: Formula Seed dict

    Returns:
        List of validation errors (empty list if valid)
    """
    errors = []

    # Required keys (filter is NO LONGER required - simplified schema)
    required_keys = ["extract", "compute", "output"]
    for key in required_keys:
        if key not in seed:
            errors.append(f"Missing required key: {key}")

    if errors:
        return errors  # Can't continue without required keys

    # Validate extract
    if "fields" not in seed["extract"]:
        errors.append("extract missing 'fields'")
    elif not isinstance(seed["extract"]["fields"], list):
        errors.append("extract.fields must be a list")
    else:
        # Check for duplicate field names
        seen_field_names = set()
        for i, field in enumerate(seed["extract"]["fields"]):
            if not isinstance(field, dict):
                continue
            field_name = field.get("name", "")
            if field_name:
                upper_name = field_name.upper()
                if upper_name in seen_field_names:
                    errors.append(
                        f"extract.fields: Duplicate field name '{field_name}'. "
                        f"Each field must appear exactly once."
                    )
                seen_field_names.add(upper_name)

        # Validate field values format (must be dict, not list)
        for i, field in enumerate(seed["extract"]["fields"]):
            if not isinstance(field, dict):
                continue
            field_name = field.get("name", f"field[{i}]")
            if field.get("type") == "enum":
                values = field.get("values")
                if values is not None and not isinstance(values, dict):
                    errors.append(
                        f"extract.fields[{field_name}].values must be a dict "
                        f"(e.g., {{\"value1\": \"desc1\"}}), not {type(values).__name__}"
                    )

    # Validate outcome_field and none_values (required for generalizable incident detection)
    extract = seed.get("extract", {})
    if "outcome_field" not in extract:
        errors.append(
            "extract.outcome_field is required (e.g., 'INCIDENT_SEVERITY', 'QUALITY_LEVEL')"
        )
    if "none_values" not in extract:
        errors.append(
            "extract.none_values is required (e.g., ['none', 'n/a'])"
        )
    elif not isinstance(extract.get("none_values"), list):
        errors.append("extract.none_values must be a list")

    # Validate compute
    if not isinstance(seed["compute"], list):
        errors.append("compute must be a list")
    elif len(seed["compute"]) == 0:
        errors.append("compute must not be empty - at minimum needs N_INCIDENTS and VERDICT operations")
    else:
        # Validate VERDICT operation exists
        has_verdict = False
        for op in seed["compute"]:
            if isinstance(op, dict) and op.get("name") == "VERDICT":
                has_verdict = True
                if op.get("op") != "case":
                    errors.append(
                        f"VERDICT operation must have op='case', got op='{op.get('op')}'"
                    )
                elif "rules" not in op:
                    errors.append("VERDICT case operation must have 'rules' list")
                break
        if not has_verdict:
            errors.append(
                "compute must include a VERDICT operation with op='case' and 'rules'"
            )

    # Validate output
    if not isinstance(seed["output"], list):
        errors.append("output must be a list")

    # Validate field references in compute operations
    field_errors = validate_field_references(seed)
    errors.extend(field_errors)

    # Validate enum consistency (warnings, not blocking errors)
    enum_warnings = validate_enum_consistency(seed)
    for warning in enum_warnings:
        logger.warning(f"Enum consistency warning: {warning}")

    # Add strict enum consistency validation that produces errors for complete mismatches
    # This catches cases where compute.where uses values that don't exist in extract.fields enums
    extract_fields = seed.get("extract", {}).get("fields", [])
    field_values = {}
    for f in extract_fields:
        if f.get("type") == "enum" and isinstance(f.get("values"), dict):
            field_values[f["name"]] = set(f["values"].keys())
            # Also store lowercase version for case-insensitive matching
            field_values[f["name"].upper()] = set(v.lower() for v in f["values"].keys())

    for op in seed.get("compute", []):
        where = op.get("where", {})
        if not isinstance(where, dict):
            continue
        for field_name, values in where.items():
            # Skip logical operators and special keys
            if field_name in ("and", "or", "field", "equals", "not_equals"):
                continue

            field_upper = field_name.upper()
            if field_upper not in field_values:
                continue  # Field not defined as enum, skip

            valid_values = field_values[field_upper]

            # Check each expected value
            values_list = values if isinstance(values, list) else [values]
            for v in values_list:
                if not isinstance(v, str):
                    continue
                v_lower = v.lower()
                # Check if value exists (exact match or fuzzy)
                exact_match = v_lower in valid_values
                partial_match = any(v_lower in vv or vv in v_lower for vv in valid_values)

                if not exact_match and not partial_match:
                    errors.append(
                        f"compute.{op.get('name')}: where value '{v}' not found in {field_name} "
                        f"enum values: {sorted(valid_values)}"
                    )

    return errors
