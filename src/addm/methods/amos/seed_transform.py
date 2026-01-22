"""Formula Seed Schema Transformation.

Transforms LLM-generated Formula Seeds into the format expected by phase2.py.

Problem: Phase 1 LLM often generates SQL-style expressions:
    {"name": "N_INCIDENTS", "expression": "SUM(CASE WHEN ACCOUNT_TYPE = 'Firsthand' THEN 1 ELSE 0 END)"}

But phase2.py expects operation-style definitions:
    {"name": "N_INCIDENTS", "op": "count", "where": {"ACCOUNT_TYPE": "firsthand"}}

This module provides transformations to bridge the gap.

Also handles:
- Verdict label normalization ("Critical" → "Critical Risk")
- Field name case normalization
"""

import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

logger = logging.getLogger(__name__)


# =============================================================================
# Verdict Label Mapping
# =============================================================================

# Standard verdict labels (GT format) vs common LLM abbreviations
VERDICT_LABEL_MAP = {
    # Abbreviated → Full (GT format)
    "critical": "Critical Risk",
    "high": "High Risk",
    "low": "Low Risk",
    # Case variations
    "Critical": "Critical Risk",
    "High": "High Risk",
    "Low": "Low Risk",
    # Already correct - pass through
    "Critical Risk": "Critical Risk",
    "High Risk": "High Risk",
    "Low Risk": "Low Risk",
}


def normalize_verdict_label(label: str) -> str:
    """Normalize a verdict label to GT format.

    Args:
        label: Raw verdict label from LLM

    Returns:
        Normalized verdict label (e.g., "Critical Risk")
    """
    if not label:
        return label

    # Try exact match first
    if label in VERDICT_LABEL_MAP:
        return VERDICT_LABEL_MAP[label]

    # Try case-insensitive match
    label_lower = label.lower().strip()
    for key, value in VERDICT_LABEL_MAP.items():
        if key.lower() == label_lower:
            return value

    # Return as-is if no mapping found
    return label


# =============================================================================
# SQL Expression Parsing
# =============================================================================

def parse_sql_sum_case(expr: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Parse SQL SUM(CASE WHEN...) expression into op/expr/where format.

    Args:
        expr: SQL expression like "SUM(CASE WHEN ACCOUNT_TYPE = 'Firsthand' THEN 1 ELSE 0 END)"

    Returns:
        Tuple of (parsed_expr, where_condition) or (None, None) if not parseable
    """
    expr_upper = expr.upper().strip()

    # Pattern 1: SUM(CASE WHEN field = 'value' THEN x ELSE 0 END)
    # This is a count operation
    count_pattern = r"SUM\s*\(\s*CASE\s+WHEN\s+(\w+)\s*=\s*['\"]([^'\"]+)['\"]\s+THEN\s+1\s+ELSE\s+0\s+END\s*\)"
    count_match = re.search(count_pattern, expr, re.IGNORECASE)
    if count_match:
        field, value = count_match.groups()
        return None, {field: value}  # Returns where condition for count op

    # Pattern 2: SUM(CASE WHEN field = 'v1' THEN p1 WHEN field = 'v2' THEN p2 ... ELSE 0 END)
    # This is a sum operation with CASE expression
    sum_case_pattern = r"SUM\s*\(\s*(CASE\s+.+\s+END)\s*\)"
    sum_match = re.search(sum_case_pattern, expr, re.IGNORECASE | re.DOTALL)
    if sum_match:
        case_expr = sum_match.group(1)
        return case_expr, None  # Returns CASE expression for sum op

    return None, None


def parse_simple_case(expr: str) -> List[Dict[str, Any]]:
    """Parse SQL CASE WHEN ... END into rules list.

    Args:
        expr: SQL CASE expression

    Returns:
        List of rule dicts: [{"when": "...", "then": "..."}, {"else": "..."}]
    """
    rules = []

    # Extract WHEN clauses
    when_pattern = r"WHEN\s+(.+?)\s+THEN\s+['\"]?([^'\"]+?)['\"]?\s*(?=WHEN|ELSE|END)"
    when_matches = re.findall(when_pattern, expr, re.IGNORECASE | re.DOTALL)

    for condition, result in when_matches:
        # Clean up condition
        condition = condition.strip()
        result = result.strip()

        # Normalize condition format
        # "SCORE >= 8" → ">= 8" (if source is SCORE)
        # Keep as-is for compound conditions
        rules.append({"when": condition, "then": normalize_verdict_label(result)})

    # Extract ELSE clause
    else_pattern = r"ELSE\s+['\"]?([^'\"]+?)['\"]?\s*END"
    else_match = re.search(else_pattern, expr, re.IGNORECASE)
    if else_match:
        else_value = else_match.group(1).strip()
        rules.append({"else": normalize_verdict_label(else_value)})

    return rules


def extract_where_from_expression(expr: str) -> Optional[Dict[str, Any]]:
    """Extract WHERE conditions from SQL expression.

    Looks for patterns like "WHERE ACCOUNT_TYPE = 'firsthand'" or
    "WHEN ACCOUNT_TYPE = 'Firsthand'" at the start of the expression.

    Args:
        expr: SQL expression

    Returns:
        Where condition dict or None
    """
    # Pattern: field = 'value' (possibly with AND)
    conditions = {}

    # Simple equality: FIELD = 'value'
    eq_pattern = r"(\w+)\s*=\s*['\"]([^'\"]+)['\"]"
    matches = re.findall(eq_pattern, expr, re.IGNORECASE)

    for field, value in matches:
        # Skip if this looks like a CASE WHEN condition (followed by THEN)
        context_pattern = rf"{field}\s*=\s*['\"]?{re.escape(value)}['\"]?\s+THEN"
        if re.search(context_pattern, expr, re.IGNORECASE):
            continue
        conditions[field] = value

    return conditions if conditions else None


# =============================================================================
# Compute Operation Transformation
# =============================================================================

def transform_compute_operation(op_def: Dict[str, Any], extraction_fields: List[str]) -> Dict[str, Any]:
    """Transform a single compute operation to phase2.py format.

    Args:
        op_def: Original operation definition
        extraction_fields: List of valid extraction field names (for validation)

    Returns:
        Transformed operation definition
    """
    result = deepcopy(op_def)
    name = op_def.get("name", "")

    # If already has "op" key, just validate and return
    if "op" in op_def:
        # Normalize verdict labels in case rules
        if op_def["op"] == "case" and "rules" in op_def:
            result["rules"] = _normalize_case_rules(op_def["rules"])
        return result

    # Handle "expression" key (LLM format) → transform to "op" format
    if "expression" in op_def:
        expr = op_def["expression"]
        expr_upper = expr.upper().strip()

        # Pattern 1: COUNT-like SUM(CASE WHEN ... THEN 1 ELSE 0)
        if "SUM" in expr_upper and "THEN 1 ELSE 0" in expr_upper:
            case_expr, where = parse_sql_sum_case(expr)
            if where:
                result = {
                    "name": name,
                    "op": "count",
                    "where": where,
                }
                logger.debug(f"Transformed {name}: expression → count op")
                return result

        # Pattern 2: SUM with CASE scoring
        if "SUM" in expr_upper and "CASE" in expr_upper:
            case_expr, _ = parse_sql_sum_case(expr)
            if case_expr:
                # Extract where condition from outside the CASE
                where = extract_where_from_expression(expr)

                result = {
                    "name": name,
                    "op": "sum",
                    "expr": case_expr,
                }
                if where:
                    result["where"] = where

                logger.debug(f"Transformed {name}: SUM(CASE...) → sum op")
                return result

        # Pattern 3: Simple CASE WHEN for verdict
        if expr_upper.startswith("CASE") and "END" in expr_upper:
            rules = parse_simple_case(expr)
            if rules:
                # Try to extract source from first rule condition
                source = None
                first_when = rules[0].get("when", "") if rules else ""
                source_match = re.match(r"(\w+)\s*[<>=!]", first_when)
                if source_match:
                    source = source_match.group(1)

                result = {
                    "name": name,
                    "op": "case",
                    "rules": rules,
                }
                if source:
                    result["source"] = source
                    # Simplify rules to use relative conditions
                    result["rules"] = _simplify_case_rules(rules, source)

                logger.debug(f"Transformed {name}: CASE → case op")
                return result

        # Pattern 4: Compound CASE expressions (e.g., "(CASE ... END) + (CASE ... END)")
        # This is used for modifier scoring with multiple conditions
        if re.search(r"\(CASE\s+.+?\s+END\s*\)\s*\+", expr, re.IGNORECASE | re.DOTALL):
            # Keep the expression as-is for sum operation
            result = {
                "name": name,
                "op": "sum",
                "expr": expr,  # phase2 evaluates this per-extraction and sums
            }
            logger.debug(f"Transformed {name}: compound CASE → sum op")
            return result

        # Pattern 5: Arithmetic expression (e.g., "BASE_POINTS + MODIFIER_POINTS")
        if re.match(r"^[\w\s+\-*/()]+$", expr):
            result = {
                "name": name,
                "op": "expr",
                "expr": expr,
            }
            logger.debug(f"Transformed {name}: arithmetic → expr op")
            return result

        # Fallback: keep expression but add warning
        logger.warning(f"Could not transform compute operation '{name}': {expr[:50]}...")
        result["_transform_warning"] = "Could not parse expression"

    return result


def _normalize_case_rules(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize verdict labels in case rules.

    Args:
        rules: List of case rules

    Returns:
        Rules with normalized verdict labels
    """
    normalized = []
    for rule in rules:
        new_rule = deepcopy(rule)
        if "then" in new_rule:
            new_rule["then"] = normalize_verdict_label(new_rule["then"])
        if "else" in new_rule:
            new_rule["else"] = normalize_verdict_label(new_rule["else"])
        normalized.append(new_rule)
    return normalized


def _simplify_case_rules(rules: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    """Simplify case rules by removing source variable name from conditions.

    Args:
        rules: List of case rules with conditions like "SCORE >= 8"
        source: Source variable name (e.g., "SCORE")

    Returns:
        Simplified rules with conditions like ">= 8"
    """
    simplified = []
    source_pattern = re.compile(rf"^\s*{re.escape(source)}\s*", re.IGNORECASE)

    for rule in rules:
        new_rule = deepcopy(rule)
        if "when" in new_rule:
            # Remove source variable prefix: "SCORE >= 8" → ">= 8"
            new_rule["when"] = source_pattern.sub("", new_rule["when"]).strip()
        simplified.append(new_rule)

    return simplified


# =============================================================================
# Semantic Fixes for Common LLM Errors
# =============================================================================

def _get_severity_field_and_values(seed: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
    """Get the severity field name and its non-'none' values from the seed.

    Args:
        seed: Formula Seed dict

    Returns:
        Tuple of (field_name, list of severity values excluding 'none')
    """
    for field in seed.get("extract", {}).get("fields", []):
        field_name = field.get("name", "")
        # Look for severity-related field names
        if any(x in field_name.lower() for x in ["severity", "outcome", "level", "intensity"]):
            values = field.get("values", {})
            if isinstance(values, dict):
                # Get all values except 'none' variants
                non_none = [
                    v for v in values.keys()
                    if v.lower() not in ["none", "no", "n/a", "not applicable", "no incident"]
                ]
                if non_none:
                    return field_name, non_none
            elif isinstance(values, list):
                non_none = [
                    v for v in values
                    if str(v).lower() not in ["none", "no", "n/a", "not applicable", "no incident"]
                ]
                if non_none:
                    return None, non_none
    return None, []


def _fix_n_incidents_logic(compute_ops: List[Dict[str, Any]], seed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fix common N_INCIDENTS semantic errors.

    If N_INCIDENTS only filters by ACCOUNT_TYPE, add severity filter.
    This is a defensive backup for when Phase 1 generates incomplete logic.

    Args:
        compute_ops: List of compute operation dicts
        seed: Full Formula Seed dict (for extracting field info)

    Returns:
        Fixed compute operations list
    """
    severity_field, severity_values = _get_severity_field_and_values(seed)

    for op in compute_ops:
        op_name = op.get("name", "")
        op_type = op.get("op", "")

        # Only fix N_INCIDENTS count operations
        if op_name != "N_INCIDENTS" or op_type != "count":
            continue

        where = op.get("where", {})
        if not isinstance(where, dict):
            continue

        # Check if severity filter is missing
        has_severity_filter = False
        for key in where.keys():
            if any(x in key.lower() for x in ["severity", "outcome", "level", "intensity"]):
                has_severity_filter = True
                break

        if not has_severity_filter and severity_field and severity_values:
            # Add the severity filter
            where[severity_field] = severity_values
            op["where"] = where
            logger.info(
                f"[seed_transform] Fixed N_INCIDENTS: added severity filter "
                f"{severity_field} IN {severity_values}"
            )

    return compute_ops


# =============================================================================
# Main Transformation Function
# =============================================================================

def _normalize_extract_field_values(field: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize extract field values to dict format.

    Phase 2 expects values as dict: {"value1": "description1", ...}
    But LLM may generate list: ["value1", "value2", ...]

    Args:
        field: Extract field definition

    Returns:
        Field with values as dict
    """
    result = deepcopy(field)
    values = field.get("values")

    if values is None:
        return result

    # Already dict format - return as-is
    if isinstance(values, dict):
        return result

    # Convert list to dict
    if isinstance(values, list):
        values_dict = {}
        for val in values:
            if isinstance(val, str):
                # Use value as both key and description
                values_dict[val] = val
            elif isinstance(val, dict):
                # Handle {"name": "...", "description": "..."} format
                name = val.get("name", val.get("value", str(val)))
                desc = val.get("description", name)
                values_dict[name] = desc
        result["values"] = values_dict

    return result


def transform_formula_seed(seed: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a Formula Seed to phase2.py compatible format.

    Handles:
    1. Compute operations: expression → op/expr/where format
    2. Verdict labels: "Critical" → "Critical Risk"
    3. Field values normalization (list → dict)
    4. Extract field values normalization

    Args:
        seed: Original Formula Seed from LLM

    Returns:
        Transformed Formula Seed compatible with phase2.py
    """
    result = deepcopy(seed)

    # Get extraction field names for validation
    extraction_fields = []
    for field in seed.get("extract", {}).get("fields", []):
        field_name = field.get("name", "")
        if field_name:
            extraction_fields.append(field_name.upper())
            extraction_fields.append(field_name)

    # Transform extract field values (list → dict)
    if "extract" in result and "fields" in result["extract"]:
        transformed_fields = []
        for field in result["extract"]["fields"]:
            if isinstance(field, dict):
                transformed_fields.append(_normalize_extract_field_values(field))
            else:
                transformed_fields.append(field)
        result["extract"]["fields"] = transformed_fields

    # Transform compute operations
    if "compute" in result:
        transformed_compute = []
        for op_def in result["compute"]:
            if isinstance(op_def, dict):
                transformed = transform_compute_operation(op_def, extraction_fields)
                transformed_compute.append(transformed)
            else:
                logger.warning(f"Skipping non-dict compute entry: {op_def}")

        # Apply semantic fixes (defensive backup for LLM errors)
        transformed_compute = _fix_n_incidents_logic(transformed_compute, result)

        result["compute"] = transformed_compute

    # Normalize verdict labels in verdict_metadata
    if "verdict_metadata" in result:
        meta = result["verdict_metadata"]
        if "verdicts" in meta:
            meta["verdicts"] = [normalize_verdict_label(v) for v in meta["verdicts"]]
        if "severe_verdicts" in meta:
            meta["severe_verdicts"] = [normalize_verdict_label(v) for v in meta["severe_verdicts"]]

    # Normalize verdict labels in search_strategy
    if "search_strategy" in result:
        strategy = result["search_strategy"]
        # Fix early_verdict_expr to use correct labels
        if "early_verdict_expr" in strategy:
            expr = strategy["early_verdict_expr"]
            for abbrev, full in VERDICT_LABEL_MAP.items():
                if abbrev != full:
                    # Replace 'Critical' with 'Critical Risk' in expressions
                    expr = re.sub(rf"'{re.escape(abbrev)}'", f"'{full}'", expr)
                    expr = re.sub(rf'"{re.escape(abbrev)}"', f'"{full}"', expr)
            strategy["early_verdict_expr"] = expr

    return result


def transform_seed_file(input_path: Path, output_path: Optional[Path] = None) -> Dict[str, Any]:
    """Transform a Formula Seed JSON file.

    Args:
        input_path: Path to input seed JSON
        output_path: Optional path to write transformed seed (if None, returns only)

    Returns:
        Transformed seed dict
    """
    with open(input_path) as f:
        seed = json.load(f)

    transformed = transform_formula_seed(seed)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(transformed, f, indent=2)
        logger.info(f"Wrote transformed seed to {output_path}")

    return transformed


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI interface for seed transformation."""
    import argparse

    parser = argparse.ArgumentParser(description="Transform Formula Seed to phase2-compatible format")
    parser.add_argument("input", type=Path, help="Input seed JSON file")
    parser.add_argument("-o", "--output", type=Path, help="Output file (default: overwrite input)")
    parser.add_argument("--dry-run", action="store_true", help="Print transformed seed without writing")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    transformed = transform_seed_file(args.input)

    if args.dry_run:
        print(json.dumps(transformed, indent=2))
    else:
        output_path = args.output or args.input
        with open(output_path, "w") as f:
            json.dump(transformed, f, indent=2)
        print(f"Wrote transformed seed to {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
