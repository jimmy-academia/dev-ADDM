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
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

logger = logging.getLogger(__name__)


# =============================================================================
# Verdict Label Mapping
# =============================================================================

# Standard verdict labels (GT format) vs common LLM abbreviations
# Maps various LLM-generated labels to canonical GT format
VERDICT_LABEL_MAP = {
    # =========================================================================
    # G1 - Customer Safety (Risk-based)
    # Verdicts: Low Risk, High Risk, Critical Risk
    # =========================================================================
    "critical": "Critical Risk",
    "critical risk": "Critical Risk",
    "Critical": "Critical Risk",
    "Critical Risk": "Critical Risk",
    "CRITICAL": "Critical Risk",
    "CRITICAL_RISK": "Critical Risk",

    "high": "High Risk",
    "high risk": "High Risk",
    "High": "High Risk",
    "High Risk": "High Risk",
    "HIGH": "High Risk",
    "HIGH_RISK": "High Risk",

    "low": "Low Risk",
    "low risk": "Low Risk",
    "Low": "Low Risk",
    "Low Risk": "Low Risk",
    "LOW": "Low Risk",
    "LOW_RISK": "Low Risk",

    # =========================================================================
    # G2 - Customer Experience (Recommendation-based)
    # Verdicts: Not Recommended, Recommended
    # =========================================================================
    "recommended": "Recommended",
    "Recommended": "Recommended",
    "RECOMMENDED": "Recommended",

    "not recommended": "Not Recommended",
    "not_recommended": "Not Recommended",
    "Not Recommended": "Not Recommended",
    "NOT_RECOMMENDED": "Not Recommended",
    "NotRecommended": "Not Recommended",

    # =========================================================================
    # G3 - Customer Value (Value-based)
    # Verdicts: Poor Value, Fair Value, Great Value
    # =========================================================================
    "great value": "Great Value",
    "great_value": "Great Value",
    "Great Value": "Great Value",
    "GREAT_VALUE": "Great Value",
    "excellent value": "Great Value",

    "fair value": "Fair Value",
    "fair_value": "Fair Value",
    "Fair Value": "Fair Value",
    "FAIR_VALUE": "Fair Value",
    "moderate value": "Fair Value",

    "poor value": "Poor Value",
    "poor_value": "Poor Value",
    "Poor Value": "Poor Value",
    "POOR_VALUE": "Poor Value",
    "bad value": "Poor Value",

    # =========================================================================
    # G4 - Owner Operations (Performance-based)
    # Verdicts: Needs Improvement, Satisfactory, Excellent
    # =========================================================================
    "excellent": "Excellent",
    "Excellent": "Excellent",
    "EXCELLENT": "Excellent",

    "satisfactory": "Satisfactory",
    "Satisfactory": "Satisfactory",
    "SATISFACTORY": "Satisfactory",
    "adequate": "Satisfactory",

    "needs improvement": "Needs Improvement",
    "needs_improvement": "Needs Improvement",
    "Needs Improvement": "Needs Improvement",
    "NEEDS_IMPROVEMENT": "Needs Improvement",
    "NeedsImprovement": "Needs Improvement",
    "poor": "Needs Improvement",
    "Poor": "Needs Improvement",

    # =========================================================================
    # G5 - Owner Performance (Performance-based - same as G4)
    # Verdicts: Needs Improvement, Satisfactory, Excellent
    # (Already covered above)
    # =========================================================================

    # =========================================================================
    # G6 - Owner Strategy (Strategy-based - varies by topic)
    # =========================================================================

    # G6 Uniqueness: Generic, Differentiated, Highly Unique
    "highly unique": "Highly Unique",
    "highly_unique": "Highly Unique",
    "Highly Unique": "Highly Unique",
    "HIGHLY_UNIQUE": "Highly Unique",
    "unique": "Highly Unique",

    "differentiated": "Differentiated",
    "Differentiated": "Differentiated",
    "DIFFERENTIATED": "Differentiated",

    "generic": "Generic",
    "Generic": "Generic",
    "GENERIC": "Generic",
    "common": "Generic",

    # G6 Comparison: Weaker, Comparable, Stronger
    "stronger": "Stronger",
    "Stronger": "Stronger",
    "STRONGER": "Stronger",
    "better": "Stronger",

    "comparable": "Comparable",
    "Comparable": "Comparable",
    "COMPARABLE": "Comparable",
    "similar": "Comparable",

    "weaker": "Weaker",
    "Weaker": "Weaker",
    "WEAKER": "Weaker",
    "worse": "Weaker",

    # G6 Loyalty: Low Loyalty, Moderate Loyalty, High Loyalty
    "high loyalty": "High Loyalty",
    "high_loyalty": "High Loyalty",
    "High Loyalty": "High Loyalty",
    "HIGH_LOYALTY": "High Loyalty",
    "strong loyalty": "High Loyalty",

    "moderate loyalty": "Moderate Loyalty",
    "moderate_loyalty": "Moderate Loyalty",
    "Moderate Loyalty": "Moderate Loyalty",
    "MODERATE_LOYALTY": "Moderate Loyalty",
    "medium loyalty": "Moderate Loyalty",

    "low loyalty": "Low Loyalty",
    "low_loyalty": "Low Loyalty",
    "Low Loyalty": "Low Loyalty",
    "LOW_LOYALTY": "Low Loyalty",
    "weak loyalty": "Low Loyalty",
}


def normalize_verdict_label(label: Union[str, Any]) -> Union[str, Any]:
    """Normalize a verdict label to GT format.

    Args:
        label: Raw verdict label from LLM (string, bool, int, or None)

    Returns:
        Normalized verdict label (e.g., "Critical Risk") or original value if not a string
    """
    # Handle non-string types (booleans, numbers, None)
    # This fixes G5 crash where case rules have boolean "then" values
    if not isinstance(label, str):
        return label

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

    # If already has "op" key, validate and fix common issues
    if "op" in op_def:
        # Normalize verdict labels in case rules
        if op_def["op"] == "case" and "rules" in op_def:
            result["rules"] = _normalize_case_rules(op_def["rules"])

        # Fix expr operations that use "source" instead of "expr"
        # LLM sometimes generates: {"op": "expr", "source": "BASE_POINTS + MODIFIER_POINTS"}
        # But phase2._compute_expr expects: {"op": "expr", "expr": "BASE_POINTS + MODIFIER_POINTS"}
        if op_def["op"] == "expr" and "source" in op_def and "expr" not in op_def:
            result["expr"] = op_def["source"]
            del result["source"]
            logger.debug(f"Fixed {name}: renamed 'source' → 'expr' for expr operation")

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

    If N_INCIDENTS only filters by ACCOUNT_TYPE, add outcome/severity filter.
    Uses seed metadata (outcome_field, none_values) if available, falls back to detection.

    Args:
        compute_ops: List of compute operation dicts
        seed: Full Formula Seed dict (for extracting field info)

    Returns:
        Fixed compute operations list
    """
    # Try to get outcome field from seed metadata first (generalizable approach)
    extract = seed.get("extract", {})
    outcome_field = extract.get("outcome_field")
    none_values = extract.get("none_values", [])

    # If outcome_field from metadata, derive non-none values from field definition
    if outcome_field:
        # Get all values for this field and exclude none_values
        for field in extract.get("fields", []):
            if field.get("name") == outcome_field:
                values = field.get("values", {})
                if isinstance(values, dict):
                    all_values = list(values.keys())
                elif isinstance(values, list):
                    all_values = values
                else:
                    all_values = []

                # Filter out none values (case-insensitive)
                none_lower = {v.lower() for v in none_values}
                severity_values = [v for v in all_values if v.lower() not in none_lower]
                severity_field = outcome_field
                break
        else:
            # Field not found, fall back to detection
            outcome_field = None

    # Fall back to detection if no metadata
    if not outcome_field:
        severity_field, severity_values = _get_severity_field_and_values(seed)
    else:
        logger.debug(
            f"[seed_transform] Using seed metadata: outcome_field={outcome_field}, "
            f"none_values={none_values}"
        )

    for op in compute_ops:
        op_name = op.get("name", "")
        op_type = op.get("op", "")

        # Only fix N_INCIDENTS count operations
        if op_name != "N_INCIDENTS" or op_type != "count":
            continue

        where = op.get("where", {})
        if not isinstance(where, dict):
            continue

        # Check if outcome/severity filter is already present
        has_outcome_filter = False
        for key in where.keys():
            key_lower = key.lower()
            # Check for the specific outcome field or common severity-related names
            if outcome_field and key.upper() == outcome_field.upper():
                has_outcome_filter = True
                break
            if any(x in key_lower for x in ["severity", "outcome", "level", "intensity", "quality"]):
                has_outcome_filter = True
                break

        if not has_outcome_filter and severity_field and severity_values:
            # Add the outcome/severity filter
            where[severity_field] = severity_values
            op["where"] = where
            logger.info(
                f"[seed_transform] Fixed N_INCIDENTS: added outcome filter "
                f"{severity_field} IN {severity_values}"
            )

    return compute_ops


# =============================================================================
# VERDICT Operation Generation
# =============================================================================

def _strip_verdict_quotes(compute_ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip extra quotes from VERDICT rule values.

    LLMs sometimes generate verdict values with extra Python-style quotes like:
    {"then": "'Critical Risk'"} instead of {"then": "Critical Risk"}

    This function strips those extra quotes to ensure clean verdict values.

    Args:
        compute_ops: List of compute operation dicts

    Returns:
        Compute operations with cleaned verdict values
    """
    for op in compute_ops:
        if op.get("name", "").upper() == "VERDICT" and op.get("op") == "case":
            rules = op.get("rules", [])
            for rule in rules:
                # Strip quotes from 'then' values
                if "then" in rule and isinstance(rule["then"], str):
                    val = rule["then"]
                    # Strip surrounding single or double quotes
                    if (val.startswith("'") and val.endswith("'")) or \
                       (val.startswith('"') and val.endswith('"')):
                        rule["then"] = val[1:-1]
                        logger.debug(f"[seed_transform] Stripped quotes from VERDICT then: {val} -> {rule['then']}")
                # Strip quotes from 'else' values
                if "else" in rule and isinstance(rule["else"], str):
                    val = rule["else"]
                    if (val.startswith("'") and val.endswith("'")) or \
                       (val.startswith('"') and val.endswith('"')):
                        rule["else"] = val[1:-1]
                        logger.debug(f"[seed_transform] Stripped quotes from VERDICT else: {val} -> {rule['else']}")
            # Also strip quotes from default_verdict if present
            if "default_verdict" in op and isinstance(op["default_verdict"], str):
                val = op["default_verdict"]
                if (val.startswith("'") and val.endswith("'")) or \
                   (val.startswith('"') and val.endswith('"')):
                    op["default_verdict"] = val[1:-1]
    return compute_ops


def _has_verdict_compute(compute_ops: List[Dict[str, Any]]) -> bool:
    """Check if VERDICT compute operation exists.

    Args:
        compute_ops: List of compute operation dicts

    Returns:
        True if VERDICT case operation exists
    """
    for op in compute_ops:
        if op.get("name", "").upper() == "VERDICT" and op.get("op") == "case":
            return True
    return False


def _parse_early_verdict_expr(expr: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Parse early_verdict_expr Python expression into case rules.

    Handles expressions like:
    "'Critical Risk' if (N_SEVERE >= 1) else ('High Risk' if (N_MODERATE >= 1) else 'Low Risk')"

    Args:
        expr: Python ternary expression string

    Returns:
        Tuple of (rules_list, default_verdict)
    """
    rules = []
    default_verdict = None

    # Pattern for nested ternary: 'VERDICT' if CONDITION else REST
    # We'll parse iteratively, extracting each condition
    remaining = expr.strip()

    while remaining:
        # Pattern: 'VERDICT' if (CONDITION) else REST
        # Or: 'VERDICT' if CONDITION else REST
        match = re.match(
            r"['\"]([^'\"]+)['\"]\s+if\s+(?:\()?(.+?)(?:\))?\s+else\s+(.+)",
            remaining,
            re.IGNORECASE | re.DOTALL
        )

        if not match:
            # Check if remaining is just a verdict string (default case)
            final_match = re.match(r"['\"]([^'\"]+)['\"]", remaining.strip())
            if final_match:
                default_verdict = normalize_verdict_label(final_match.group(1))
            break

        verdict_str = match.group(1)
        condition = match.group(2).strip()
        rest = match.group(3).strip()

        # Normalize verdict label
        verdict = normalize_verdict_label(verdict_str)

        # Clean up condition - remove surrounding parens if balanced
        while condition.startswith("(") and condition.endswith(")"):
            # Check if parens are balanced
            depth = 0
            balanced = True
            for i, c in enumerate(condition):
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                if depth == 0 and i < len(condition) - 1:
                    balanced = False
                    break
            if balanced:
                condition = condition[1:-1].strip()
            else:
                break

        rules.append({"when": condition, "then": verdict})

        # Handle nested ternary in rest
        # Rest might be: ('High Risk' if ... else 'Low Risk')
        if rest.startswith("(") and rest.endswith(")"):
            rest = rest[1:-1].strip()

        remaining = rest

    # If no rules were extracted but we have a default, return empty rules with default
    if not rules and default_verdict:
        return [], default_verdict

    return rules, default_verdict


def _generate_verdict_from_early_expr(seed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate VERDICT compute operation from early_verdict_expr.

    If the seed has early_verdict_expr but no VERDICT case operation,
    this creates one from the expression.

    Args:
        seed: Formula Seed dict

    Returns:
        VERDICT compute operation dict, or None if cannot generate
    """
    strategy = seed.get("search_strategy", {})
    early_expr = strategy.get("early_verdict_expr", "")

    if not early_expr:
        return None

    rules, default_verdict = _parse_early_verdict_expr(early_expr)

    if not rules and not default_verdict:
        logger.warning(f"Could not parse early_verdict_expr: {early_expr[:100]}...")
        return None

    # Build VERDICT case operation
    verdict_op = {
        "name": "VERDICT",
        "op": "case",
        "rules": rules,
    }

    if default_verdict:
        verdict_op["default_verdict"] = default_verdict

    logger.info(
        f"[seed_transform] Generated VERDICT operation from early_verdict_expr: "
        f"{len(rules)} rules, default={default_verdict}"
    )

    return verdict_op


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

        # ===== GENERATE VERDICT IF MISSING =====
        # If no VERDICT case operation exists but early_verdict_expr is available,
        # generate VERDICT from the expression. This is a specification enforcement:
        # every seed MUST have a deterministic VERDICT compute operation.
        if not _has_verdict_compute(transformed_compute):
            verdict_op = _generate_verdict_from_early_expr(result)
            if verdict_op:
                transformed_compute.append(verdict_op)
                logger.info("[seed_transform] Added generated VERDICT operation to compute")
            else:
                logger.warning(
                    "[seed_transform] No VERDICT compute operation and could not generate from "
                    "early_verdict_expr. Verdict may be undefined."
                )

        # Strip extra quotes from VERDICT rules (LLM sometimes outputs "'Critical Risk'" instead of "Critical Risk")
        transformed_compute = _strip_verdict_quotes(transformed_compute)

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
