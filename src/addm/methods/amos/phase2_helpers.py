"""Phase 2 Helper Functions: Validation and Field Operations.

Extracted from phase2.py for clarity. These are stateless functions
that operate on extractions and field definitions.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from addm.methods.amos.phase2_prompts import (
    normalize_enum_value,
    validate_enum_strict,
)
from addm.utils.text_validation import validate_multi_span_snippet

logger = logging.getLogger(__name__)


# =============================================================================
# Field Value Helpers (case-insensitive lookups)
# =============================================================================

def get_field_value(extraction: Dict[str, Any], field_name: str) -> Any:
    """Get field value from extraction with case-insensitive lookup.

    Args:
        extraction: Extraction dict with field values
        field_name: Field name to look up

    Returns:
        Field value if found, None otherwise
    """
    # Try exact, lowercase, uppercase
    for key in (field_name, field_name.lower(), field_name.upper()):
        if key in extraction:
            return extraction[key]
    # Case-insensitive search
    lower_name = field_name.lower()
    for key in extraction:
        if key.lower() == lower_name:
            return extraction[key]
    return None


def find_extraction_key(extraction: Dict[str, Any], field_name: str) -> Optional[str]:
    """Find the actual key used in extraction for a field (case-insensitive).

    Args:
        extraction: Extraction dict
        field_name: Field name to look up

    Returns:
        The actual key in extraction that matches, or None
    """
    for key in (field_name, field_name.lower(), field_name.upper()):
        if key in extraction:
            return key
    lower_name = field_name.lower()
    for key in extraction:
        if key.lower() == lower_name:
            return key
    return None


def fuzzy_enum_match(actual: str, expected: str) -> bool:
    """Fuzzy match enum values to handle label inconsistencies.

    Handles: "moderate incident" vs "Moderate", "Severe incident" vs "Severe"

    Args:
        actual: The actual value from extraction
        expected: The expected value from compute.where

    Returns:
        True if values match (fuzzy)
    """
    if actual is None or expected is None:
        return actual == expected

    a, e = str(actual).lower().strip(), str(expected).lower().strip()

    # Exact match
    if a == e:
        return True

    # Partial containment
    if e in a or a in e:
        return True

    # Strip common suffixes
    for suffix in (' incident', ' reaction', ' risk', ' level', ' quality'):
        a = a.replace(suffix, '')
        e = e.replace(suffix, '')

    return a == e


# =============================================================================
# Validation Functions
# =============================================================================

def validate_snippet(
    extraction: Dict[str, Any],
    quote: str,
    review_text: str,
    stats: Dict[str, int],
) -> Dict[str, Any]:
    """Validate that the supporting quote exists in the review text.

    Args:
        extraction: Extraction result dict
        quote: The supporting_quote from extraction
        review_text: The original review text
        stats: Stats dict to update (mutated in place)

    Returns:
        Updated extraction dict (may have is_relevant=False if validation fails)
    """
    stats["total_relevant"] += 1

    if not quote or not quote.strip():
        extraction["is_relevant"] = False
        extraction["_rejected_reason"] = "no_quote_provided"
        stats["rejected_no_quote"] += 1
        return extraction

    validation = validate_multi_span_snippet(quote, review_text)

    if validation["valid"]:
        extraction["_snippet"] = quote
        extraction["_snippet_validated"] = True
        extraction["_snippet_match_type"] = validation["match_type"]
        if validation["match_type"] in ("multi_span", "word_overlap"):
            extraction["_snippet_match_quality"] = f"{validation['match_ratio']:.0%}"
        stats["valid_quotes"] += 1
    else:
        extraction["is_relevant"] = False
        extraction["_rejected_reason"] = "quote_not_found"
        extraction["_attempted_quote"] = quote[:100]
        stats["rejected_quote_not_found"] += 1

    return extraction


def validate_enum_fields(
    extraction: Dict[str, Any],
    fields: List[Dict[str, Any]],
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate enum fields in extraction against expected values.

    Args:
        extraction: Extraction dict with field values
        fields: Field definitions from Formula Seed
        stats: Enum validation stats dict (mutated in place)

    Returns:
        Extraction with validation errors added if any fields invalid
    """
    if not extraction.get("is_relevant", False):
        return extraction

    validation_errors = extraction.get("_validation_errors", [])

    for field in fields:
        field_name = field.get("name", "")
        field_type = field.get("type", "")
        values = field.get("values", {})

        if field_type != "enum" or not values:
            continue

        actual = get_field_value(extraction, field_name)
        if actual is None:
            continue

        expected_values = list(values.keys())
        is_valid, error_msg, normalized = validate_enum_strict(
            actual, expected_values, field_name
        )

        stats["total_validated"] += 1

        if is_valid:
            stats["valid"] += 1
            # Store normalized value, remove original if different case
            original_key = find_extraction_key(extraction, field_name)
            if original_key and original_key != field_name.lower():
                del extraction[original_key]
            extraction[field_name.lower()] = normalized
        else:
            stats["rejected"] += 1
            stats["rejection_details"].append({
                "review_id": extraction.get("review_id", "unknown"),
                "field": field_name,
                "value": actual,
                "expected": expected_values,
                "error": error_msg,
            })
            validation_errors.append(error_msg)

    if validation_errors:
        extraction["_validation_errors"] = validation_errors

    return extraction


# =============================================================================
# Outcome Field Detection
# =============================================================================

def get_outcome_field_info(seed: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
    """Get outcome field name and none values from seed metadata.

    Args:
        seed: Formula Seed dict

    Returns:
        Tuple of (outcome_field_name, list_of_none_values)
    """
    extract = seed.get("extract", {})
    outcome_field = extract.get("outcome_field")
    none_values = extract.get("none_values", ["none", "n/a"])
    return outcome_field, none_values


def is_none_value(value: Any, none_values: List[str]) -> bool:
    """Check if a value represents "no incident".

    Args:
        value: The value to check
        none_values: List of values that mean "no incident"

    Returns:
        True if value represents "no incident"
    """
    if value is None:
        return True
    value_str = str(value).strip().lower()
    none_values_lower = {nv.lower() for nv in none_values}
    return value_str in none_values_lower


def get_enum_values_for_field(seed: Dict[str, Any], field_name: str) -> Optional[List[str]]:
    """Get expected enum values for a field from the Formula Seed.

    Args:
        seed: Formula Seed dict
        field_name: Name of the field

    Returns:
        List of valid enum values if field is an enum, None otherwise
    """
    fields = seed.get("extract", {}).get("fields", [])
    for field in fields:
        if field.get("name") == field_name and field.get("type") == "enum":
            values = field.get("values", {})
            if values:
                return list(values.keys())
    return None


def normalize_actual_value(seed: Dict[str, Any], field_name: str, actual: Any) -> Any:
    """Normalize an actual value from extraction for comparison.

    Args:
        seed: Formula Seed dict
        field_name: Name of the field
        actual: The actual value from extraction

    Returns:
        Normalized value if field is an enum, otherwise original value
    """
    expected_values = get_enum_values_for_field(seed, field_name)
    if expected_values:
        return normalize_enum_value(actual, expected_values)
    return actual


# =============================================================================
# Condition Matching
# =============================================================================

def matches_condition(
    extraction: Dict[str, Any],
    condition: Dict[str, Any],
    seed: Dict[str, Any],
) -> bool:
    """Check if an extraction matches a where condition.

    Supports:
    - Simple: {"field": "value"}
    - List: {"field": ["value1", "value2"]}
    - AND: {"and": [{...}, ...]}
    - OR: {"or": [{...}, ...]}
    - Field/equals: {"field": "X", "equals": "Y"}
    - Not equals: {"field": "X", "not_equals": "Y"}

    Args:
        extraction: Single extraction result
        condition: Condition dict to match
        seed: Formula Seed (for enum normalization)

    Returns:
        True if all conditions match
    """
    # Handle field/equals at top level
    if "field" in condition and ("equals" in condition or "not_equals" in condition):
        field = condition["field"]
        actual = get_field_value(extraction, field)
        actual = normalize_actual_value(seed, field, actual)

        if "equals" in condition:
            expected = condition["equals"]
            if isinstance(expected, list):
                if isinstance(actual, str):
                    if not any(
                        fuzzy_enum_match(actual, e) if isinstance(e, str) else actual == e
                        for e in expected
                    ):
                        return False
                elif actual not in expected:
                    return False
            else:
                if isinstance(actual, str) and isinstance(expected, str):
                    if not fuzzy_enum_match(actual, expected):
                        return False
                elif actual != expected:
                    return False

        if "not_equals" in condition:
            not_expected = condition["not_equals"]
            if isinstance(not_expected, list):
                if isinstance(actual, str):
                    if any(
                        fuzzy_enum_match(actual, e) if isinstance(e, str) else actual == e
                        for e in not_expected
                    ):
                        return False
                elif actual in not_expected:
                    return False
            else:
                if isinstance(actual, str) and isinstance(not_expected, str):
                    if fuzzy_enum_match(actual, not_expected):
                        return False
                elif actual == not_expected:
                    return False

    # Handle AND conditions
    if "and" in condition:
        for sub_cond in condition["and"]:
            if not matches_condition(extraction, sub_cond, seed):
                return False

    # Handle OR conditions
    if "or" in condition:
        if not any(matches_condition(extraction, sub_cond, seed) for sub_cond in condition["or"]):
            return False

    # Simple key-value matching
    special_keys = {"field", "equals", "not_equals", "and", "or"}
    for field, expected in condition.items():
        if field in special_keys:
            continue

        actual = get_field_value(extraction, field)
        actual = normalize_actual_value(seed, field, actual)

        if isinstance(actual, str) and isinstance(expected, str):
            if not fuzzy_enum_match(actual, expected):
                return False
        elif isinstance(expected, list):
            if isinstance(actual, str):
                if not any(
                    fuzzy_enum_match(actual, e) if isinstance(e, str) else actual == e
                    for e in expected
                ):
                    return False
            elif actual not in expected:
                return False
        elif actual != expected:
            return False

    return True
