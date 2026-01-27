"""Phase 1 Helper Functions.

Utilities for parsing LLM JSON responses, normalization, and validation.
"""

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# JSON Extraction / Parsing
# =============================================================================

def extract_json_from_response(response: str) -> str:
    """Extract JSON object from an LLM response.

    Handles:
    - Markdown code blocks (```json ... ```)
    - Generic code blocks (``` ... ```)
    - Raw JSON with extra text
    """
    response = response.strip()

    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        if end > start:
            return response[start:end].strip()

    if "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end > start:
            return response[start:end].strip()

    # Fallback: take the first JSON object substring
    brace_start = response.find("{")
    brace_end = response.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        return response[brace_start:brace_end + 1].strip()

    return response


def parse_json_safely(json_str: str) -> Dict[str, Any]:
    """Parse JSON string with strict error handling."""
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse error: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")

    return data


# =============================================================================
# Normalization
# =============================================================================

_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")


def normalize_term_name(name: str) -> str:
    """Normalize term name to UPPERCASE_WITH_UNDERSCORES."""
    if not isinstance(name, str):
        return ""
    cleaned = _NON_ALNUM.sub("_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned.upper()


def normalize_terms(terms: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize term list: names + values + descriptions."""
    normalized = []
    for term in terms:
        if not isinstance(term, dict):
            continue
        name = normalize_term_name(term.get("name", ""))
        values = term.get("values", [])
        descriptions = term.get("descriptions") or {}

        # Allow values to be dict (keys as values)
        if isinstance(values, dict):
            values_list = [str(v) for v in values.keys()]
        elif isinstance(values, list):
            values_list = [str(v) for v in values]
        else:
            values_list = []

        # Normalize descriptions to string keys
        if not isinstance(descriptions, dict):
            descriptions = {}
        descriptions_norm = {str(k): str(v) for k, v in descriptions.items()}

        normalized.append({
            "name": name,
            "type": "enum",
            "values": values_list,
            "descriptions": descriptions_norm,
        })
    return normalized


def normalize_verdict_rules(rules: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize rule fields (logic, field names)."""
    normalized = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if rule.get("default"):
            normalized.append({"verdict": rule.get("verdict"), "default": True})
            continue

        logic = str(rule.get("logic") or "ANY").upper()
        if logic not in ("ANY", "ALL"):
            logic = "ANY"

        conditions = []
        for cond in rule.get("conditions", []) or []:
            if not isinstance(cond, dict):
                continue
            field = normalize_term_name(cond.get("field", ""))
            values = cond.get("values", [])
            if isinstance(values, list):
                values_list = [str(v) for v in values]
            elif isinstance(values, dict):
                values_list = [str(v) for v in values.keys()]
            else:
                values_list = []
            min_count = cond.get("min_count", 1)
            try:
                min_count = int(min_count)
            except (TypeError, ValueError):
                min_count = 1

            conditions.append({
                "field": field,
                "values": values_list,
                "min_count": min_count,
            })

        normalized.append({
            "verdict": rule.get("verdict"),
            "logic": logic,
            "conditions": conditions,
        })

    return normalized


# =============================================================================
# Validation
# =============================================================================

def validate_terms(terms: List[Dict[str, Any]]) -> List[str]:
    """Validate term definitions."""
    errors = []
    seen = set()
    for i, term in enumerate(terms):
        name = term.get("name", "")
        if not name:
            errors.append(f"terms[{i}] missing name")
            continue
        name_upper = name.upper()
        if name_upper in seen:
            errors.append(f"duplicate term name: {name}")
        seen.add(name_upper)

        values = term.get("values", [])
        if not isinstance(values, list) or not values:
            errors.append(f"term '{name}' has no values")
        else:
            # Ensure unique values (case-sensitive)
            if len(values) != len(list(dict.fromkeys(values))):
                errors.append(f"term '{name}' has duplicate values")
    return errors


def validate_verdict_spec(
    verdicts: List[str],
    rules: List[Dict[str, Any]],
    terms: List[Dict[str, Any]],
) -> List[str]:
    """Validate verdict labels and rules against terms."""
    errors = []

    if not verdicts:
        errors.append("verdicts list is empty")

    term_map = {t["name"]: set(t.get("values", [])) for t in terms if t.get("name")}

    # Exactly one default rule
    defaults = [r for r in rules if r.get("default")]
    if len(defaults) != 1:
        errors.append(f"expected exactly one default rule, got {len(defaults)}")

    for i, rule in enumerate(rules):
        verdict = rule.get("verdict")
        if not verdict:
            errors.append(f"rules[{i}] missing verdict")
            continue
        if verdicts and verdict not in verdicts:
            errors.append(f"rules[{i}] verdict '{verdict}' not in verdicts list")

        if rule.get("default"):
            if rule.get("conditions"):
                errors.append(f"default rule '{verdict}' must not have conditions")
            continue

        conditions = rule.get("conditions", [])
        if not conditions:
            errors.append(f"rule '{verdict}' has no conditions")
            continue

        for j, cond in enumerate(conditions):
            field = cond.get("field")
            values = cond.get("values", [])
            min_count = cond.get("min_count", 1)

            if not field:
                errors.append(f"rules[{i}].conditions[{j}] missing field")
                continue
            if field not in term_map:
                errors.append(f"rules[{i}].conditions[{j}] field '{field}' not defined in terms")
                continue
            if not values:
                errors.append(f"rules[{i}].conditions[{j}] has no values")
                continue

            term_values = term_map[field]
            for v in values:
                if v not in term_values:
                    errors.append(
                        f"rules[{i}].conditions[{j}] value '{v}' not in term '{field}' values"
                    )

            try:
                if int(min_count) < 1:
                    errors.append(f"rules[{i}].conditions[{j}] min_count must be >= 1")
            except (TypeError, ValueError):
                errors.append(f"rules[{i}].conditions[{j}] min_count must be int")

    return errors


def referenced_fields_from_rules(rules: List[Dict[str, Any]]) -> List[str]:
    """Get unique field names referenced by rules."""
    fields = []
    for rule in rules:
        if rule.get("default"):
            continue
        for cond in rule.get("conditions", []) or []:
            field = cond.get("field")
            if field and field not in fields:
                fields.append(field)
    return fields


# =============================================================================
# Usage Accumulation
# =============================================================================

def accumulate_usage(usages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Accumulate usage metrics from multiple LLM calls."""
    total = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
        "latency_ms": 0.0,
    }
    for u in usages:
        if not u:
            continue
        total["prompt_tokens"] += u.get("prompt_tokens", 0)
        total["completion_tokens"] += u.get("completion_tokens", 0)
        total["cost_usd"] += u.get("cost_usd", 0.0)
        total["latency_ms"] += u.get("latency_ms", 0.0)
    return total
