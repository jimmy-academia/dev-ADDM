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
# Normalization helpers
# =============================================================================

_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    return []


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def normalize_term_name(name: str) -> str:
    """Normalize term name to UPPERCASE_WITH_UNDERSCORES."""
    if not isinstance(name, str):
        return ""
    cleaned = _NON_ALNUM.sub("_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned.upper()


# =============================================================================
# Segment blocks
# =============================================================================

def normalize_segment_blocks(data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    definitions = (
        data.get("definitions_blocks")
        or data.get("definitions_block")
        or data.get("terms_block")
        or []
    )
    verdicts = (
        data.get("verdict_rules_blocks")
        or data.get("verdict_rules_block")
        or data.get("verdicts_block")
        or []
    )
    return _as_list(definitions), _as_list(verdicts)


def validate_segment_blocks(definitions: List[str], verdicts: List[str]) -> List[str]:
    errors = []
    if not definitions:
        errors.append("definitions_blocks is empty")
    if not verdicts:
        errors.append("verdict_rules_blocks is empty")
    return errors


# =============================================================================
# Verdict labels
# =============================================================================

def normalize_verdict_labels(data: Dict[str, Any]) -> Tuple[List[str], str]:
    verdicts = _as_list(data.get("verdicts") or data.get("verdict_labels"))
    default_verdict = str(
        data.get("default_verdict")
        or data.get("default")
        or ""
    ).strip()
    return verdicts, default_verdict


def validate_verdict_labels(verdicts: List[str], default_verdict: str) -> List[str]:
    errors = []
    if not verdicts:
        errors.append("verdicts list is empty")
    if default_verdict and default_verdict not in verdicts:
        errors.append("default_verdict not in verdicts list")
    if not default_verdict:
        errors.append("default_verdict missing")
    if len(verdicts) != len(dedupe_preserve_order(verdicts)):
        errors.append("verdicts list contains duplicates")
    return errors


# =============================================================================
# Verdict outline (per verdict)
# =============================================================================

def normalize_verdict_outline(
    data: Dict[str, Any],
    target_verdict: str,
    default_verdict: str,
) -> Dict[str, Any]:
    verdict = str(data.get("verdict") or target_verdict).strip()
    if verdict == default_verdict or data.get("default"):
        return {
            "verdict": target_verdict,
            "default": True,
            "hint_texts": _as_list(data.get("hint_texts")),
        }

    connective = str(data.get("connective") or data.get("logic") or "").upper().strip()
    clause_texts = _as_list(data.get("clause_texts"))
    return {
        "verdict": verdict,
        "connective": connective,
        "clause_texts": clause_texts,
    }


def validate_verdict_outline(
    outline: Dict[str, Any],
    target_verdict: str,
    default_verdict: str,
) -> List[str]:
    errors = []
    verdict = outline.get("verdict")
    if verdict != target_verdict:
        errors.append("verdict label does not match target_verdict")

    if outline.get("default"):
        # Default verdict must not include clauses
        if outline.get("clause_texts"):
            errors.append("default verdict must not include clause_texts")
        return errors

    connective = outline.get("connective")
    if connective not in ("ANY", "ALL"):
        errors.append("connective must be ANY or ALL")

    clause_texts = outline.get("clause_texts", [])
    if not isinstance(clause_texts, list) or not clause_texts:
        errors.append("clause_texts is empty")

    return errors


# =============================================================================
# Required terms
# =============================================================================

def normalize_required_terms(data: Dict[str, Any]) -> List[str]:
    titles = _as_list(
        data.get("required_term_titles")
        or data.get("required_terms")
        or data.get("terms")
    )
    return dedupe_preserve_order(titles)


def validate_required_terms(required: List[str], definitions_text: str) -> List[str]:
    errors = []
    if not required:
        errors.append("required_term_titles is empty")
        return errors
    for title in required:
        if title not in definitions_text:
            errors.append(f"term title not found in definitions: {title}")
    return errors


# =============================================================================
# Term definition spec
# =============================================================================

def normalize_term_definition(
    data: Dict[str, Any],
    requested_title: str,
) -> Dict[str, Any]:
    title = str(data.get("term_title") or data.get("title") or requested_title).strip()
    values = _as_list(data.get("values"))
    descriptions = data.get("descriptions") or {}
    if not isinstance(descriptions, dict):
        descriptions = {}
    descriptions = {str(k): str(v) for k, v in descriptions.items()}

    return {
        "title": title,
        "field": normalize_term_name(title),
        "type": "enum",
        "values": values,
        "descriptions": descriptions,
    }


def validate_term_definition(term: Dict[str, Any], requested_title: str) -> List[str]:
    errors = []
    title = term.get("title", "")
    if title != requested_title:
        errors.append("term_title does not match requested title")

    values = term.get("values", [])
    if not isinstance(values, list) or not values:
        errors.append("values must be a non-empty list")
    elif len(values) != len(dedupe_preserve_order(values)):
        errors.append("values contains duplicates")

    descriptions = term.get("descriptions", {})
    if not isinstance(descriptions, dict):
        errors.append("descriptions must be a dict")
    else:
        for k in descriptions.keys():
            if k not in values:
                errors.append(f"description key '{k}' not in values")

    return errors


# =============================================================================
# Clause spec (per clause)
# =============================================================================

def normalize_clause_spec(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        min_count = int(data.get("min_count", 1))
    except (TypeError, ValueError):
        min_count = 1

    conditions = []
    for cond in data.get("conditions", []) or []:
        if not isinstance(cond, dict):
            continue
        field = normalize_term_name(cond.get("field", ""))
        values = _as_list(cond.get("values"))
        conditions.append({"field": field, "values": values})

    return {"min_count": min_count, "conditions": conditions}


def validate_clause_spec(
    clause: Dict[str, Any],
    term_values_map: Dict[str, List[str]],
) -> List[str]:
    errors = []
    min_count = clause.get("min_count")
    if not isinstance(min_count, int) or min_count < 1:
        errors.append("min_count must be int >= 1")

    conditions = clause.get("conditions", [])
    if not isinstance(conditions, list) or not conditions:
        errors.append("conditions must be a non-empty list")
        return errors

    for idx, cond in enumerate(conditions):
        field = cond.get("field")
        if not field:
            errors.append(f"conditions[{idx}] missing field")
            continue
        if field not in term_values_map:
            errors.append(f"conditions[{idx}] field '{field}' not in terms")
            continue
        values = cond.get("values", [])
        if not values:
            errors.append(f"conditions[{idx}] values empty")
            continue
        allowed = set(term_values_map[field])
        for v in values:
            if v not in allowed:
                errors.append(
                    f"conditions[{idx}] value '{v}' not in {field} values"
                )
    return errors


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
