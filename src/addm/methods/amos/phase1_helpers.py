"""Phase 1 Helper Functions.

Utilities for parsing LLM JSON responses, normalization, slicing, and validation.
"""

import json
import re
from typing import Any, Dict, Iterable, List, Tuple


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
# Basic helpers
# =============================================================================

_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return []


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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


def normalize_value_id(value: str) -> str:
    """Normalize enum value to lower_snake_case."""
    if not isinstance(value, str):
        return ""
    cleaned = _NON_ALNUM.sub("_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned.lower()


def slice_block_by_anchors(agenda: str, start_quote: str, end_quote: str) -> str:
    start_idx = agenda.find(start_quote)
    if start_idx < 0:
        raise ValueError("start_quote not found in agenda")
    end_idx = agenda.find(end_quote, start_idx)
    if end_idx < 0:
        raise ValueError("end_quote not found in agenda")
    if end_idx < start_idx:
        raise ValueError("end_quote occurs before start_quote")
    return agenda[start_idx:end_idx + len(end_quote)]


# =============================================================================
# Step 0: Locate blocks (anchors)
# =============================================================================

def normalize_block_anchors(
    data: Dict[str, Any],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    definitions = _as_list(data.get("definitions_blocks"))
    verdicts = _as_list(data.get("verdict_blocks"))

    def _clean(blocks: List[Any]) -> List[Dict[str, str]]:
        output: List[Dict[str, str]] = []
        for item in blocks:
            if not isinstance(item, dict):
                continue
            start_quote = _as_str(item.get("start_quote"))
            end_quote = _as_str(item.get("end_quote"))
            output.append({"start_quote": start_quote, "end_quote": end_quote})
        return output

    return _clean(definitions), _clean(verdicts)


def validate_block_anchors(
    definitions: List[Dict[str, str]],
    verdicts: List[Dict[str, str]],
    agenda: str,
) -> List[str]:
    errors: List[str] = []
    if not definitions:
        errors.append("definitions_blocks is empty")
    if not verdicts:
        errors.append("verdict_blocks is empty")

    for group_name, blocks in (("definitions_blocks", definitions),
                               ("verdict_blocks", verdicts)):
        for idx, blk in enumerate(blocks):
            start = blk.get("start_quote", "")
            end = blk.get("end_quote", "")
            if not start:
                errors.append(f"{group_name}[{idx}].start_quote is empty")
                continue
            if not end:
                errors.append(f"{group_name}[{idx}].end_quote is empty")
                continue
            if start not in agenda:
                errors.append(f"{group_name}[{idx}].start_quote not in agenda")
                continue
            if end not in agenda:
                errors.append(f"{group_name}[{idx}].end_quote not in agenda")
                continue
            if agenda.count(start) != 1:
                errors.append(f"{group_name}[{idx}].start_quote is not unique")
            if agenda.count(end) != 1:
                errors.append(f"{group_name}[{idx}].end_quote is not unique")
            start_idx = agenda.find(start)
            end_idx = agenda.find(end, start_idx)
            if end_idx < start_idx:
                errors.append(f"{group_name}[{idx}].end_quote before start_quote")
    return errors


# =============================================================================
# Step 1: Verdict rules skeleton
# =============================================================================

def normalize_verdict_rules(data: Dict[str, Any]) -> Dict[str, Any]:
    labels = [_as_str(v) for v in _as_list(data.get("labels")) if _as_str(v)]
    default_label = _as_str(data.get("default_label"))
    order = [_as_str(v) for v in _as_list(data.get("order")) if _as_str(v)]
    rules = _as_list(data.get("rules"))
    normalized_rules: List[Dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        normalized = {
            "label": _as_str(rule.get("label")),
            "default": bool(rule.get("default")),
            "connective": _as_str(rule.get("connective")).upper(),
            "connective_quote": _as_str(rule.get("connective_quote")),
            "clause_quotes": [
                _as_str(v)
                for v in _as_list(rule.get("clause_quotes"))
                if _as_str(v)
            ],
            "default_quote": _as_str(rule.get("default_quote")),
            "hints": [
                {"hint_quote": _as_str(h.get("hint_quote"))}
                for h in _as_list(rule.get("hints"))
                if isinstance(h, dict) and _as_str(h.get("hint_quote"))
            ],
        }
        normalized_rules.append(normalized)
    return {
        "labels": labels,
        "default_label": default_label,
        "order": order,
        "rules": normalized_rules,
    }


def validate_verdict_rules(
    verdict_rules: Dict[str, Any],
    agenda: str,
) -> List[str]:
    errors: List[str] = []
    labels = verdict_rules.get("labels", [])
    default_label = verdict_rules.get("default_label", "")
    order = verdict_rules.get("order", [])
    rules = verdict_rules.get("rules", [])

    if not labels:
        errors.append("labels is empty")
    if len(labels) != len(dedupe_preserve_order(labels)):
        errors.append("labels contains duplicates")
    if not default_label:
        errors.append("default_label is missing")
    if default_label and default_label not in labels:
        errors.append("default_label not in labels")
    if not order:
        errors.append("order is empty")
    if order and set(order) != set(labels):
        errors.append("order must be a permutation of labels")
    if order and default_label and order[-1] != default_label:
        errors.append("order must end with default_label")

    if not rules:
        errors.append("rules is empty")
        return errors

    label_set = set(labels)
    default_rules = [r for r in rules if r.get("default")]
    if len(default_rules) != 1:
        errors.append("rules must contain exactly one default rule")
    if default_rules and default_label and default_rules[0].get("label") != default_label:
        errors.append("default rule label must match default_label")

    seen_labels = set()
    for idx, rule in enumerate(rules):
        label = rule.get("label")
        if not label:
            errors.append(f"rules[{idx}].label is empty")
            continue
        if label not in label_set:
            errors.append(f"rules[{idx}].label not in labels")
        if label in seen_labels:
            errors.append(f"rules[{idx}].label duplicated")
        seen_labels.add(label)
        if label not in agenda:
            errors.append(f"rules[{idx}].label not found in agenda")

        if rule.get("default"):
            default_quote = rule.get("default_quote", "")
            if not default_quote:
                errors.append("default_rule.default_quote is missing")
            elif default_quote not in agenda:
                errors.append("default_rule.default_quote not in agenda")
            for h_idx, hint in enumerate(rule.get("hints", [])):
                hint_quote = hint.get("hint_quote", "")
                if not hint_quote:
                    errors.append(f"default_rule.hints[{h_idx}] is empty")
                elif hint_quote not in agenda:
                    errors.append(f"default_rule.hints[{h_idx}] not in agenda")
            if rule.get("clause_quotes"):
                errors.append("default rule must not include clause_quotes")
            if rule.get("connective"):
                errors.append("default rule must not include connective")
        else:
            connective = rule.get("connective", "")
            if connective not in ("ANY", "ALL"):
                errors.append(f"rules[{idx}].connective must be ANY or ALL")
            connective_quote = rule.get("connective_quote", "")
            if not connective_quote:
                errors.append(f"rules[{idx}].connective_quote is missing")
            elif connective_quote not in agenda:
                errors.append(f"rules[{idx}].connective_quote not in agenda")
            clause_quotes = rule.get("clause_quotes", [])
            if not clause_quotes:
                errors.append(f"rules[{idx}].clause_quotes is empty")
            for c_idx, quote in enumerate(clause_quotes):
                if not quote:
                    errors.append(f"rules[{idx}].clause_quotes[{c_idx}] is empty")
                elif quote not in agenda:
                    errors.append(f"rules[{idx}].clause_quotes[{c_idx}] not in agenda")

    if labels and seen_labels != set(labels):
        errors.append("rules must cover all labels")
    return errors


# =============================================================================
# Step 2: Term blocks
# =============================================================================

def normalize_term_blocks(data: Dict[str, Any]) -> List[Dict[str, str]]:
    terms = _as_list(data.get("terms"))
    output: List[Dict[str, str]] = []
    for item in terms:
        if not isinstance(item, dict):
            continue
        term_title = _as_str(item.get("term_title"))
        start_quote = _as_str(item.get("start_quote"))
        end_quote = _as_str(item.get("end_quote"))
        output.append({
            "term_title": term_title,
            "start_quote": start_quote,
            "end_quote": end_quote,
        })
    return output


def validate_term_blocks(
    terms: List[Dict[str, str]],
    agenda: str,
    definitions_text: str,
    allow_empty: bool,
) -> List[str]:
    errors: List[str] = []
    if not terms and not allow_empty:
        errors.append("terms is empty")
        return errors

    seen_titles = set()
    for idx, term in enumerate(terms):
        title = term.get("term_title", "")
        start = term.get("start_quote", "")
        end = term.get("end_quote", "")
        if not title:
            errors.append(f"terms[{idx}].term_title is empty")
        elif title in seen_titles:
            errors.append(f"terms[{idx}].term_title is duplicate")
        else:
            seen_titles.add(title)
        if not start:
            errors.append(f"terms[{idx}].start_quote is empty")
            continue
        if not end:
            errors.append(f"terms[{idx}].end_quote is empty")
            continue
        if start not in agenda:
            errors.append(f"terms[{idx}].start_quote not in agenda")
            continue
        if end not in agenda:
            errors.append(f"terms[{idx}].end_quote not in agenda")
            continue
        if start not in definitions_text:
            errors.append(f"terms[{idx}].start_quote not in definitions_text")
        if end not in definitions_text:
            errors.append(f"terms[{idx}].end_quote not in definitions_text")
        if agenda.count(start) != 1:
            errors.append(f"terms[{idx}].start_quote is not unique")
        if agenda.count(end) != 1:
            errors.append(f"terms[{idx}].end_quote is not unique")
        start_idx = agenda.find(start)
        end_idx = agenda.find(end, start_idx)
        if end_idx < start_idx:
            errors.append(f"terms[{idx}].end_quote before start_quote")
            continue
        term_block = agenda[start_idx:end_idx + len(end)]
        if len(term_block) < 80:
            errors.append(f"terms[{idx}].block too short")
        if title and title not in term_block:
            errors.append(f"terms[{idx}].term_title not in block")
        elif title:
            head = term_block.lstrip()[:200]
            if title not in head:
                errors.append(f"terms[{idx}].term_title not near start of block")
    return errors


# =============================================================================
# Step 3: Term enum extraction
# =============================================================================

def normalize_term_enum(
    data: Dict[str, Any],
    requested_title: str,
) -> Dict[str, Any]:
    term_title = _as_str(data.get("term_title") or requested_title)
    values = _as_list(data.get("values"))
    normalized_values: List[Dict[str, str]] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        source_value = _as_str(item.get("source_value"))
        description = _as_str(item.get("description"))
        value_quote = _as_str(item.get("value_quote"))
        normalized_values.append({
            "value_id": normalize_value_id(source_value),
            "source_value": source_value,
            "description": description,
            "value_quote": value_quote,
        })
    return {
        "term_title": term_title,
        "field_id": normalize_term_name(term_title),
        "type": "enum",
        "values": normalized_values,
    }


def validate_term_enum(
    term: Dict[str, Any],
    requested_title: str,
    term_block: str,
) -> List[str]:
    errors: List[str] = []
    term_title = term.get("term_title", "")
    if term_title != requested_title:
        errors.append("term_title does not match requested title")

    values = term.get("values", [])
    if not isinstance(values, list) or not values:
        errors.append("values must be a non-empty list")
        return errors
    if len(values) < 2:
        errors.append("values must contain at least 2 options")

    source_values = []
    value_ids = []
    for idx, val in enumerate(values):
        if not isinstance(val, dict):
            errors.append(f"values[{idx}] must be an object")
            continue
        source_value = val.get("source_value", "")
        description = val.get("description", "")
        value_quote = val.get("value_quote", "")
        if not source_value:
            errors.append(f"values[{idx}].source_value is empty")
        elif source_value not in term_block:
            errors.append(f"values[{idx}].source_value not in term_block")
        if not description:
            errors.append(f"values[{idx}].description is empty")
        if not value_quote:
            errors.append(f"values[{idx}].value_quote is empty")
        else:
            if value_quote not in term_block:
                errors.append(f"values[{idx}].value_quote not in term_block")
            if source_value and source_value not in value_quote:
                errors.append(f"values[{idx}].value_quote must contain source_value")
        source_values.append(source_value)
        value_ids.append(val.get("value_id", ""))

    if len(source_values) != len(dedupe_preserve_order(source_values)):
        errors.append("values contains duplicate source_value")
    if len(value_ids) != len(dedupe_preserve_order(value_ids)):
        errors.append("values contains duplicate value_id")

    return errors


# =============================================================================
# Step 4: Clause compilation
# =============================================================================

def normalize_clause_spec(data: Dict[str, Any]) -> Dict[str, Any]:
    min_count = data.get("min_count")
    logic = _as_str(data.get("logic")).upper()
    conditions_raw = _as_list(data.get("conditions"))
    conditions: List[Dict[str, Any]] = []
    for item in conditions_raw:
        if not isinstance(item, dict):
            continue
        field_id = _as_str(item.get("field_id"))
        values = [_as_str(v) for v in _as_list(item.get("values")) if _as_str(v)]
        conditions.append({"field_id": field_id, "values": values})
    return {
        "min_count": min_count,
        "logic": logic,
        "conditions": conditions,
    }


def validate_clause_spec(
    clause: Dict[str, Any],
    allowed_term_values: Dict[str, List[str]],
) -> List[str]:
    errors: List[str] = []
    min_count = clause.get("min_count")
    if not isinstance(min_count, int) or min_count < 1:
        errors.append("min_count must be int >= 1")

    logic = clause.get("logic")
    if logic not in ("ANY", "ALL"):
        errors.append("logic must be ANY or ALL")

    conditions = clause.get("conditions", [])
    if not isinstance(conditions, list) or not conditions:
        errors.append("conditions must be a non-empty list")
        return errors

    for idx, cond in enumerate(conditions):
        field_id = cond.get("field_id")
        if not field_id:
            errors.append(f"conditions[{idx}].field_id is empty")
            continue
        if field_id not in allowed_term_values:
            errors.append(f"conditions[{idx}].field_id not in allowed terms")
            continue
        values = cond.get("values", [])
        if not values:
            errors.append(f"conditions[{idx}].values is empty")
            continue
        allowed = set(allowed_term_values[field_id])
        for v in values:
            if v not in allowed:
                errors.append(
                    f"conditions[{idx}].value '{v}' not in {field_id} values"
                )
    return errors


# =============================================================================
# Usage accumulation
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
