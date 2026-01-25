"""Phase 1 Helper Functions.

Contains utility functions for Formula Seed generation:
- YAML extraction and parsing
- Usage accumulation
- Minimal structural validation

Full validation functions archived in: backup/phase1_validations.py
"""

import logging
import re
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# YAML Extraction and Parsing
# =============================================================================

def extract_yaml_from_response(response: str) -> str:
    """Extract YAML content from LLM response.

    Handles:
    - Markdown code blocks (```yaml ... ```)
    - Plain YAML text
    - Response with explanations before/after
    """
    response = response.strip()

    # Try to find YAML in markdown code block
    if "```yaml" in response:
        start = response.find("```yaml") + 7
        end = response.find("```", start)
        if end > start:
            return response[start:end].strip()

    # Try generic code block
    if "```" in response:
        start = response.find("```") + 3
        # Skip language identifier if present
        first_newline = response.find("\n", start)
        if first_newline > start:
            start = first_newline + 1
        end = response.find("```", start)
        if end > start:
            return response[start:end].strip()

    # Assume entire response is YAML
    return response


def parse_yaml_safely(yaml_str: str) -> Dict[str, Any]:
    """Parse YAML string with error handling.

    Handles common LLM YAML generation issues:
    1. Apostrophes in single-quoted strings (e.g., 'customer's experience')
    2. Unquoted strings that look like numbers
    3. Mixed quote styles

    Falls back progressively from least to most aggressive fixes.
    """
    # First try: normal parse
    try:
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data).__name__}")
        return data
    except yaml.YAMLError as e:
        original_error = e

    # Second try: Fix common quote issues
    fixed = yaml_str

    # Fix unquoted strings that look like numbers
    fixed = re.sub(r":\s*(\d+\.\d+)(?=\s*$)", r': "\1"', fixed, flags=re.MULTILINE)

    # Fix apostrophes in single-quoted strings by converting to double quotes
    # Pattern: key: 'text with 'inner' quote' -> key: "text with 'inner' quote"
    def fix_quotes_in_line(line: str) -> str:
        if ':' not in line:
            return line

        key, _, value = line.partition(':')
        value = value.strip()

        # Skip if no quotes or empty value
        if not value or (not value.startswith("'") and not value.startswith('"')):
            return line

        # If value has unbalanced single quotes (apostrophe issue), convert to double quotes
        if value.startswith("'"):
            # Count single quotes - if odd or >2, likely has embedded apostrophe
            if value.count("'") != 2 or (value.count("'") == 2 and not value.endswith("'")):
                # Extract content between first quote and try to find ending
                content = value[1:]
                if content.endswith("'"):
                    content = content[:-1]
                # Escape any double quotes in content, preserve single quotes (apostrophes)
                content = content.replace('"', '\\"')
                return f'{key}: "{content}"'

        return line

    fixed_lines = [fix_quotes_in_line(line) for line in fixed.split('\n')]
    fixed = '\n'.join(fixed_lines)

    try:
        data = yaml.safe_load(fixed)
        if isinstance(data, dict):
            return data
    except yaml.YAMLError:
        pass

    # Third try: More aggressive - wrap all string values in double quotes
    def force_double_quotes(line: str) -> str:
        if ':' not in line:
            return line

        # Preserve indentation
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]

        if ':' not in stripped:
            return line

        key, _, value = stripped.partition(':')
        value = value.strip()

        # Skip if empty, already double-quoted, a list item, or a number
        if not value or value.startswith('"') or value.startswith('[') or value.startswith('-'):
            return line
        if re.match(r'^-?\d+(\.\d+)?$', value):
            return line
        if value.lower() in ('true', 'false', 'null', 'none'):
            return line

        # Remove existing quotes and re-quote with double quotes
        if value.startswith("'"):
            # Find the content - handle unbalanced quotes
            if value.endswith("'") and value.count("'") == 2:
                content = value[1:-1]
            else:
                # Unbalanced - take everything after first quote
                content = value[1:].rstrip("'")
        else:
            content = value

        # Escape double quotes in content
        content = content.replace('"', '\\"')
        return f'{indent}{key}: "{content}"'

    forced_lines = [force_double_quotes(line) for line in yaml_str.split('\n')]
    forced = '\n'.join(forced_lines)

    try:
        data = yaml.safe_load(forced)
        if isinstance(data, dict):
            logger.debug("Used double-quote forcing for YAML parsing")
            return data
    except yaml.YAMLError:
        pass

    # Last resort: Simplified parsing - replace problematic values with placeholders
    # This loses information but allows the seed to be generated
    lines_simplified = []
    lost_values = []
    for line in yaml_str.split('\n'):
        if ':' in line and ("'" in line or '"' in line):
            key_part = line.split(':')[0]
            # Keep indentation
            indent = len(key_part) - len(key_part.lstrip())
            key = key_part.strip()
            # Track what we're losing
            value_part = line.split(':', 1)[1].strip() if ':' in line else ''
            if value_part and value_part not in ('', '[]', '{}'):
                lost_values.append(f"{key}: {value_part[:50]}")
            lines_simplified.append(f'{" " * indent}{key}: "value"')
        else:
            lines_simplified.append(line)

    try:
        data = yaml.safe_load('\n'.join(lines_simplified))
        if isinstance(data, dict):
            if lost_values:
                logger.warning(
                    f"Used simplified YAML parsing, {len(lost_values)} descriptions may be lost: "
                    f"{lost_values[:3]}{'...' if len(lost_values) > 3 else ''}"
                )
            return data
    except yaml.YAMLError:
        pass

    raise ValueError(f"YAML parse error: {original_error}")


# =============================================================================
# Usage Accumulation
# =============================================================================

def accumulate_usage(usages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Accumulate usage metrics from multiple LLM calls.

    Args:
        usages: List of usage dicts from LLM calls

    Returns:
        Combined usage dict
    """
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


# =============================================================================
# Validation Functions (Minimal - Structural Only)
# =============================================================================
# Full validation archived in: backup/phase1_validations.py

def validate_formula_seed(seed: Dict[str, Any]) -> List[str]:
    """Minimal structural validation of Formula Seed.

    Only checks for structural issues that would cause Phase 2 to crash:
    - Required top-level keys exist
    - VERDICT operation exists
    - extract.fields is a list

    Semantic checks (enum consistency, field references) are skipped.
    These may cause silent failures but are rare with modern LLMs.

    Args:
        seed: Formula Seed dict

    Returns:
        List of validation errors (empty list if valid)
    """
    errors = []

    # Check required top-level keys
    for key in ["extract", "compute", "output"]:
        if key not in seed:
            errors.append(f"Missing required key: {key}")

    if errors:
        return errors  # Can't continue without required keys

    # Check extract.fields exists and is a list
    if "fields" not in seed["extract"]:
        errors.append("extract missing 'fields'")
    elif not isinstance(seed["extract"]["fields"], list):
        errors.append("extract.fields must be a list")

    # Check compute is a non-empty list with VERDICT operation
    if not isinstance(seed["compute"], list):
        errors.append("compute must be a list")
    elif len(seed["compute"]) == 0:
        errors.append("compute must not be empty")
    else:
        has_verdict = any(
            isinstance(op, dict) and op.get("name") == "VERDICT"
            for op in seed["compute"]
        )
        if not has_verdict:
            errors.append("compute must include a VERDICT operation")

    # Check output is a list
    if not isinstance(seed["output"], list):
        errors.append("output must be a list")

    return errors
