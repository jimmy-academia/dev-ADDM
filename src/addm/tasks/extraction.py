"""Judgment extraction and caching."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from addm.tasks.prompt_parser import get_l0_field_order


class JudgmentCache:
    """
    Unified cache for extracted judgments.

    Structure: {"task_id": {"review_id": {...judgment...}}}
    """

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self._data: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                self._data = json.load(f)

    def save(self) -> None:
        """Save cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._data, f, indent=2)

    def get(self, task_id: str, review_id: str) -> Optional[Dict[str, Any]]:
        """Get cached judgment for a review."""
        return self._data.get(task_id, {}).get(review_id)

    def set(self, task_id: str, review_id: str, judgment: Dict[str, Any]) -> None:
        """Cache a judgment."""
        if task_id not in self._data:
            self._data[task_id] = {}
        self._data[task_id][review_id] = judgment

    def has(self, task_id: str, review_id: str) -> bool:
        """Check if a judgment is cached."""
        return task_id in self._data and review_id in self._data[task_id]

    def get_all(self, task_id: str) -> Dict[str, Dict[str, Any]]:
        """Get all cached judgments for a task."""
        return self._data.get(task_id, {})

    def count(self, task_id: str) -> int:
        """Count cached judgments for a task."""
        return len(self._data.get(task_id, {}))


def build_extraction_prompt(
    l0_schema: Dict[str, Dict[str, str]],
    review_text: str,
    review_id: str,
    task_description: str = "allergy-related information",
) -> str:
    """
    Build an extraction prompt from L0 schema and review text.

    Args:
        l0_schema: Parsed L0 definitions from prompt
        review_text: The review text to analyze
        review_id: Review identifier for output
        task_description: What to analyze the review for (e.g., "allergy-related information")

    Returns:
        Prompt string for LLM extraction
    """
    # Build field definitions
    field_defs = []
    fields = get_l0_field_order(l0_schema)

    for field in fields:
        values = l0_schema[field]
        field_upper = field.upper()
        value_list = ", ".join(values.keys())
        field_defs.append(f"{field_upper} // one of {{{value_list}}}")

        for value, description in values.items():
            field_defs.append(f"  - {value}: {description}")
        field_defs.append("")

    field_definitions = "\n".join(field_defs)

    # Build output format
    output_fields = ", ".join(f'"{f}": "<value>"' for f in fields)

    prompt = f"""Analyze this review for {task_description}.

FIELD DEFINITIONS:
{field_definitions}

REVIEW TEXT:
{review_text}

If this review contains relevant content, extract the L0 primitives.
If no relevant content, output: {{"is_allergy_related": false}}

Output JSON only:
{{
  "review_id": "{review_id}",
  "is_allergy_related": true,
  {output_fields}
}}"""

    return prompt


def validate_judgment(
    judgment: Dict[str, Any],
    l0_schema: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    """
    Validate and normalize L0 field values.

    If LLM returns an invalid value (e.g., "critical" instead of "severe"),
    replace it with the first valid value (usually "none" or safest option).

    Args:
        judgment: Raw judgment from LLM
        l0_schema: Valid values for each L0 field

    Returns:
        Validated judgment with normalized field values
    """
    if not judgment.get("is_allergy_related", False):
        return judgment

    validated = dict(judgment)
    for field, valid_values in l0_schema.items():
        value = judgment.get(field, "")
        if value not in valid_values:
            # Default to first value (usually "none" or safest option)
            validated[field] = list(valid_values.keys())[0]
    return validated


def parse_extraction_response(response: str) -> Dict[str, Any]:
    """
    Parse LLM extraction response to JSON.

    Args:
        response: Raw LLM response

    Returns:
        Parsed judgment dictionary
    """
    # Try to find JSON in response
    response = response.strip()

    # Handle markdown code blocks
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        response = response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        response = response[start:end].strip()

    # Parse JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse extraction response: {e}\nResponse: {response}")
