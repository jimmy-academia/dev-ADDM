"""Judgment extraction and caching."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from addm.tasks.prompt_parser import get_l0_field_order

# Multi-model configuration for GT extraction (cost-optimized)
# See docs/future/high_quality_gt.md for more robust config
REQUIRED_RUNS = {
    "gpt-5-mini": 1,
    "gpt-5-nano": 3,
}


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


class PolicyJudgmentCache:
    """
    Extended cache for policy-based GT extraction with multi-model support.

    Supports:
    - Raw storage: Individual extractions per model/run
    - Aggregated storage: Weighted majority vote results
    - Quota checking: Track which model/run combinations are complete
    - Term version tracking: Hash-based invalidation

    Structure:
    {
        "_metadata": {
            "topic": {"term_hash": "...", "created_at": "...", "model_config": {...}}
        },
        "raw": {
            "topic::review_id::model::run": {...judgment...}
        },
        "aggregated": {
            "topic::review_id": {...aggregated_judgment...}
        }
    }
    """

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self._data: Dict[str, Any] = {
            "_metadata": {},
            "raw": {},
            "aggregated": {},
        }
        self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                loaded = json.load(f)
                # Merge with default structure
                self._data["_metadata"] = loaded.get("_metadata", {})
                self._data["raw"] = loaded.get("raw", {})
                self._data["aggregated"] = loaded.get("aggregated", {})

    def save(self) -> None:
        """Save cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._data, f, indent=2)

    # -------------------------------------------------------------------------
    # Raw cache operations
    # -------------------------------------------------------------------------

    def _raw_key(self, topic: str, review_id: str, model: str, run: int) -> str:
        """Build raw cache key."""
        return f"{topic}::{review_id}::{model}::run{run}"

    def set_raw(
        self, topic: str, review_id: str, model: str, run: int, judgment: Dict[str, Any]
    ) -> None:
        """Store a raw extraction result."""
        key = self._raw_key(topic, review_id, model, run)
        # Add metadata
        judgment["_model"] = model
        judgment["_run"] = run
        judgment["_extracted_at"] = datetime.now().isoformat()
        self._data["raw"][key] = judgment

    def get_raw(
        self, topic: str, review_id: str, model: str, run: int
    ) -> Optional[Dict[str, Any]]:
        """Get a raw extraction result."""
        key = self._raw_key(topic, review_id, model, run)
        return self._data["raw"].get(key)

    def has_raw(self, topic: str, review_id: str, model: str, run: int) -> bool:
        """Check if raw extraction exists."""
        key = self._raw_key(topic, review_id, model, run)
        return key in self._data["raw"]

    def get_raw_by_review(
        self, topic: str, review_id: str
    ) -> List[Dict[str, Any]]:
        """Get all raw extractions for a review (across all models/runs)."""
        prefix = f"{topic}::{review_id}::"
        results = []
        for key, value in self._data["raw"].items():
            if key.startswith(prefix):
                results.append(value)
        return results

    def get_cached_runs(self, topic: str, review_id: str, model: str) -> set:
        """Get which run numbers are cached for a model.

        Uses direct dict lookups (O(1) each) instead of scanning all keys.
        Checks runs 1-10 which covers all practical use cases.
        """
        runs = set()
        raw = self._data["raw"]
        for run in range(1, 11):  # Check runs 1-10 directly
            key = f"{topic}::{review_id}::{model}::run{run}"
            if key in raw:
                runs.add(run)
        return runs

    def count_cached_runs(self, topic: str, review_id: str, model: str) -> int:
        """Count how many runs are cached for a model."""
        return len(self.get_cached_runs(topic, review_id, model))

    def count_raw_by_model(self, topic: str) -> Dict[str, int]:
        """Count raw entries by model for a topic.

        Returns:
            Dict mapping model name to count of raw entries
        """
        from collections import Counter
        counts: Counter = Counter()
        prefix = f"{topic}::"
        for key in self._data["raw"].keys():
            if key.startswith(prefix):
                # Key format: topic::review_id::model::runN
                parts = key.split("::")
                if len(parts) >= 3:
                    model = parts[2]
                    counts[model] += 1
        return dict(counts)

    def get_raw_review_ids(self, topic: str) -> set:
        """Get all unique review_ids that have raw extractions for a topic.

        Returns:
            Set of review_ids
        """
        prefix = f"{topic}::"
        review_ids = set()
        for key in self._data["raw"].keys():
            if key.startswith(prefix):
                # Key format: topic::review_id::model::runN
                parts = key.split("::")
                if len(parts) >= 2:
                    review_ids.add(parts[1])
        return review_ids

    # -------------------------------------------------------------------------
    # Aggregated cache operations
    # -------------------------------------------------------------------------

    def _agg_key(self, topic: str, review_id: str) -> str:
        """Build aggregated cache key."""
        return f"{topic}::{review_id}"

    def set_aggregated(
        self, topic: str, review_id: str, judgment: Dict[str, Any]
    ) -> None:
        """Store an aggregated judgment."""
        key = self._agg_key(topic, review_id)
        judgment["_aggregated_at"] = datetime.now().isoformat()
        self._data["aggregated"][key] = judgment

    def get_aggregated(self, topic: str, review_id: str) -> Optional[Dict[str, Any]]:
        """Get aggregated judgment for a review."""
        key = self._agg_key(topic, review_id)
        return self._data["aggregated"].get(key)

    def has_aggregated(self, topic: str, review_id: str) -> bool:
        """Check if aggregated judgment exists."""
        key = self._agg_key(topic, review_id)
        return key in self._data["aggregated"]

    def get_all_aggregated(self, topic: str) -> Dict[str, Dict[str, Any]]:
        """Get all aggregated judgments for a topic."""
        prefix = f"{topic}::"
        results = {}
        for key, value in self._data["aggregated"].items():
            if key.startswith(prefix):
                review_id = key[len(prefix):]
                results[review_id] = value
        return results

    def count_aggregated(self, topic: str) -> int:
        """Count aggregated judgments for a topic."""
        prefix = f"{topic}::"
        return sum(1 for key in self._data["aggregated"] if key.startswith(prefix))

    # -------------------------------------------------------------------------
    # Quota checking
    # -------------------------------------------------------------------------

    def needs_extraction(
        self, topic: str, review_id: str, required_runs: Optional[Dict[str, int]] = None
    ) -> List[Tuple[str, int]]:
        """
        Check which model/run combinations still need extraction.

        Args:
            topic: Topic identifier
            review_id: Review identifier
            required_runs: Dict of model -> required run count (default: REQUIRED_RUNS)

        Returns:
            List of (model, run) tuples that need extraction
        """
        if required_runs is None:
            required_runs = REQUIRED_RUNS

        needed: List[Tuple[str, int]] = []
        for model, required in required_runs.items():
            # Get WHICH runs exist, not just count
            existing_runs = self.get_cached_runs(topic, review_id, model)
            # Check each required run (1, 2, ..., required)
            for run in range(1, required + 1):
                if run not in existing_runs:
                    needed.append((model, run))

        return needed

    def is_quota_satisfied(
        self, topic: str, review_id: str, required_runs: Optional[Dict[str, int]] = None
    ) -> bool:
        """Check if all model/run quotas are satisfied for a review."""
        return len(self.needs_extraction(topic, review_id, required_runs)) == 0

    # -------------------------------------------------------------------------
    # Metadata operations
    # -------------------------------------------------------------------------

    def set_topic_metadata(
        self,
        topic: str,
        term_hash: str,
        model_config: Optional[Dict[str, int]] = None,
    ) -> None:
        """Store metadata for a topic."""
        if model_config is None:
            model_config = REQUIRED_RUNS
        self._data["_metadata"][topic] = {
            "term_hash": term_hash,
            "created_at": datetime.now().isoformat(),
            "model_config": model_config,
        }

    def get_topic_metadata(self, topic: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a topic."""
        return self._data["_metadata"].get(topic)

    def check_term_hash(self, topic: str, current_hash: str) -> bool:
        """
        Check if current term hash matches cached.

        Returns True if match (safe to use cache), False if mismatch.
        """
        meta = self.get_topic_metadata(topic)
        if not meta:
            return True  # No existing cache, OK to proceed
        return meta.get("term_hash") == current_hash

    def invalidate_topic(self, topic: str) -> int:
        """
        Invalidate all cache entries for a topic.

        Returns count of entries removed.
        """
        count = 0

        # Remove raw entries
        prefix = f"{topic}::"
        raw_keys = [k for k in self._data["raw"].keys() if k.startswith(prefix)]
        for key in raw_keys:
            del self._data["raw"][key]
            count += 1

        # Remove aggregated entries
        agg_keys = [k for k in self._data["aggregated"].keys() if k.startswith(prefix)]
        for key in agg_keys:
            del self._data["aggregated"][key]
            count += 1

        # Remove metadata
        if topic in self._data["_metadata"]:
            del self._data["_metadata"][topic]

        return count


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
If no relevant content, output: {{"is_relevant": false}}

Output JSON only:
{{
  "review_id": "{review_id}",
  "is_relevant": true,
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
    # Support both field names for backward compatibility
    if not judgment.get("is_relevant", judgment.get("is_allergy_related", False)):
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
