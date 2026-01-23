"""Per-item LLM interaction logging for ADDM experiments.

Logs LLM input/output to item_logs/{business_id}.json for each restaurant.
Also creates _query_template.json with the prompt template for inspection.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ItemLogger:
    """Logger for per-item (restaurant) LLM interactions.

    Writes to:
    - item_logs/_query_template.json: Query template (no context)
    - item_logs/{business_id}.json: Per-restaurant LLM interaction
    """

    def __init__(self, run_dir: Path):
        """Initialize ItemLogger.

        Args:
            run_dir: Base run directory (contains item_logs/ subdirectory)
        """
        self.run_dir = run_dir
        self.item_logs_dir = run_dir / "item_logs"
        self.item_logs_dir.mkdir(parents=True, exist_ok=True)
        self._enabled = True

    @property
    def enabled(self) -> bool:
        """Whether item logging is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable item logging."""
        self._enabled = True

    def disable(self) -> None:
        """Disable item logging."""
        self._enabled = False

    def log_query_template(
        self,
        policy_id: str,
        query_text: str,
        example_context: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Log the query template for inspection.

        Args:
            policy_id: Policy identifier
            query_text: The agenda/query text
            example_context: Optional example context (e.g., "Review 1 of K...")
            system_prompt: Optional system prompt
        """
        if not self._enabled:
            return

        template_data = {
            "policy_id": policy_id,
            "query_text": query_text,
        }

        if example_context:
            template_data["example_context"] = example_context

        if system_prompt:
            template_data["system_prompt"] = system_prompt

        template_path = self.item_logs_dir / "_query_template.json"
        with open(template_path, "w") as f:
            json.dump(template_data, f, indent=2)

    def log_item(
        self,
        sample_id: str,
        verdict: str,
        output: str,
        parsed: Optional[Dict[str, Any]] = None,
        ground_truth: Optional[str] = None,
        correct: Optional[bool] = None,
        metrics: Optional[Dict[str, Any]] = None,
        **extra,
    ) -> None:
        """Log a single sample result with full details.

        Written immediately as each sample completes for streaming output.
        Optimized for debugging - easy to inspect individual samples.

        Args:
            sample_id: Sample identifier (business_id)
            verdict: Predicted verdict
            output: Raw LLM response text
            parsed: Parsed response (if successful)
            ground_truth: True verdict (if available)
            correct: Whether prediction matches GT
            metrics: Token/cost/latency metrics dict with keys:
                - prompt_tokens, completion_tokens, total_tokens
                - cost_usd, latency_ms, llm_calls
                - For AMOS: total, phase1, phase2_stage1, phase2_stage2
            **extra: Method-specific fields (filter_stats, early_exit, etc.)
        """
        if not self._enabled:
            return

        item_data: Dict[str, Any] = {
            "sample_id": sample_id,
            "verdict": verdict,
            "output": output,
        }

        if parsed is not None:
            item_data["parsed"] = parsed
        if ground_truth is not None:
            item_data["ground_truth"] = ground_truth
        if correct is not None:
            item_data["correct"] = correct
        if metrics is not None:
            item_data["metrics"] = metrics

        # Add any method-specific extra fields
        for key, value in extra.items():
            if value is not None:
                item_data[key] = value

        # Sanitize sample_id for filename
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in sample_id)
        item_path = self.item_logs_dir / f"{safe_id}.json"

        with open(item_path, "w") as f:
            json.dump(item_data, f, indent=2)

    def log_item_error(
        self,
        sample_id: str,
        error: str,
        ground_truth: Optional[str] = None,
    ) -> None:
        """Log an error for a single sample.

        Args:
            sample_id: Sample identifier (business_id)
            error: Error message
            ground_truth: True verdict (if available)
        """
        if not self._enabled:
            return

        item_data: Dict[str, Any] = {
            "sample_id": sample_id,
            "error": error,
            "verdict": None,
            "correct": False,
        }

        if ground_truth is not None:
            item_data["ground_truth"] = ground_truth

        # Sanitize sample_id for filename
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in sample_id)
        item_path = self.item_logs_dir / f"{safe_id}.json"

        with open(item_path, "w") as f:
            json.dump(item_data, f, indent=2)


# Global item logger (disabled by default)
_global_item_logger: Optional[ItemLogger] = None


def get_item_logger() -> Optional[ItemLogger]:
    """Get the global item logger if configured."""
    return _global_item_logger


def set_item_logger(logger: Optional[ItemLogger]) -> None:
    """Set the global item logger."""
    global _global_item_logger
    _global_item_logger = logger
