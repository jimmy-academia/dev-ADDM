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
        business_id: str,
        messages: List[Dict[str, str]],
        model: str,
        response: str,
        parsed: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[float] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a single item (restaurant) LLM interaction.

        Args:
            business_id: Restaurant/business identifier
            messages: Input messages to LLM
            model: Model name used
            response: Raw LLM response text
            parsed: Parsed response (if successful)
            latency_ms: API call latency in milliseconds
            prompt_tokens: Input token count
            completion_tokens: Output token count
            total_tokens: Total token count
            cost_usd: Cost in USD
            metadata: Additional metadata
        """
        if not self._enabled:
            return

        item_data: Dict[str, Any] = {
            "business_id": business_id,
            "input": {
                "messages": messages,
                "model": model,
            },
            "output": {
                "response": response,
            },
            "metrics": {},
        }

        if parsed is not None:
            item_data["output"]["parsed"] = parsed

        # Build metrics
        metrics = item_data["metrics"]
        if latency_ms is not None:
            metrics["latency_ms"] = latency_ms
        if prompt_tokens is not None:
            metrics["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            metrics["completion_tokens"] = completion_tokens
        if total_tokens is not None:
            metrics["total_tokens"] = total_tokens
        elif prompt_tokens is not None and completion_tokens is not None:
            metrics["total_tokens"] = prompt_tokens + completion_tokens
        if cost_usd is not None:
            metrics["cost_usd"] = cost_usd

        # Add metadata if provided
        if metadata:
            item_data["metadata"] = metadata

        # Sanitize business_id for filename
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in business_id)
        item_path = self.item_logs_dir / f"{safe_id}.json"

        with open(item_path, "w") as f:
            json.dump(item_data, f, indent=2)

    def log_item_error(
        self,
        business_id: str,
        error: str,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
    ) -> None:
        """Log an error for a single item.

        Args:
            business_id: Restaurant/business identifier
            error: Error message
            messages: Input messages (if available)
            model: Model name (if available)
        """
        if not self._enabled:
            return

        item_data: Dict[str, Any] = {
            "business_id": business_id,
            "error": error,
        }

        if messages is not None:
            item_data["input"] = {
                "messages": messages,
                "model": model or "unknown",
            }

        # Sanitize business_id for filename
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in business_id)
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
