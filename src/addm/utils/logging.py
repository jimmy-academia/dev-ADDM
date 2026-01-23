"""Key result logging for ADDM experiments.

Captures important experiment outcomes and metrics for analysis.
Separate from output.py (display) and debug_logger.py (LLM capture).

NOTE: ResultLogger is DEPRECATED. Per-sample results are now written to
item_logs/{sample_id}.json via ItemLogger. Aggregated usage is in results.json.
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any


class ResultLogger:
    """DEPRECATED: Logger for capturing key experiment results.

    This class is deprecated. Use ItemLogger for per-sample logging.
    Per-sample data should go to item_logs/{sample_id}.json.
    Aggregated metrics should go directly to results.json.

    Kept for backward compatibility only.
    """

    def __init__(self, output_file: Path | None = None):
        """Initialize result logger.

        DEPRECATED: Use ItemLogger instead.

        Args:
            output_file: Path to write results. If None, logging disabled.
        """
        warnings.warn(
            "ResultLogger is deprecated. Use ItemLogger for per-sample logging. "
            "Per-sample data goes to item_logs/{sample_id}.json.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.output_file = output_file
        self._results: list[dict] = []
        self._enabled = output_file is not None

    @property
    def enabled(self) -> bool:
        """Whether result logging is enabled."""
        return self._enabled

    def enable(self, output_file: Path):
        """Enable result logging to specified file."""
        self.output_file = output_file
        self._enabled = True
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def disable(self):
        """Disable result logging."""
        self._enabled = False

    def log_result(
        self,
        sample_id: str,
        verdict: str,
        ground_truth: str | None = None,
        risk_score: float | None = None,
        correct: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Log a single sample result.

        Args:
            sample_id: Sample identifier (business_id)
            verdict: Predicted verdict
            ground_truth: True verdict (if available)
            risk_score: Numeric risk score
            correct: Whether prediction matches GT
            metadata: Additional result data
        """
        if not self._enabled:
            return

        result = {
            "timestamp": datetime.now().isoformat(),
            "sample_id": sample_id,
            "verdict": verdict,
        }
        if ground_truth is not None:
            result["ground_truth"] = ground_truth
        if risk_score is not None:
            result["risk_score"] = risk_score
        if correct is not None:
            result["correct"] = correct
        if metadata:
            result.update(metadata)

        self._results.append(result)

        # Write immediately (append mode)
        if self.output_file:
            with open(self.output_file, "a") as f:
                f.write(json.dumps(result) + "\n")

    def log_metrics(
        self,
        metrics: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ):
        """Log aggregate metrics at end of run.

        Args:
            metrics: Metric values (accuracy, AUPRC, etc.)
            metadata: Additional context (run_id, config, etc.)
        """
        if not self._enabled:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "metrics",
            "metrics": metrics,
        }
        if metadata:
            entry["metadata"] = metadata

        if self.output_file:
            with open(self.output_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def clear(self):
        """Clear in-memory results."""
        self._results.clear()


# Global result logger (disabled by default)
_global_result_logger: ResultLogger | None = None


def get_result_logger() -> ResultLogger | None:
    """Get the global result logger if configured."""
    return _global_result_logger


def set_result_logger(logger: ResultLogger | None):
    """Set the global result logger."""
    global _global_result_logger
    _global_result_logger = logger


# Keep the old setup_logging for backward compatibility
def setup_logging(verbose: bool = True) -> logging.Logger:
    """Legacy logging setup for backward compatibility."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger("addm")
