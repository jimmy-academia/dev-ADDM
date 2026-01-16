"""Debug logger for full prompt/response capture."""

from pathlib import Path
from threading import RLock
from datetime import datetime
import json
from typing import Any


class DebugLogger:
    """Logger for capturing full prompts and responses.

    Organizes debug entries by sample_id for easy inspection.
    Thread-safe for concurrent logging.
    """

    def __init__(self, output_dir: Path | None = None):
        """Initialize the debug logger.

        Args:
            output_dir: Directory where debug files will be written.
                       If None, logging is disabled.
        """
        self.output_dir = output_dir
        self._entries: list[dict] = []
        self._lock = RLock()
        self._enabled = output_dir is not None

    @property
    def enabled(self) -> bool:
        """Whether debug logging is enabled."""
        return self._enabled

    def enable(self, output_dir: Path):
        """Enable debug logging to the specified directory.

        Args:
            output_dir: Directory for debug output
        """
        self.output_dir = output_dir
        self._enabled = True

    def disable(self):
        """Disable debug logging."""
        self._enabled = False

    def log_llm_call(
        self,
        sample_id: str,
        phase: str,
        prompt: str,
        response: str,
        model: str,
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
    ):
        """Log a complete LLM call with full prompt/response.

        Args:
            sample_id: Identifier for the sample being processed
            phase: Phase or step name (e.g., "main", "extraction", "verification")
            prompt: Full prompt text sent to the LLM
            response: Full response text from the LLM
            model: Model name used
            latency_ms: API call duration in milliseconds
            metadata: Optional additional metadata
        """
        if not self._enabled:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "sample_id": sample_id,
            "phase": phase,
            "model": model,
            "latency_ms": latency_ms,
            "prompt": prompt,
            "response": response,
        }
        if metadata:
            entry["metadata"] = metadata

        with self._lock:
            self._entries.append(entry)

    def log_event(
        self,
        sample_id: str,
        event_type: str,
        data: dict[str, Any],
    ):
        """Log a generic event.

        Args:
            sample_id: Identifier for the sample
            event_type: Type of event (e.g., "parse_error", "retry", "timeout")
            data: Event data
        """
        if not self._enabled:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "sample_id": sample_id,
            "event_type": event_type,
            **data,
        }

        with self._lock:
            self._entries.append(entry)

    def flush(self):
        """Write accumulated entries to disk, organized by sample.

        Creates a debug/ subdirectory with per-sample JSONL files.
        """
        if not self._enabled or not self._entries or self.output_dir is None:
            return

        debug_dir = self.output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Group by sample_id
        by_sample: dict[str, list[dict]] = {}
        with self._lock:
            for entry in self._entries:
                sid = entry.get("sample_id", "unknown")
                if sid not in by_sample:
                    by_sample[sid] = []
                by_sample[sid].append(entry)
            self._entries.clear()

        # Write per-sample files
        for sample_id, entries in by_sample.items():
            # Sanitize sample_id for use as filename
            safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in sample_id)
            path = debug_dir / f"{safe_id}.jsonl"
            with open(path, "a") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")

    def get_entries(self, sample_id: str | None = None) -> list[dict]:
        """Get accumulated entries, optionally filtered by sample_id.

        Note: Returns only unflushed entries still in memory.

        Args:
            sample_id: If provided, only return entries for this sample

        Returns:
            List of log entries
        """
        with self._lock:
            if sample_id is None:
                return list(self._entries)
            return [e for e in self._entries if e.get("sample_id") == sample_id]

    def clear(self):
        """Clear all accumulated entries without writing to disk."""
        with self._lock:
            self._entries.clear()


# Optional global debug logger (disabled by default)
_global_debug_logger: DebugLogger | None = None


def get_debug_logger() -> DebugLogger | None:
    """Get the global debug logger if configured."""
    return _global_debug_logger


def set_debug_logger(logger: DebugLogger | None):
    """Set the global debug logger."""
    global _global_debug_logger
    _global_debug_logger = logger
