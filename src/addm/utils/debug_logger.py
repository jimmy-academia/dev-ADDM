"""Debug logger for full prompt/response capture.

Supports two modes:
- Consolidated (default): Single debug.jsonl in run directory
- Per-sample: Separate files per sample_id (legacy mode)
"""

from pathlib import Path
from threading import RLock
from datetime import datetime
import json
from typing import Any


class DebugLogger:
    """Logger for capturing full prompts and responses.

    Supports consolidated mode (single debug.jsonl) or per-sample files.
    Thread-safe for concurrent logging.
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        consolidated: bool = True,
    ):
        """Initialize the debug logger.

        Args:
            output_dir: Directory where debug files will be written.
                       If None, logging is disabled.
            consolidated: If True, write all entries to single debug.jsonl.
                         If False, write per-sample files (legacy mode).
        """
        self.output_dir = output_dir
        self.consolidated = consolidated
        self._entries: list[dict] = []
        self._lock = RLock()
        self._enabled = output_dir is not None
        self._debug_file_path: Path | None = None

        # Setup consolidated file path
        if self._enabled and self.consolidated and self.output_dir:
            self._debug_file_path = self.output_dir / "debug.jsonl"

    @property
    def enabled(self) -> bool:
        """Whether debug logging is enabled."""
        return self._enabled

    def enable(self, output_dir: Path, consolidated: bool | None = None):
        """Enable debug logging to the specified directory.

        Args:
            output_dir: Directory for debug output
            consolidated: If provided, override consolidated mode setting
        """
        self.output_dir = output_dir
        self._enabled = True

        if consolidated is not None:
            self.consolidated = consolidated

        # Setup consolidated file path
        if self.consolidated:
            self._debug_file_path = self.output_dir / "debug.jsonl"
        else:
            self._debug_file_path = None

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
        """Write accumulated entries to disk.

        In consolidated mode: Appends all entries to single debug.jsonl
        In per-sample mode: Creates debug/ subdirectory with per-sample JSONL files
        """
        if not self._enabled or not self._entries or self.output_dir is None:
            return

        with self._lock:
            entries_to_write = list(self._entries)
            self._entries.clear()

        if self.consolidated:
            # Write all entries to single debug.jsonl
            with open(self._debug_file_path, "a") as f:
                for entry in entries_to_write:
                    f.write(json.dumps(entry) + "\n")
        else:
            # Legacy mode: Write per-sample files
            debug_dir = self.output_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)

            # Group by sample_id
            by_sample: dict[str, list[dict]] = {}
            for entry in entries_to_write:
                sid = entry.get("sample_id", "unknown")
                if sid not in by_sample:
                    by_sample[sid] = []
                by_sample[sid].append(entry)

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
