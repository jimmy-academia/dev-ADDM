"""Debug logger for full prompt/response capture.

Supports directory-based logging with per-item files:
- debug/phase1.jsonl: AMOS Phase 1 LLM calls
- debug/{item_id}.jsonl: Per-restaurant LLM calls
"""

from pathlib import Path
from threading import RLock
from datetime import datetime
import json
from typing import Any, Optional


class DebugLogger:
    """Logger for capturing full prompts and responses.

    Uses directory-based logging with per-item JSONL files for easy debugging.
    Thread-safe for concurrent logging.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        immediate_write: bool = True,
    ):
        """Initialize the debug logger.

        Args:
            output_dir: Directory where debug files will be written.
                       If None, logging is disabled.
            immediate_write: If True, write entries immediately (useful for debugging hangs).
                            If False, buffer entries until flush() is called.
        """
        self.output_dir = output_dir
        self.immediate_write = immediate_write
        self._entries: list[dict] = []
        self._lock = RLock()
        self._enabled = output_dir is not None
        self._debug_dir: Optional[Path] = None

        # Context routing: determines which file to write to
        self._current_context: Optional[str] = None  # item_id for per-item logging
        self._phase1_mode: bool = False  # True = write to phase1.jsonl

        # Setup debug directory
        if self._enabled and self.output_dir:
            self._debug_dir = self.output_dir / "debug"
            self._debug_dir.mkdir(parents=True, exist_ok=True)

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
        self._debug_dir = output_dir / "debug"
        self._debug_dir.mkdir(parents=True, exist_ok=True)

    def disable(self):
        """Disable debug logging."""
        self._enabled = False

    def set_context(self, item_id: str) -> None:
        """Set context for per-item logging.

        After calling this, all LLM calls will be logged to debug/{item_id}.jsonl.

        Args:
            item_id: Restaurant business ID or sample identifier
        """
        self._current_context = item_id
        self._phase1_mode = False

    def set_phase1_mode(self) -> None:
        """Set logger to Phase 1 mode.

        After calling this, all LLM calls will be logged to debug/phase1.jsonl.
        """
        self._phase1_mode = True
        self._current_context = None

    def clear_context(self) -> None:
        """Clear the current context.

        After calling this, logging falls back to sample_id from log_llm_call().
        """
        self._current_context = None
        self._phase1_mode = False

    def _get_output_path(self, sample_id: str) -> Path:
        """Get the output file path based on current context.

        Args:
            sample_id: Fallback sample ID if no context is set

        Returns:
            Path to the JSONL file to write to
        """
        if not self._debug_dir:
            raise RuntimeError("Debug directory not initialized")

        if self._phase1_mode:
            return self._debug_dir / "phase1.jsonl"

        # Use context item_id if set, otherwise use sample_id from the call
        item_id = self._current_context or sample_id

        # Sanitize for use as filename
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in item_id)
        return self._debug_dir / f"{safe_id}.jsonl"

    def log_llm_call(
        self,
        sample_id: str,
        phase: str,
        prompt: str,
        response: str,
        model: str,
        latency_ms: float,
        throttle_delay_ms: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """Log a complete LLM call with full prompt/response.

        Args:
            sample_id: Identifier for the sample being processed
            phase: Phase or step name (e.g., "phase1_extract_terms", "phase2_extract")
            prompt: Full prompt text sent to the LLM
            response: Full response text from the LLM
            model: Model name used
            latency_ms: Actual LLM API call duration in milliseconds
            throttle_delay_ms: Added delay before request (from request_delay config)
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            cost_usd: Cost of this LLM call in USD
            metadata: Optional additional metadata
        """
        if not self._enabled:
            return

        entry = {
            "timestamp": datetime.now().isoformat() + "Z",
            "phase": phase,
            "model": model,
            "prompt": prompt,
            "response": response,
            "latency_ms": latency_ms,
            "throttle_delay_ms": throttle_delay_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
        }
        if metadata:
            entry["metadata"] = metadata

        with self._lock:
            if self.immediate_write and self._debug_dir:
                # Write immediately to appropriate file
                output_path = self._get_output_path(sample_id)
                with open(output_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            else:
                # Buffer for later flush
                entry["_sample_id"] = sample_id  # Store for routing during flush
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
            "timestamp": datetime.now().isoformat() + "Z",
            "event_type": event_type,
            **data,
        }

        with self._lock:
            if self.immediate_write and self._debug_dir:
                # Write immediately to appropriate file
                output_path = self._get_output_path(sample_id)
                with open(output_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            else:
                # Buffer for later flush
                entry["_sample_id"] = sample_id
                self._entries.append(entry)

    def flush(self):
        """Write accumulated entries to disk.

        Writes to per-item files in debug/ subdirectory.
        """
        if not self._enabled or not self._entries or self._debug_dir is None:
            return

        with self._lock:
            entries_to_write = list(self._entries)
            self._entries.clear()

        # Group by sample_id and write to per-item files
        by_sample: dict[str, list[dict]] = {}
        for entry in entries_to_write:
            sid = entry.pop("_sample_id", "unknown")
            if sid not in by_sample:
                by_sample[sid] = []
            by_sample[sid].append(entry)

        # Write per-sample files
        for sample_id, entries in by_sample.items():
            output_path = self._get_output_path(sample_id)
            with open(output_path, "a") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")

    def get_entries(self, sample_id: Optional[str] = None) -> list[dict]:
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
            return [e for e in self._entries if e.get("_sample_id") == sample_id]

    def clear(self):
        """Clear all accumulated entries without writing to disk."""
        with self._lock:
            self._entries.clear()


# Optional global debug logger (disabled by default)
_global_debug_logger: Optional[DebugLogger] = None


def get_debug_logger() -> Optional[DebugLogger]:
    """Get the global debug logger if configured."""
    return _global_debug_logger


def set_debug_logger(logger: Optional[DebugLogger]):
    """Set the global debug logger."""
    global _global_debug_logger
    _global_debug_logger = logger
