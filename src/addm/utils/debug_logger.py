"""Debug logger for full prompt/response capture.

Supports directory-based logging with per-item files:
- debug/phase1.jsonl: AMOS Phase 1 LLM calls
- debug/{item_id}.jsonl: Per-restaurant LLM calls
"""

from pathlib import Path
from threading import RLock
from datetime import datetime
from contextvars import ContextVar
import json
from typing import Any, Optional

# Task-local routing context (safe for async concurrency)
_output_dir_ctx: ContextVar[Optional[Path]] = ContextVar("debug_output_dir", default=None)
_phase1_mode_ctx: ContextVar[bool] = ContextVar("debug_phase1_mode", default=False)
_item_id_ctx: ContextVar[Optional[str]] = ContextVar("debug_item_id", default=None)


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

        # Context routing: kept for backward compatibility but not used for routing.
        self._current_context: Optional[str] = None
        self._phase1_mode: bool = False

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

    def set_output_dir(self, output_dir: Path) -> None:
        """Set task-local output directory for debug routing."""
        _output_dir_ctx.set(output_dir)
        if output_dir:
            self._enabled = True

    def set_context(self, item_id: str) -> None:
        """Set context for per-item logging.

        After calling this, all LLM calls will be logged to debug/{item_id}.jsonl.

        Args:
            item_id: Restaurant business ID or sample identifier
        """
        self._current_context = item_id
        self._phase1_mode = False
        _item_id_ctx.set(item_id)
        _phase1_mode_ctx.set(False)

    def set_phase1_mode(self) -> None:
        """Set logger to Phase 1 mode.

        After calling this, all LLM calls will be logged to debug/phase1.jsonl.
        """
        self._phase1_mode = True
        self._current_context = None
        _phase1_mode_ctx.set(True)
        _item_id_ctx.set(None)

    def clear_context(self) -> None:
        """Clear the current context.

        After calling this, logging falls back to sample_id from log_llm_call().
        """
        self._current_context = None
        self._phase1_mode = False
        _item_id_ctx.set(None)
        _phase1_mode_ctx.set(False)

    def _get_output_path_for(
        self,
        output_dir: Path,
        sample_id: str,
        phase1_mode: bool,
        context_item_id: Optional[str],
    ) -> Path:
        """Get the output file path based on current context.

        Args:
            sample_id: Fallback sample ID if no context is set

        Returns:
            Path to the JSONL file to write to
        """
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        if phase1_mode:
            return debug_dir / "phase1.jsonl"

        # Use context item_id if set, otherwise use sample_id from the call
        item_id = context_item_id or sample_id
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in item_id)
        return debug_dir / f"{safe_id}.jsonl"

    def _get_output_path(self, sample_id: str) -> Path:
        """Get the output file path based on task-local context."""
        output_dir = _output_dir_ctx.get() or self.output_dir
        if not output_dir:
            raise RuntimeError("Debug output_dir not initialized")
        return self._get_output_path_for(
            output_dir=output_dir,
            sample_id=sample_id,
            phase1_mode=_phase1_mode_ctx.get(),
            context_item_id=_item_id_ctx.get(),
        )

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
            if self.immediate_write:
                # Write immediately to appropriate file
                output_path = self._get_output_path(sample_id)
                with open(output_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            else:
                # Buffer for later flush
                entry["_sample_id"] = sample_id  # Store for routing during flush
                entry["_output_dir"] = str(_output_dir_ctx.get() or self.output_dir or "")
                entry["_phase1_mode"] = _phase1_mode_ctx.get()
                entry["_context_item_id"] = _item_id_ctx.get()
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
            if self.immediate_write:
                # Write immediately to appropriate file
                output_path = self._get_output_path(sample_id)
                with open(output_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            else:
                # Buffer for later flush
                entry["_sample_id"] = sample_id
                entry["_output_dir"] = str(_output_dir_ctx.get() or self.output_dir or "")
                entry["_phase1_mode"] = _phase1_mode_ctx.get()
                entry["_context_item_id"] = _item_id_ctx.get()
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

        # Group by (output_dir, sample_id) and write to per-item files
        by_sample: dict[tuple[str, str], list[dict]] = {}
        for entry in entries_to_write:
            sid = entry.pop("_sample_id", "unknown")
            output_dir = entry.pop("_output_dir", "") or ""
            key = (output_dir, sid)
            if key not in by_sample:
                by_sample[key] = []
            by_sample[key].append(entry)

        # Write per-sample files
        for (output_dir, sample_id), entries in by_sample.items():
            if not output_dir:
                # Fallback to default output_dir if missing
                output_dir = str(self.output_dir or "")
            if not output_dir:
                continue
            phase1_mode = False
            context_item_id = None
            if entries:
                phase1_mode = entries[0].pop("_phase1_mode", False)
                context_item_id = entries[0].pop("_context_item_id", None)
            output_path = self._get_output_path_for(
                output_dir=Path(output_dir),
                sample_id=sample_id,
                phase1_mode=phase1_mode,
                context_item_id=context_item_id,
            )
            with open(output_path, "a") as f:
                for entry in entries:
                    entry.pop("_phase1_mode", None)
                    entry.pop("_context_item_id", None)
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
