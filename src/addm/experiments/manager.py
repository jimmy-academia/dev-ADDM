"""Experiment run management and result persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from addm.utils.io import write_json, write_jsonl


@dataclass
class RunPaths:
    """Paths for a single experiment run."""

    root: Path
    config_path: Path
    results_path: Path
    metrics_path: Path
    usage_path: Path  # NEW: usage summary
    logs_path: Path
    debug_dir: Path  # NEW: debug logs directory


class ExperimentManager:
    """Manages experiment run creation and result persistence."""

    def __init__(
        self,
        run_name: str,
        results_dir: Path,
        benchmark: bool = False,
        method: Optional[str] = None,
        dataset: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.run_name = run_name
        self.results_dir = results_dir
        self.benchmark = benchmark
        self.method = method
        self.dataset = dataset
        self.model = model

    def create_run(self) -> RunPaths:
        """Create a new run directory with all necessary paths.

        Returns:
            RunPaths with all output file paths
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        if self.benchmark:
            parts = [p for p in [self.method, self.dataset, self.model] if p]
            parent = self.results_dir / "benchmarks" / "_".join(parts)
        else:
            parent = self.results_dir / "dev"
        run_dir = parent / f"{timestamp}_{self.run_name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create debug directory
        debug_dir = run_dir / "debug"
        debug_dir.mkdir(exist_ok=True)

        return RunPaths(
            root=run_dir,
            config_path=run_dir / "config.json",
            results_path=run_dir / "results.jsonl",
            metrics_path=run_dir / "metrics.json",
            usage_path=run_dir / "usage.json",
            logs_path=run_dir / "run.log",
            debug_dir=debug_dir,
        )

    def save_config(self, path: Path, config: Dict[str, Any]) -> None:
        """Save run configuration to JSON file."""
        write_json(path, config)

    def save_results(self, path: Path, results: Iterable[Dict[str, Any]]) -> None:
        """Save per-sample results to JSONL file."""
        write_jsonl(path, results)

    def save_metrics(self, path: Path, metrics: Dict[str, Any]) -> None:
        """Save evaluation metrics to JSON file."""
        write_json(path, metrics)

    def save_usage(self, path: Path, usage_summary: Dict[str, Any]) -> None:
        """Save usage summary to JSON file.

        Args:
            path: Output path for usage.json
            usage_summary: Dict from UsageTracker.get_summary()
        """
        write_json(path, usage_summary)
