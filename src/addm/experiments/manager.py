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
    root: Path
    config_path: Path
    results_path: Path
    metrics_path: Path
    logs_path: Path


class ExperimentManager:
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
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        if self.benchmark:
            parts = [p for p in [self.method, self.dataset, self.model] if p]
            parent = self.results_dir / "benchmarks" / "_".join(parts)
        else:
            parent = self.results_dir / "dev"
        run_dir = parent / f"{timestamp}_{self.run_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return RunPaths(
            root=run_dir,
            config_path=run_dir / "config.json",
            results_path=run_dir / "results.jsonl",
            metrics_path=run_dir / "metrics.json",
            logs_path=run_dir / "run.log",
        )

    def save_config(self, path: Path, config: Dict[str, Any]) -> None:
        write_json(path, config)

    def save_results(self, path: Path, results: Iterable[Dict[str, Any]]) -> None:
        write_jsonl(path, results)

    def save_metrics(self, path: Path, metrics: Dict[str, Any]) -> None:
        write_json(path, metrics)
