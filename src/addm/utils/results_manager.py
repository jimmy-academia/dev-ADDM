"""Results directory management for ADDM experiments.

Handles the directory structure for dev vs benchmark runs:
- Dev: results/dev/{YYYYMMDD}_{run_name}/
- Benchmark: results/{method}/{policy_id}/run_N/

Includes quota system for benchmark runs (5 total: 1 ondemand + 4 batch).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class ResultsManager:
    """Manages results directory structure and benchmark quota system.

    Provides:
    - Directory creation for dev and benchmark runs
    - Quota tracking (5 runs per method/policy: 1 ondemand, 4 batch)
    - Aggregation across multiple benchmark runs
    """

    DEFAULT_QUOTA = 5  # 1 ondemand + 4 batch
    RESULTS_ROOT = Path("results")

    def __init__(self, results_root: Optional[Path] = None):
        """Initialize ResultsManager.

        Args:
            results_root: Override root directory (default: results/)
        """
        self.results_root = results_root or self.RESULTS_ROOT

    # -------------------------------------------------------------------------
    # Dev mode
    # -------------------------------------------------------------------------

    def get_dev_run_dir(self, run_name: str, timestamp: Optional[str] = None) -> Path:
        """Get directory for a dev run.

        Args:
            run_name: Run identifier (e.g., policy_id)
            timestamp: Optional timestamp (default: current time YYYYMMDD_HHMMSS)

        Returns:
            Path to dev run directory: results/dev/{YYYYMMDD_HHMMSS}_{run_name}/
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Use full timestamp for unique directories (prevents overwrites)
        run_dir = self.results_root / "dev" / f"{timestamp}_{run_name}"
        return run_dir

    def create_dev_run_dir(self, run_name: str, timestamp: Optional[str] = None) -> Path:
        """Create directory for a dev run.

        Args:
            run_name: Run identifier (e.g., policy_id)
            timestamp: Optional timestamp (default: current time)

        Returns:
            Path to created dev run directory
        """
        run_dir = self.get_dev_run_dir(run_name, timestamp)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "item_logs").mkdir(exist_ok=True)
        return run_dir

    # -------------------------------------------------------------------------
    # Benchmark mode
    # -------------------------------------------------------------------------

    def get_benchmark_dir(self, method: str, policy_id: str) -> Path:
        """Get base directory for benchmark runs.

        Args:
            method: Method name (e.g., "direct", "amos", "rag")
            policy_id: Policy identifier (e.g., "G1_allergy_V2")

        Returns:
            Path to benchmark directory: results/{method}/{policy_id}/
        """
        return self.results_root / method / policy_id

    def get_completed_runs(self, method: str, policy_id: str) -> Dict[str, Any]:
        """Count valid completed runs by mode.

        Args:
            method: Method name
            policy_id: Policy identifier

        Returns:
            Dict with counts: {"ondemand": int, "batch": int, "total": int, "runs": list}
        """
        benchmark_dir = self.get_benchmark_dir(method, policy_id)

        ondemand_count = 0
        batch_count = 0
        runs: List[Dict[str, Any]] = []

        if not benchmark_dir.exists():
            return {"ondemand": 0, "batch": 0, "total": 0, "runs": []}

        # Scan for run_N directories
        for run_dir in sorted(benchmark_dir.glob("run_*")):
            if not run_dir.is_dir():
                continue

            results_path = run_dir / "results.json"
            if not results_path.exists():
                continue

            try:
                with open(results_path) as f:
                    results = json.load(f)

                run_mode = results.get("mode", "ondemand")
                run_info = {
                    "run_id": run_dir.name,
                    "mode": run_mode,
                    "accuracy": results.get("accuracy", 0.0),
                    "n": results.get("n", 0),
                    "timestamp": results.get("timestamp", ""),
                    "has_latency": results.get("has_latency", run_mode == "ondemand"),
                }
                runs.append(run_info)

                if run_mode == "ondemand":
                    ondemand_count += 1
                else:
                    batch_count += 1

            except (json.JSONDecodeError, IOError):
                continue

        return {
            "ondemand": ondemand_count,
            "batch": batch_count,
            "total": ondemand_count + batch_count,
            "runs": runs,
        }

    def get_next_mode(self, method: str, policy_id: str) -> Optional[str]:
        """Determine mode for next run, or None if quota met.

        Logic:
        - run_1: always ondemand (captures latency)
        - run_2-5: always batch (cost efficient)
        - run_6+: None (quota met)

        Args:
            method: Method name
            policy_id: Policy identifier

        Returns:
            "ondemand", "batch", or None if quota met
        """
        completed = self.get_completed_runs(method, policy_id)

        if completed["total"] >= self.DEFAULT_QUOTA:
            return None  # Quota met

        if completed["ondemand"] == 0:
            return "ondemand"  # First run must be ondemand

        return "batch"  # Remaining runs are batch

    def needs_more_runs(self, method: str, policy_id: str) -> bool:
        """Check if more runs are needed.

        Args:
            method: Method name
            policy_id: Policy identifier

        Returns:
            True if under quota
        """
        return self.get_next_mode(method, policy_id) is not None

    def get_next_run_number(self, method: str, policy_id: str) -> int:
        """Get the next run number.

        Args:
            method: Method name
            policy_id: Policy identifier

        Returns:
            Next run number (1-indexed)
        """
        benchmark_dir = self.get_benchmark_dir(method, policy_id)

        if not benchmark_dir.exists():
            return 1

        existing = [
            int(d.name.split("_")[1])
            for d in benchmark_dir.glob("run_*")
            if d.is_dir() and d.name.split("_")[1].isdigit()
        ]

        return max(existing, default=0) + 1

    def get_or_create_run_dir(
        self,
        method: str,
        policy_id: str,
        force: bool = False,
    ) -> Optional[Path]:
        """Get next run_N directory, creating if needed.

        Args:
            method: Method name
            policy_id: Policy identifier
            force: If True, create even if quota met

        Returns:
            Path to run directory, or None if quota met (and not force)
        """
        if not force and not self.needs_more_runs(method, policy_id):
            return None

        run_num = self.get_next_run_number(method, policy_id)
        run_dir = self.get_benchmark_dir(method, policy_id) / f"run_{run_num}"

        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "item_logs").mkdir(exist_ok=True)

        return run_dir

    # -------------------------------------------------------------------------
    # Aggregation
    # -------------------------------------------------------------------------

    def aggregate_benchmark_runs(
        self,
        method: str,
        policy_id: str,
    ) -> Dict[str, Any]:
        """Aggregate results across all benchmark runs.

        Args:
            method: Method name
            policy_id: Policy identifier

        Returns:
            Dict with aggregated metrics:
            {
                "runs": [...],  # Individual run summaries
                "accuracy": {"mean": float, "std": float},
                "auprc": {"mean": float, "std": float},
                "latency": {"p50": float, "p95": float},  # From ondemand runs only
                "total_runs": int,
            }
        """
        completed = self.get_completed_runs(method, policy_id)
        runs = completed["runs"]

        if not runs:
            return {
                "runs": [],
                "accuracy": {"mean": 0.0, "std": 0.0},
                "auprc": {"mean": 0.0, "std": 0.0},
                "latency": None,
                "total_runs": 0,
            }

        # Load full results for detailed metrics
        benchmark_dir = self.get_benchmark_dir(method, policy_id)
        accuracies = []
        auprcs = []
        latencies_p50 = []
        latencies_p95 = []

        for run_info in runs:
            run_dir = benchmark_dir / run_info["run_id"]
            results_path = run_dir / "results.json"

            if not results_path.exists():
                continue

            try:
                with open(results_path) as f:
                    results = json.load(f)

                accuracies.append(results.get("accuracy", 0.0))

                # AUPRC from auprc or unified_scores
                auprc_val = None
                if "auprc" in results and isinstance(results["auprc"], dict):
                    auprc_val = results["auprc"].get("mean", 0.0)
                elif "unified_scores" in results and results["unified_scores"]:
                    auprc_val = results["unified_scores"].get("auprc", 0.0)
                if auprc_val is not None:
                    auprcs.append(auprc_val)

                # Latency (only from ondemand runs)
                if run_info.get("has_latency") or results.get("mode") == "ondemand":
                    usage = results.get("usage", {})
                    if "latency_p50_ms" in usage:
                        latencies_p50.append(usage["latency_p50_ms"])
                    if "latency_p95_ms" in usage:
                        latencies_p95.append(usage["latency_p95_ms"])

            except (json.JSONDecodeError, IOError):
                continue

        result = {
            "runs": runs,
            "total_runs": len(runs),
            "accuracy": {
                "mean": float(np.mean(accuracies)) if accuracies else 0.0,
                "std": float(np.std(accuracies)) if len(accuracies) > 1 else 0.0,
            },
            "auprc": {
                "mean": float(np.mean(auprcs)) if auprcs else 0.0,
                "std": float(np.std(auprcs)) if len(auprcs) > 1 else 0.0,
            },
        }

        # Latency from ondemand runs
        if latencies_p50:
            result["latency"] = {
                "p50_ms": float(np.mean(latencies_p50)),
                "p95_ms": float(np.mean(latencies_p95)) if latencies_p95 else None,
            }
        else:
            result["latency"] = None

        return result

    def print_aggregate_summary(
        self,
        method: str,
        policy_id: str,
        output_fn=None,
    ) -> None:
        """Print aggregated summary across benchmark runs.

        Args:
            method: Method name
            policy_id: Policy identifier
            output_fn: Output function (default: print)
        """
        if output_fn is None:
            output_fn = print

        agg = self.aggregate_benchmark_runs(method, policy_id)

        output_fn(f"\n{'='*60}")
        output_fn(f"BENCHMARK AGGREGATE: {method}/{policy_id}")
        output_fn(f"Total runs: {agg['total_runs']}/{self.DEFAULT_QUOTA}")
        output_fn(f"{'='*60}")

        if agg["total_runs"] == 0:
            output_fn("No completed runs found.")
            return

        # Accuracy
        acc = agg["accuracy"]
        output_fn(f"Accuracy: {acc['mean']:.1%} +/- {acc['std']:.1%}")

        # AUPRC
        auprc = agg["auprc"]
        output_fn(f"AUPRC:    {auprc['mean']:.3f} +/- {auprc['std']:.3f}")

        # Latency (if available)
        if agg["latency"]:
            lat = agg["latency"]
            output_fn(f"Latency:  p50={lat['p50_ms']:.0f}ms", end="")
            if lat.get("p95_ms"):
                output_fn(f", p95={lat['p95_ms']:.0f}ms")
            else:
                output_fn("")

        # Individual runs
        output_fn(f"\nRuns:")
        for run in agg["runs"]:
            mode_indicator = "[O]" if run["mode"] == "ondemand" else "[B]"
            output_fn(
                f"  {run['run_id']} {mode_indicator}: "
                f"accuracy={run['accuracy']:.1%}, n={run['n']}"
            )

        output_fn(f"{'='*60}\n")

    # -------------------------------------------------------------------------
    # Special tasks
    # -------------------------------------------------------------------------

    def get_special_task_dir(self, task_name: str, multi_run: bool = False) -> Path:
        """Get directory for special/one-off tasks.

        Args:
            task_name: Name of the special task
            multi_run: If True, returns base dir; use run_N subdirs

        Returns:
            Path to special task directory
        """
        return self.results_root / task_name

    # -------------------------------------------------------------------------
    # Shared cache
    # -------------------------------------------------------------------------

    def get_shared_cache_dir(self) -> Path:
        """Get path to shared cache directory.

        Returns:
            Path to results/shared/
        """
        shared_dir = self.results_root / "shared"
        shared_dir.mkdir(parents=True, exist_ok=True)
        return shared_dir

    def get_shared_embeddings_path(self) -> Path:
        """Get path to shared embeddings file.

        Returns:
            Path to results/shared/yelp_embeddings.json
        """
        return self.get_shared_cache_dir() / "yelp_embeddings.json"


# Global instance for convenience
_global_results_manager: Optional[ResultsManager] = None


def get_results_manager() -> ResultsManager:
    """Get global ResultsManager instance."""
    global _global_results_manager
    if _global_results_manager is None:
        _global_results_manager = ResultsManager()
    return _global_results_manager


def set_results_manager(manager: ResultsManager) -> None:
    """Set global ResultsManager instance."""
    global _global_results_manager
    _global_results_manager = manager
