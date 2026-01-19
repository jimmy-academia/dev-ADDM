"""
CLI: Display formatted result summaries from results.json files.

Usage:
    # Default: show latest benchmark results for ALL methods
    .venv/bin/python -m addm.eval.cli.show_results

    # Single method
    .venv/bin/python -m addm.eval.cli.show_results --method amos

    # Multiple methods
    .venv/bin/python -m addm.eval.cli.show_results --method amos,direct,rag

    # Filter by policy
    .venv/bin/python -m addm.eval.cli.show_results --policy G1_allergy_V2

    # Specific dev path(s)
    .venv/bin/python -m addm.eval.cli.show_results --dev results/dev/20260119_004007_G1_allergy_V2/

    # Future: LaTeX table for paper
    .venv/bin/python -m addm.eval.cli.show_results --format latex
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

# Default directories
BENCHMARK_DIR = Path("results/benchmark")
DEV_DIR = Path("results/dev")
ALL_METHODS = ["direct", "rag", "rlm", "amos"]

# Process score weights for display
PROCESS_WEIGHTS = {
    "incident_precision": 0.35,
    "severity_accuracy": 0.30,
    "modifier_accuracy": 0.15,
    "verdict_support_rate": 0.15,
    "snippet_validity": 0.05,
}


def find_benchmark_results(
    methods: Optional[List[str]] = None,
    policy: Optional[str] = None,
) -> List[Path]:
    """Find benchmark result files by method and/or policy.

    Args:
        methods: List of methods to include (default: all)
        policy: Filter by policy ID

    Returns:
        List of result file paths
    """
    methods = methods or ALL_METHODS
    results = []

    if not BENCHMARK_DIR.exists():
        return results

    # Scan benchmark directory
    for policy_dir in sorted(BENCHMARK_DIR.iterdir()):
        if not policy_dir.is_dir():
            continue
        if policy and policy_dir.name != policy:
            continue

        for method in methods:
            result_file = policy_dir / method / "results.json"
            if result_file.exists():
                results.append(result_file)

    return results


def find_latest_dev_results(
    methods: Optional[List[str]] = None,
    policy: Optional[str] = None,
    limit: int = 5,
) -> List[Path]:
    """Find latest dev result files.

    Args:
        methods: List of methods to include (default: all)
        policy: Filter by policy ID
        limit: Max number of results to return

    Returns:
        List of result file paths (newest first)
    """
    methods = methods or ALL_METHODS
    results = []

    if not DEV_DIR.exists():
        return results

    # Get all dev runs sorted by timestamp (newest first)
    dev_runs = sorted(DEV_DIR.iterdir(), reverse=True)

    for run_dir in dev_runs:
        if not run_dir.is_dir():
            continue

        result_file = run_dir / "results.json"
        if not result_file.exists():
            continue

        # Load to check method/policy filters
        try:
            with open(result_file) as f:
                data = json.load(f)

            run_method = data.get("method", "direct")
            run_policy = data.get("policy_id") or data.get("run_id")

            if methods and run_method not in methods:
                continue
            if policy and run_policy != policy:
                continue

            results.append(result_file)
            if len(results) >= limit:
                break
        except (json.JSONDecodeError, OSError):
            continue

    return results


def load_results(path: Path) -> Dict[str, Any]:
    """Load results.json file.

    Args:
        path: Path to results.json or directory containing it

    Returns:
        Parsed results dict
    """
    if path.is_dir():
        path = path / "results.json"
    with open(path) as f:
        return json.load(f)


def print_table(results: Dict[str, Any], console: Console) -> None:
    """Print results as Rich table.

    Args:
        results: Loaded results dict
        console: Rich console for output
    """
    # Header info
    run_id = results.get("run_id") or results.get("policy_id", "Unknown")
    method = results.get("method", "direct")
    k = results.get("k", "?")
    n = results.get("n") or results.get("total", "?")
    timestamp = results.get("timestamp", "")

    console.print()
    console.rule(f"[bold cyan]{run_id}[/] | Method: [yellow]{method}[/] | K={k} | N={n}")
    if timestamp:
        console.print(f"[dim]Timestamp: {timestamp}[/]")
    console.print()

    # Unified scores table
    unified = results.get("unified_scores", {})
    scores_table = Table(title="[bold]Unified Scores[/]", show_header=True, header_style="bold")
    scores_table.add_column("Metric", style="cyan", min_width=15)
    scores_table.add_column("Score", justify="right", min_width=10)
    scores_table.add_column("Status", justify="center", min_width=8)

    score_mappings = [
        ("AUPRC", "auprc"),
        ("Process", "process_score"),
        ("Consistency", "consistency_score"),
    ]

    for name, key in score_mappings:
        val = unified.get(key)
        if val is None:
            scores_table.add_row(name, "N/A", "[dim]-[/]")
            continue

        # Convert to percentage (AUPRC is 0-1, others are 0-100)
        if key == "auprc":
            pct = val * 100
        else:
            pct = val if val > 1 else val * 100

        status = "[green]pass[/]" if pct >= 75 else "[red]FAIL[/]"
        scores_table.add_row(name, f"{pct:.1f}%", status)

    console.print(scores_table)
    console.print()

    # Verdict distribution table
    correct = results.get("correct")
    total = results.get("total")
    accuracy = results.get("accuracy")

    if correct is not None and total is not None:
        console.print(f"[bold]Verdict Accuracy:[/] {correct}/{total} ({accuracy*100:.1f}%)")
        console.print()

    # Process components table
    unified_full = results.get("unified_metrics_full", {})
    components = unified_full.get("process_components", {})

    if components and "error" not in components:
        proc_table = Table(title="[bold]Process Components[/]", show_header=True, header_style="bold")
        proc_table.add_column("Component", style="cyan", min_width=20)
        proc_table.add_column("Score", justify="right", min_width=10)
        proc_table.add_column("Weight", justify="right", min_width=8)

        component_names = [
            ("Incident Precision", "incident_precision"),
            ("Severity Accuracy", "severity_accuracy"),
            ("Modifier Accuracy", "modifier_accuracy"),
            ("Verdict Support", "verdict_support_rate"),
            ("Snippet Validity", "snippet_validity"),
        ]

        for display_name, key in component_names:
            val = components.get(key)
            weight = PROCESS_WEIGHTS.get(key, 0)
            weight_str = f"{int(weight * 100)}%"

            if val is None:
                proc_table.add_row(display_name, "[dim]N/A[/]", weight_str)
            else:
                pct = val * 100 if val <= 1 else val
                color = "green" if pct >= 75 else "yellow" if pct >= 50 else "red"
                proc_table.add_row(display_name, f"[{color}]{pct:.1f}%[/{color}]", weight_str)

        console.print(proc_table)
        console.print()

    # AUPRC breakdown
    auprc_metrics = results.get("auprc", {})
    if auprc_metrics:
        auprc_high = auprc_metrics.get("auprc_ge_high")
        auprc_crit = auprc_metrics.get("auprc_ge_critical")
        n_samples = auprc_metrics.get("n_samples")

        console.print("[bold]AUPRC Breakdown:[/]")
        if auprc_high is not None:
            console.print(f"  >=High: {auprc_high*100:.1f}%")
        if auprc_crit is not None:
            console.print(f"  >=Critical: {auprc_crit*100:.1f}%")
        if n_samples is not None:
            console.print(f"  Samples: {n_samples}")
        console.print()

    # Consistency details
    consistency = unified_full.get("consistency_details", {})
    if consistency:
        cons_total = consistency.get("total", 0)
        cons_match = consistency.get("consistent", 0)
        if cons_total > 0:
            console.print(f"[bold]Consistency:[/] {cons_match}/{cons_total} samples with matching verdict")
            console.print()


def print_comparison_table(
    results_list: List[Dict[str, Any]],
    console: Console,
) -> None:
    """Print comparison table across multiple runs.

    Args:
        results_list: List of loaded results dicts
        console: Rich console for output
    """
    if not results_list:
        console.print("[yellow]No results to compare[/]")
        return

    # Build comparison table
    table = Table(title="[bold]Results Comparison[/]", show_header=True, header_style="bold")
    table.add_column("Policy", style="cyan")
    table.add_column("Method", style="yellow")
    table.add_column("K")
    table.add_column("N")
    table.add_column("AUPRC", justify="right")
    table.add_column("Process", justify="right")
    table.add_column("Consistency", justify="right")
    table.add_column("Accuracy", justify="right")

    for results in results_list:
        policy = results.get("policy_id") or results.get("run_id", "?")
        method = results.get("method", "direct")
        k = str(results.get("k", "?"))
        n = str(results.get("n") or results.get("total", "?"))

        unified = results.get("unified_scores", {})
        auprc = unified.get("auprc")
        process = unified.get("process_score")
        consistency = unified.get("consistency_score")
        accuracy = results.get("accuracy")

        def fmt_score(val, is_pct=False):
            if val is None:
                return "[dim]N/A[/]"
            pct = val * 100 if (not is_pct and val <= 1) else val
            color = "green" if pct >= 75 else "yellow" if pct >= 50 else "red"
            return f"[{color}]{pct:.1f}%[/{color}]"

        table.add_row(
            policy,
            method,
            k,
            n,
            fmt_score(auprc),
            fmt_score(process, is_pct=True),
            fmt_score(consistency, is_pct=True),
            fmt_score(accuracy) if accuracy else "[dim]N/A[/]",
        )

    console.print()
    console.print(table)
    console.print()


def main():
    parser = argparse.ArgumentParser(
        description="Display formatted result summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--method",
        type=str,
        help="Method(s) to show, comma-separated (default: all)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        help="Filter by policy ID",
    )
    parser.add_argument(
        "--dev",
        type=Path,
        nargs="+",
        help="Dev result path(s) instead of benchmark",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=0,
        help="Show N latest dev results (default: 0, uses benchmark)",
    )
    parser.add_argument(
        "--format",
        choices=["table", "comparison", "latex"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all results (don't limit)",
    )

    args = parser.parse_args()
    console = Console()

    # Parse methods
    methods = None
    if args.method:
        methods = [m.strip() for m in args.method.split(",")]

    # Find result files
    paths = []
    if args.dev:
        # Use provided dev paths
        paths = args.dev
    elif args.latest > 0:
        # Find latest dev results
        paths = find_latest_dev_results(
            methods=methods,
            policy=args.policy,
            limit=args.latest if not args.all else 100,
        )
    else:
        # Find benchmark results
        paths = find_benchmark_results(methods=methods, policy=args.policy)

        # If no benchmark results, fall back to latest dev
        if not paths:
            console.print("[dim]No benchmark results found, showing latest dev results...[/]")
            paths = find_latest_dev_results(
                methods=methods,
                policy=args.policy,
                limit=5 if not args.all else 100,
            )

    if not paths:
        console.print("[yellow]No results found[/]")
        console.print()
        console.print("Tips:")
        console.print("  - Use --dev <path> to specify a dev result directory")
        console.print("  - Use --latest N to show N latest dev results")
        console.print("  - Check if results/benchmark/ or results/dev/ exist")
        return

    # Load all results
    results_list = []
    for path in paths:
        try:
            results = load_results(path)
            results["_source_path"] = str(path)
            results_list.append(results)
        except (json.JSONDecodeError, OSError) as e:
            console.print(f"[red]Error loading {path}: {e}[/]")

    if not results_list:
        console.print("[yellow]No valid results loaded[/]")
        return

    # Output based on format
    if args.format == "comparison" or len(results_list) > 1:
        print_comparison_table(results_list, console)
    elif args.format == "table":
        for results in results_list:
            print_table(results, console)
    elif args.format == "latex":
        console.print("[yellow]LaTeX format not yet implemented[/]")
        # Future: generate LaTeX table for papers
        print_comparison_table(results_list, console)


if __name__ == "__main__":
    main()
