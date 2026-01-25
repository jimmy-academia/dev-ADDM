#!/usr/bin/env python3
"""Show ground truth verdict distribution across all K values.

Usage:
    .venv/bin/python scripts/data/show_gt_distribution.py
    .venv/bin/python scripts/data/show_gt_distribution.py --topic G1_allergy
    .venv/bin/python scripts/data/show_gt_distribution.py --group G1
    .venv/bin/python scripts/data/show_gt_distribution.py --policy G1_allergy_V2
    .venv/bin/python scripts/data/show_gt_distribution.py --variant V2
    .venv/bin/python scripts/data/show_gt_distribution.py --k 200
    .venv/bin/python scripts/data/show_gt_distribution.py --aggregate
    .venv/bin/python scripts/data/show_gt_distribution.py --imbalance
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from addm.tasks.constants import (
    ALL_TOPICS,
    K_VALUES,
    expand_topics,
    get_topic_from_policy_id,
)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Constants
GT_DIR = Path(__file__).parent.parent.parent / "data" / "answers" / "yelp"
VARIANTS = ["V1", "V2", "V3", "V4"]

# Verdict order: worst → best (for coloring: red, yellow, green)
VERDICT_ORDER = {
    "G1_allergy": ["Critical Risk", "High Risk", "Low Risk"],
    "G1_dietary": ["Critical Risk", "High Risk", "Low Risk"],
    "G1_hygiene": ["Critical Risk", "High Risk", "Low Risk"],
    "G2_romance": ["Not Recommended", "Acceptable", "Recommended"],
    "G2_business": ["Not Recommended", "Acceptable", "Recommended"],
    "G2_group": ["Not Recommended", "Acceptable", "Recommended"],
    "G3_price_worth": ["Poor Value", "Fair Value", "Good Value"],
    "G3_hidden_costs": ["Critical Risk", "High Risk", "Low Risk"],
    "G3_time_value": ["Not Worth Waiting", "Worth Waiting", "Definitely Worth Waiting"],
    "G4_server": ["Needs Improvement", "Satisfactory", "Excellent"],
    "G4_kitchen": ["Needs Improvement", "Satisfactory", "Excellent"],
    "G4_environment": ["Needs Improvement", "Satisfactory", "Excellent"],
    "G5_capacity": ["Needs Improvement", "Satisfactory", "Excellent"],
    "G5_execution": ["Needs Improvement", "Satisfactory", "Excellent"],
    "G5_consistency": ["Needs Improvement", "Satisfactory", "Excellent"],
    "G6_uniqueness": ["Generic", "Differentiated", "Highly Unique"],
    "G6_comparison": ["Weaker", "Comparable", "Stronger"],
    "G6_loyalty": ["Low Loyalty", "Moderate Loyalty", "High Loyalty"],
}

console = Console()


def load_gt_file(policy_id: str, k: int) -> dict | None:
    """Load a ground truth file."""
    path = GT_DIR / f"{policy_id}_K{k}_groundtruth.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_verdict_distribution(gt_data: dict) -> dict[str, int]:
    """Extract verdict counts from GT data."""
    counts = defaultdict(int)
    for restaurant in gt_data.get("restaurants", {}).values():
        verdict = restaurant.get("ground_truth", {}).get("verdict", "UNKNOWN")
        counts[verdict] += 1
    return dict(counts)


def get_all_verdicts_for_topic(topic: str, variants: list[str], k_values: list[int]) -> list[str]:
    """Get all unique verdicts for a topic, in policy-defined order (worst → best)."""
    # Use predefined order if available
    if topic in VERDICT_ORDER:
        return VERDICT_ORDER[topic]

    # Fallback: collect from data and sort alphabetically
    verdicts = set()
    for variant in variants:
        policy_id = f"{topic}_{variant}"
        for k in k_values:
            gt_data = load_gt_file(policy_id, k)
            if gt_data:
                dist = get_verdict_distribution(gt_data)
                verdicts.update(dist.keys())
    return sorted(verdicts)


def get_verdict_color(verdict: str, all_verdicts: list[str]) -> str:
    """Get color based on verdict position. First=worst(red), last=best(green)."""
    if verdict not in all_verdicts:
        return "cyan"
    idx = all_verdicts.index(verdict)
    n = len(all_verdicts)
    if n == 1:
        return "cyan"
    elif idx == 0:
        return "red"    # First = worst
    elif idx == n - 1:
        return "green"  # Last = best
    else:
        return "yellow" # Middle


def format_cell(count: int, total: int, verdict: str, all_verdicts: list[str]) -> str:
    """Format a cell value with percentage and coloring."""
    if total == 0:
        return "[dim]-[/dim]"

    pct = count / total * 100

    if count == 0:
        # Highlight 0% cells with background
        return "[on red3][bold white]0[/bold white][/on red3]"

    color = get_verdict_color(verdict, all_verdicts)
    return f"[{color}]{count}[/] [dim]({pct:.0f}%)[/dim]"


def print_topic_summary(topic: str, variants: list[str], k_values: list[int]):
    """Print distribution summary for a topic with verdicts as columns."""
    # Get all verdicts for this topic
    all_verdicts = get_all_verdicts_for_topic(topic, variants, k_values)

    if not all_verdicts:
        console.print(f"[dim]No data for {topic}[/dim]")
        return

    table = Table(title=f"[bold]{topic}[/bold]", show_header=True, header_style="bold blue")
    table.add_column("Var", style="cyan", width=4)
    table.add_column("K", justify="right", width=4)
    table.add_column("N", justify="right", width=4)

    # Add verdict columns
    for verdict in all_verdicts:
        color = get_verdict_color(verdict, all_verdicts)
        table.add_column(verdict, justify="center", style=color, min_width=8)

    for variant in variants:
        policy_id = f"{topic}_{variant}"
        first_row = True

        for k in k_values:
            gt_data = load_gt_file(policy_id, k)

            row = [
                variant if first_row else "",
                str(k),
            ]

            if gt_data is None:
                row.append("-")
                row.extend(["[dim]-[/dim]"] * len(all_verdicts))
            else:
                dist = get_verdict_distribution(gt_data)
                total = sum(dist.values())
                row.append(str(total))

                for verdict in all_verdicts:
                    count = dist.get(verdict, 0)
                    row.append(format_cell(count, total, verdict, all_verdicts))

            table.add_row(*row)
            first_row = False

        # Add separator between variants
        if variant != variants[-1]:
            table.add_row(*[""] * (3 + len(all_verdicts)))

    console.print(table)
    console.print()


def print_aggregate_stats(topics: list[str], variants: list[str], k_values: list[int]):
    """Print aggregate statistics across all policies."""
    console.print(Panel("[bold]Aggregate Statistics[/bold]", expand=False))

    for k in k_values:
        all_verdicts = defaultdict(int)
        total_restaurants = 0

        for topic in topics:
            for variant in variants:
                policy_id = f"{topic}_{variant}"
                gt_data = load_gt_file(policy_id, k)
                if gt_data is None:
                    continue

                dist = get_verdict_distribution(gt_data)
                for verdict, count in dist.items():
                    all_verdicts[verdict] += count
                    total_restaurants += count

        table = Table(title=f"[bold]K={k}[/bold] (total: {total_restaurants} samples)", show_header=True)
        table.add_column("Verdict", style="cyan", width=25)
        table.add_column("Count", justify="right", width=8)
        table.add_column("Percentage", justify="right", width=10)
        table.add_column("Bar", width=30)

        for verdict, count in sorted(all_verdicts.items(), key=lambda x: -x[1]):
            pct = count / total_restaurants * 100 if total_restaurants > 0 else 0
            bar_len = int(pct / 100 * 25)
            bar = f"[cyan]{'█' * bar_len}[/][dim]{'░' * (25 - bar_len)}[/dim]"
            table.add_row(verdict, str(count), f"{pct:.1f}%", bar)

        console.print(table)
        console.print()


def print_imbalance_report(topics: list[str], variants: list[str], k_values: list[int]):
    """Print policies with severe class imbalance."""
    console.print(Panel("[bold]Imbalance Report[/bold] (dominant class > 80%)", expand=False))

    table = Table(show_header=True, header_style="bold red")
    table.add_column("Policy", style="cyan", width=20)
    table.add_column("K", justify="right", width=5)
    table.add_column("Dominant", width=18)
    table.add_column("%", justify="right", width=6)
    table.add_column("Other Classes", width=40)

    imbalanced = []
    for topic in topics:
        for variant in variants:
            policy_id = f"{topic}_{variant}"
            for k in k_values:
                gt_data = load_gt_file(policy_id, k)
                if gt_data is None:
                    continue

                dist = get_verdict_distribution(gt_data)
                total = sum(dist.values())
                if total == 0:
                    continue

                dominant = max(dist.items(), key=lambda x: x[1])
                pct = dominant[1] / total * 100

                if pct > 80:
                    others = {k: v for k, v in dist.items() if k != dominant[0]}
                    others_str = ", ".join(f"{k}:{v}" for k, v in sorted(others.items(), key=lambda x: -x[1]))
                    imbalanced.append((policy_id, k, dominant[0], pct, others_str))

    for policy_id, k, dominant, pct, others in sorted(imbalanced, key=lambda x: -x[3]):
        color = "red" if pct > 90 else "yellow"
        table.add_row(policy_id, str(k), dominant, f"[{color}]{pct:.0f}%[/]", others or "[dim]none[/dim]")

    console.print(table)
    console.print(f"\n[dim]Found {len(imbalanced)} imbalanced policy/K combinations[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Show ground truth verdict distribution")

    # Target selection (mutually exclusive)
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--group",
        type=str,
        help="Group ID (e.g., G1) - shows all 3 topics in group",
    )
    target_group.add_argument(
        "--topic",
        type=str,
        help="Topic ID (e.g., G1_allergy)",
    )
    target_group.add_argument(
        "--policy",
        type=str,
        help="Policy ID (e.g., G1_allergy_V2) - shows just that policy's topic",
    )

    # Additional filters
    parser.add_argument("--variant", help="Filter to specific variant (e.g., V2)")
    parser.add_argument("--k", type=int, help="Filter to specific K value")

    # Output modes
    parser.add_argument("--aggregate", action="store_true", help="Show aggregate stats only")
    parser.add_argument("--imbalance", action="store_true", help="Show imbalance report only")

    args = parser.parse_args()

    # Determine topics using expand_topics
    try:
        if args.policy:
            # Extract topic from policy ID
            topic = get_topic_from_policy_id(args.policy)
            topics = [topic]
        else:
            topics = expand_topics(topic=args.topic, group=args.group)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return

    # Apply variant filter
    variants = [args.variant] if args.variant else VARIANTS
    if args.variant and args.variant not in VARIANTS:
        console.print(f"[yellow]Warning: Unknown variant '{args.variant}'[/yellow]")

    # Apply K filter
    k_values = [args.k] if args.k else K_VALUES
    if args.k and args.k not in K_VALUES:
        console.print(f"[yellow]Warning: K={args.k} not in standard values {K_VALUES}[/yellow]")

    # Run selected mode
    if args.imbalance:
        print_imbalance_report(topics, variants, k_values)
    elif args.aggregate:
        print_aggregate_stats(topics, variants, k_values)
    else:
        for topic in topics:
            print_topic_summary(topic, variants, k_values)


if __name__ == "__main__":
    main()
