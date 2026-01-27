#!/usr/bin/env python3
"""Show ground truth verdict distribution for T* policies.

Usage:
    .venv/bin/python scripts/data/show_gt_distribution.py              # All 35 T* policies
    .venv/bin/python scripts/data/show_gt_distribution.py --tier T1    # All T1 variants (P1-P7)
    .venv/bin/python scripts/data/show_gt_distribution.py --policy T1P3  # Single policy
    .venv/bin/python scripts/data/show_gt_distribution.py --variant P3 # All P3 across tiers
    .venv/bin/python scripts/data/show_gt_distribution.py --k 200      # Filter to K=200
    .venv/bin/python scripts/data/show_gt_distribution.py --aggregate  # Show aggregate stats
    .venv/bin/python scripts/data/show_gt_distribution.py --imbalance  # Show imbalance report
    .venv/bin/python scripts/data/show_gt_distribution.py --compare    # Compare P3 (ALL) vs P1 (ANY)
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from addm.tasks.constants import (
    TIERS,
    TIER_TO_GT_TOPIC,
    K_VALUES,
    expand_policies,
)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Constants
GT_DIR = Path(__file__).parent.parent.parent / "data" / "answers" / "yelp"
VARIANTS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]

# Verdict order by tier topic (worst → best for coloring)
VERDICT_ORDER = {
    "T1": ["Critical Risk", "High Risk", "Low Risk"],  # allergy
    "T2": ["Poor Value", "Fair Value", "Good Value"],  # price_worth
    "T3": ["Needs Improvement", "Satisfactory", "Excellent"],  # environment
    "T4": ["Needs Improvement", "Satisfactory", "Excellent"],  # execution
    "T5": ["Needs Improvement", "Satisfactory", "Excellent"],  # server
}

# Variant descriptions
VARIANT_DESC = {
    "P1": "base (ANY)",
    "P2": "extended (ANY)",
    "P3": "ALL logic",
    "P4": "reorder v1",
    "P5": "reorder v2",
    "P6": "XML format",
    "P7": "prose format",
}

console = Console()


def load_gt_file(policy_id: str, k: int) -> dict | None:
    """Load a ground truth file for T* or G* policy."""
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


def get_all_verdicts_for_tier(tier: str) -> list[str]:
    """Get verdict order for a tier (worst → best)."""
    return VERDICT_ORDER.get(tier, ["Critical Risk", "High Risk", "Low Risk"])


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
        return "yellow"  # Middle


def format_cell(count: int, total: int, verdict: str, all_verdicts: list[str]) -> str:
    """Format a cell value with percentage and coloring."""
    if total == 0:
        return "[dim]-[/dim]"

    pct = count / total * 100

    if count == 0:
        return "[on red3][bold white]0[/bold white][/on red3]"

    color = get_verdict_color(verdict, all_verdicts)
    return f"[{color}]{count}[/] [dim]({pct:.0f}%)[/dim]"


def print_tier_summary(tier: str, variants: list[str], k_values: list[int]):
    """Print distribution summary for a tier with verdicts as columns."""
    all_verdicts = get_all_verdicts_for_tier(tier)
    topic = TIER_TO_GT_TOPIC.get(tier, tier)

    table = Table(
        title=f"[bold]{tier}[/bold] ({topic})",
        show_header=True,
        header_style="bold blue"
    )
    table.add_column("Var", style="cyan", width=4)
    table.add_column("Type", style="dim", width=12)
    table.add_column("K", justify="right", width=4)
    table.add_column("N", justify="right", width=4)

    # Add verdict columns
    for verdict in all_verdicts:
        color = get_verdict_color(verdict, all_verdicts)
        table.add_column(verdict, justify="center", style=color, min_width=10)

    for variant in variants:
        policy_id = f"{tier}{variant}"
        first_row = True

        # Determine which GT file to load
        # P3 has its own GT, others use G* GT
        if variant == "P3":
            gt_policy_id = policy_id  # T1P3
        elif variant == "P2":
            gt_policy_id = f"{topic}_V2"  # G1_allergy_V2
        else:
            gt_policy_id = f"{topic}_V1"  # G1_allergy_V1

        for k in k_values:
            gt_data = load_gt_file(gt_policy_id, k)

            row = [
                variant if first_row else "",
                VARIANT_DESC.get(variant, "") if first_row else "",
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
            table.add_row(*[""] * (4 + len(all_verdicts)))

    console.print(table)
    console.print()


def print_aggregate_stats(tiers: list[str], variants: list[str], k_values: list[int]):
    """Print aggregate statistics across all policies."""
    console.print(Panel("[bold]Aggregate Statistics[/bold]", expand=False))

    for k in k_values:
        all_verdicts = defaultdict(int)
        total_restaurants = 0

        for tier in tiers:
            topic = TIER_TO_GT_TOPIC.get(tier, tier)
            for variant in variants:
                # Determine GT file
                if variant == "P3":
                    gt_policy_id = f"{tier}{variant}"
                elif variant == "P2":
                    gt_policy_id = f"{topic}_V2"
                else:
                    gt_policy_id = f"{topic}_V1"

                gt_data = load_gt_file(gt_policy_id, k)
                if gt_data is None:
                    continue

                dist = get_verdict_distribution(gt_data)
                for verdict, count in dist.items():
                    all_verdicts[verdict] += count
                    total_restaurants += count

        table = Table(
            title=f"[bold]K={k}[/bold] (total: {total_restaurants} samples)",
            show_header=True
        )
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


def print_imbalance_report(tiers: list[str], variants: list[str], k_values: list[int]):
    """Print policies with severe class imbalance."""
    console.print(Panel("[bold]Imbalance Report[/bold] (dominant class > 80%)", expand=False))

    table = Table(show_header=True, header_style="bold red")
    table.add_column("Policy", style="cyan", width=12)
    table.add_column("K", justify="right", width=5)
    table.add_column("Dominant", width=18)
    table.add_column("%", justify="right", width=6)
    table.add_column("Other Classes", width=40)

    imbalanced = []
    for tier in tiers:
        topic = TIER_TO_GT_TOPIC.get(tier, tier)
        for variant in variants:
            policy_id = f"{tier}{variant}"

            # Determine GT file
            if variant == "P3":
                gt_policy_id = policy_id
            elif variant == "P2":
                gt_policy_id = f"{topic}_V2"
            else:
                gt_policy_id = f"{topic}_V1"

            for k in k_values:
                gt_data = load_gt_file(gt_policy_id, k)
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
                    others_str = ", ".join(
                        f"{k}:{v}" for k, v in sorted(others.items(), key=lambda x: -x[1])
                    )
                    imbalanced.append((policy_id, k, dominant[0], pct, others_str))

    for policy_id, k, dominant, pct, others in sorted(imbalanced, key=lambda x: -x[3]):
        color = "red" if pct > 90 else "yellow"
        table.add_row(
            policy_id, str(k), dominant,
            f"[{color}]{pct:.0f}%[/]",
            others or "[dim]none[/dim]"
        )

    console.print(table)
    console.print(f"\n[dim]Found {len(imbalanced)} imbalanced policy/K combinations[/dim]")


def print_compare_p3_vs_p1(tiers: list[str], k_values: list[int]):
    """Compare P3 (ALL logic) vs P1 (ANY logic) distributions."""
    console.print(Panel(
        "[bold]P3 (ALL logic) vs P1 (ANY logic) Comparison[/bold]",
        expand=False
    ))

    for tier in tiers:
        topic = TIER_TO_GT_TOPIC.get(tier, tier)
        all_verdicts = get_all_verdicts_for_tier(tier)

        table = Table(
            title=f"[bold]{tier}[/bold] ({topic})",
            show_header=True,
            header_style="bold blue"
        )
        table.add_column("K", justify="right", width=4)
        table.add_column("Logic", style="cyan", width=8)

        for verdict in all_verdicts:
            color = get_verdict_color(verdict, all_verdicts)
            table.add_column(verdict, justify="center", style=color, min_width=10)

        table.add_column("Diff", justify="center", width=15)

        for k in k_values:
            # Load P1 (ANY) - uses G*_V1
            p1_gt = load_gt_file(f"{topic}_V1", k)
            # Load P3 (ALL) - uses T*P3
            p3_gt = load_gt_file(f"{tier}P3", k)

            if p1_gt is None and p3_gt is None:
                continue

            # P1 row
            if p1_gt:
                p1_dist = get_verdict_distribution(p1_gt)
                p1_total = sum(p1_dist.values())
                p1_row = [str(k), "P1 (ANY)"]
                for verdict in all_verdicts:
                    count = p1_dist.get(verdict, 0)
                    p1_row.append(format_cell(count, p1_total, verdict, all_verdicts))
                p1_row.append("")
                table.add_row(*p1_row)
            else:
                table.add_row(str(k), "P1 (ANY)", *["[dim]-[/dim]"] * len(all_verdicts), "")

            # P3 row with diff
            if p3_gt:
                p3_dist = get_verdict_distribution(p3_gt)
                p3_total = sum(p3_dist.values())
                p3_row = ["", "P3 (ALL)"]
                for verdict in all_verdicts:
                    count = p3_dist.get(verdict, 0)
                    p3_row.append(format_cell(count, p3_total, verdict, all_verdicts))

                # Calculate diff (more Low Risk expected for P3)
                if p1_gt:
                    p1_low = p1_dist.get(all_verdicts[-1], 0)  # Last = best (Low Risk)
                    p3_low = p3_dist.get(all_verdicts[-1], 0)
                    diff = p3_low - p1_low
                    if diff > 0:
                        p3_row.append(f"[green]+{diff} {all_verdicts[-1][:4]}[/]")
                    elif diff < 0:
                        p3_row.append(f"[red]{diff} {all_verdicts[-1][:4]}[/]")
                    else:
                        p3_row.append("[dim]same[/dim]")
                else:
                    p3_row.append("")

                table.add_row(*p3_row)
            else:
                table.add_row("", "P3 (ALL)", *["[dim]-[/dim]"] * len(all_verdicts), "")

            # Separator between K values
            if k != k_values[-1]:
                table.add_row(*[""] * (3 + len(all_verdicts)))

        console.print(table)
        console.print()


def main():
    parser = argparse.ArgumentParser(description="Show T* ground truth verdict distribution")

    # Target selection
    parser.add_argument(
        "--tier",
        type=str,
        help="Tier ID (e.g., T1) - shows all 7 variants for that tier",
    )
    parser.add_argument(
        "--policy",
        type=str,
        help="Policy ID (e.g., T1P3) - shows just that policy's tier",
    )

    # Additional filters
    parser.add_argument("--variant", help="Filter to specific variant (e.g., P3)")
    parser.add_argument("--k", type=int, help="Filter to specific K value")

    # Output modes
    parser.add_argument("--aggregate", action="store_true", help="Show aggregate stats only")
    parser.add_argument("--imbalance", action="store_true", help="Show imbalance report only")
    parser.add_argument("--compare", action="store_true", help="Compare P3 (ALL) vs P1 (ANY)")

    args = parser.parse_args()

    # Determine tiers
    if args.policy:
        # Extract tier from policy ID (T1P3 -> T1)
        if args.policy[0] == "T" and args.policy[1].isdigit():
            tiers = [args.policy[:2]]
        else:
            console.print(f"[red]Error: Invalid T* policy format: {args.policy}[/red]")
            return
    elif args.tier:
        tier_upper = args.tier.upper()
        if tier_upper not in TIERS:
            console.print(f"[red]Error: Unknown tier '{args.tier}'. Valid: {TIERS}[/red]")
            return
        tiers = [tier_upper]
    else:
        tiers = TIERS

    # Apply variant filter
    variants = [args.variant.upper()] if args.variant else VARIANTS
    if args.variant and args.variant.upper() not in VARIANTS:
        console.print(f"[yellow]Warning: Unknown variant '{args.variant}'[/yellow]")

    # Apply K filter
    k_values = [args.k] if args.k else K_VALUES
    if args.k and args.k not in K_VALUES:
        console.print(f"[yellow]Warning: K={args.k} not in standard values {K_VALUES}[/yellow]")

    # Run selected mode
    if args.compare:
        print_compare_p3_vs_p1(tiers, k_values)
    elif args.imbalance:
        print_imbalance_report(tiers, variants, k_values)
    elif args.aggregate:
        print_aggregate_stats(tiers, variants, k_values)
    else:
        for tier in tiers:
            print_tier_summary(tier, variants, k_values)


if __name__ == "__main__":
    main()
