"""
CLI: Audit cached judgments in human-readable format.

Usage:
    python -m addm.tasks.cli.audit --task G1a --domain yelp
    python -m addm.tasks.cli.audit --task G1a --domain yelp --restaurant "Aladdin Pizzeria"
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from addm.tasks.base import load_task
from addm.tasks.extraction import JudgmentCache


def print_judgment(judgment: Dict[str, Any], indent: int = 2) -> None:
    """Print a single judgment in readable format."""
    prefix = " " * indent
    review_id = judgment.get("review_id", "unknown")[:8]
    is_allergy = judgment.get("is_allergy_related", False)

    if not is_allergy:
        print(f"{prefix}[{review_id}] Not allergy-related")
        return

    severity = judgment.get("incident_severity", "none")
    account = judgment.get("account_type", "unknown")
    assurance = judgment.get("assurance_claim", "unknown")
    staff = judgment.get("staff_response", "unknown")
    date = judgment.get("date", "")[:10]

    print(f"{prefix}[{review_id}] {date}")
    print(f"{prefix}  severity={severity}, account={account}")
    print(f"{prefix}  assurance={assurance}, staff={staff}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Audit cached judgments")
    parser.add_argument("--task", type=str, required=True, help="Task ID (e.g., G1a)")
    parser.add_argument("--domain", type=str, default="yelp", help="Domain (default: yelp)")
    parser.add_argument("--k", type=int, default=50, help="Dataset K value (default: 50)")
    parser.add_argument("--restaurant", type=str, help="Filter by restaurant name")
    parser.add_argument("--allergy-only", action="store_true", help="Show only allergy-related")

    args = parser.parse_args()

    # Load task config
    task = load_task(args.task, args.domain)

    # Load dataset to get restaurant info
    dataset_path = Path(f"data/context/{args.domain}/dataset_K{args.k}.jsonl")
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    with open(dataset_path) as f:
        restaurants = [json.loads(line) for line in f]

    # Load judgment cache
    cache = JudgmentCache(task.cache_path)
    all_judgments = cache.get_all(task.task_id)
    print(f"Cache has {len(all_judgments)} judgments for {task.task_id}\n")

    # Process restaurants
    for restaurant in restaurants:
        business = restaurant.get("business", {})
        name = business.get("name", "Unknown")

        # Filter by restaurant name if specified
        if args.restaurant and args.restaurant.lower() not in name.lower():
            continue

        # Get judgments for this restaurant
        reviews = restaurant.get("reviews", [])
        review_ids = [r.get("review_id", "") for r in reviews]
        restaurant_judgments = [
            all_judgments[rid]
            for rid in review_ids
            if rid in all_judgments
        ]

        # Filter to allergy-only if requested
        if args.allergy_only:
            restaurant_judgments = [
                j for j in restaurant_judgments
                if j.get("is_allergy_related", False)
            ]

        if not restaurant_judgments:
            continue

        # Count allergy vs non-allergy
        allergy_count = sum(
            1 for j in restaurant_judgments if j.get("is_allergy_related", False)
        )
        total = len(restaurant_judgments)

        print(f"{'='*60}")
        print(f"{name}")
        print(f"  {total} cached judgments ({allergy_count} allergy-related)")
        print(f"{'='*60}")

        for j in restaurant_judgments:
            print_judgment(j)

        print()


if __name__ == "__main__":
    main()
