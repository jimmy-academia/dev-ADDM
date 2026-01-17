#!/usr/bin/env python3
"""
CLI: Search for restaurants with keyword hits by topic.

Usage:
    # Search single topic
    python scripts/search_restaurants.py --data yelp --topic allergy

    # Search all G1 topics
    python scripts/search_restaurants.py --data yelp --group G1

    # Search all topics
    python scripts/search_restaurants.py --data yelp --all

    # Override paths for custom dataset
    python scripts/search_restaurants.py --data custom \\
        --review-file data/raw/custom/reviews.json \\
        --business-file data/raw/custom/business.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console

from addm.data.keyword_search import (
    TOPIC_KEYWORDS,
    search_and_rank_restaurants,
    print_topic_summary,
)

console = Console()

# Group to topics mapping
GROUP_TOPICS = {
    "G1": ["allergy", "dietary", "hygiene"],
    "G2": ["romance", "business", "group"],
    "G3": ["price_worth", "hidden_costs", "time_value"],
    "G4": ["server", "kitchen", "environment"],
    "G5": ["capacity", "execution", "consistency"],
    "G6": ["uniqueness", "comparison", "loyalty"],
}

# Reverse mapping: topic to group
TOPIC_TO_GROUP = {
    topic: group
    for group, topics in GROUP_TOPICS.items()
    for topic in topics
}


def main():
    parser = argparse.ArgumentParser(description="Search restaurants by keyword hits")
    parser.add_argument("--data", required=True, help="Dataset name (e.g., yelp)")
    parser.add_argument("--topic", type=str, help="Single topic to search")
    parser.add_argument("--group", type=str, choices=GROUP_TOPICS.keys(), help="Task group (G1-G6)")
    parser.add_argument("--all", action="store_true", help="Search all topics")
    parser.add_argument("--min-hits", type=int, default=10, help="Minimum keyword hits (default: 10)")
    parser.add_argument("--min-reviews", type=int, default=20, help="Minimum reviews per restaurant (default: 20)")
    parser.add_argument("--max-reviews", type=int, default=500, help="Max reviews per restaurant (default: 500)")
    parser.add_argument("--top-n", type=int, default=100, help="Top N restaurants per topic (default: 100)")
    parser.add_argument("--output", type=str, help="Output directory (default: data/keyword_hits/{data}/)")
    parser.add_argument("--review-file", type=str, default=None,
                        help="Override review JSONL path")
    parser.add_argument("--business-file", type=str, default=None,
                        help="Override business JSONL path")

    args = parser.parse_args()

    # Derive paths from --data, allow overrides
    data_root = Path("data")
    review_file = args.review_file or str(
        data_root / "raw" / args.data / f"{args.data}_academic_dataset_review.json"
    )
    business_file = args.business_file or str(
        data_root / "raw" / args.data / f"{args.data}_academic_dataset_business.json"
    )

    # Determine topics to search
    if args.all:
        topics = list(TOPIC_KEYWORDS.keys())
    elif args.group:
        topics = GROUP_TOPICS[args.group]
    elif args.topic:
        if args.topic not in TOPIC_KEYWORDS:
            console.print(f"[red]Unknown topic:[/red] {args.topic}")
            console.print(f"[dim]Available:[/dim] {list(TOPIC_KEYWORDS.keys())}")
            sys.exit(1)
        topics = [args.topic]
    else:
        console.print("[red]Specify --topic, --group, or --all[/red]")
        parser.print_help()
        sys.exit(1)

    # Output directory - default to data/keyword_hits/{data}/
    output_dir = Path(args.output) if args.output else (data_root / "keyword_hits" / args.data)

    console.print(f"[bold]Searching for topics:[/bold] {topics}")
    console.print(f"[bold]Min hits:[/bold] {args.min_hits}, [bold]Min reviews:[/bold] {args.min_reviews}, [bold]Max reviews:[/bold] {args.max_reviews}")
    console.print(f"[bold]Top N:[/bold] {args.top_n}, [bold]Output:[/bold] {output_dir}")
    console.print("=" * 70)

    # Run search (saves partial results during scan)
    results = search_and_rank_restaurants(
        topics=topics,
        review_file=review_file,
        business_file=business_file,
        min_hits=args.min_hits,
        min_reviews=args.min_reviews,
        max_reviews_per_biz=args.max_reviews,
        top_n=args.top_n,
        output_dir=output_dir,
    )

    # Print summary
    print_topic_summary(results)

    # Save each topic separately with group prefix (e.g., G1_allergy.json)
    for topic, restaurants in results.items():
        group = TOPIC_TO_GROUP[topic]
        output_file = output_dir / f"{group}_{topic}.json"
        with open(output_file, "w") as f:
            json.dump(restaurants, f, indent=2)
        console.print(f"[green]Saved[/green] {group}_{topic}: {len(restaurants)} restaurants -> {output_file}")

    # Print totals
    console.print("\n" + "=" * 70)
    console.print("[bold]TOTALS:[/bold]")
    for topic, restaurants in results.items():
        console.print(f"  [cyan]{topic}[/cyan]: {len(restaurants)} restaurants")
    console.print(f"\n[bold]Output directory:[/bold] {output_dir}")


if __name__ == "__main__":
    main()
