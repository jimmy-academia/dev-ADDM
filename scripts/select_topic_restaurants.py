#!/usr/bin/env python3
"""CLI for topic-aware restaurant selection.

This script selects restaurants for the ADDM benchmark based on their
coverage across topics identified by keyword search.

Outputs:
  - topic_ranked_all.json: Full ranked list of all ~1,200 restaurants
  - topic_N.json: Balanced selection of N restaurants (default 100)

Examples:
    # Default selection (100 restaurants, balanced)
    python scripts/select_topic_restaurants.py

    # Select 150 restaurants
    python scripts/select_topic_restaurants.py --target 150

    # Unbalanced selection (just top N by score)
    python scripts/select_topic_restaurants.py --no-balance
"""

import argparse
from pathlib import Path

from addm.data.pipelines.topic_selection import TopicSelectionConfig, run_selection


def main():
    parser = argparse.ArgumentParser(
        description="Select restaurants based on topic coverage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--keyword-hits-dir",
        type=Path,
        default=Path("data/keyword_hits/yelp"),
        help="Directory containing G*_*.json keyword hit files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/selected/yelp"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=100,
        help="Number of restaurants to select (default: 100)",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Skip quadrant balancing, just take top N by score",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    config = TopicSelectionConfig(
        keyword_hits_dir=args.keyword_hits_dir,
        output_dir=args.output_dir,
        target_count=args.target,
        balanced=not args.no_balance,
        seed=args.seed,
    )

    ranked_path, selected_path = run_selection(config, verbose=not args.quiet)

    if not args.quiet:
        print(f"\nOutput files:")
        print(f"  Full ranking: {ranked_path}")
        print(f"  Selected {args.target}: {selected_path}")


if __name__ == "__main__":
    main()
