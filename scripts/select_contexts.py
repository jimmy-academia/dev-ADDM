"""Select stratified core contexts from raw datasets."""

import argparse
from pathlib import Path
from typing import List, Optional

from addm.data.pipelines import run_selection, SelectionConfig

DATA_ROOT = Path("data/raw")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select core contexts")
    parser.add_argument("--data", required=True, choices=["yelp"], help="Raw dataset name")
    parser.add_argument("--output", default=None, help="Output JSON path (optional)")
    parser.add_argument("--city", default="Philadelphia", help="Target city")
    parser.add_argument("--min-reviews", type=int, default=50, help="Minimum review count")
    parser.add_argument("--category", action="append", default=None, help="Target category (repeatable)")
    parser.add_argument("--per-category", type=int, default=20, help="Samples per category")
    parser.add_argument("--total", type=int, default=100, help="Total businesses to select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    raw_dir = DATA_ROOT / args.data
    total = args.total
    categories = args.category
    if categories is None:
        per_category = args.per_category
    else:
        per_category = max(1, total // len(categories))

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("data/processed") / args.data / f"core_{total}.json"

    config = SelectionConfig(
        raw_dir=raw_dir,
        output_path=output_path,
        target_city=args.city,
        min_reviews=args.min_reviews,
        categories=categories,
        per_category=per_category,
        seed=args.seed,
    )
    run_selection(config, verbose=args.verbose)


if __name__ == "__main__":
    main()
