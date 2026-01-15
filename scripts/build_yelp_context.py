"""Build context dataset from Yelp raw JSONL files."""

import argparse
from pathlib import Path

from addm.data.pipelines import build_context_dataset_from_dir

DATA_ROOT = Path("data/raw")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Yelp context dataset")
    parser.add_argument("--data", required=True, choices=["yelp"], help="Raw dataset name")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--city", default=None, help="Filter by city")
    parser.add_argument("--category", action="append", default=None, help="Filter by category (repeatable)")
    parser.add_argument("--max-businesses", type=int, default=None, help="Limit businesses")
    parser.add_argument("--max-reviews", type=int, default=20, help="Max reviews per business")
    parser.add_argument("--min-reviews", type=int, default=1, help="Min reviews per business")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    raw_dir = DATA_ROOT / args.data
    build_context_dataset_from_dir(
        raw_dir=raw_dir,
        output_path=Path(args.output),
        city=args.city,
        categories=args.category,
        max_businesses=args.max_businesses,
        max_reviews=args.max_reviews,
        min_reviews=args.min_reviews,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
