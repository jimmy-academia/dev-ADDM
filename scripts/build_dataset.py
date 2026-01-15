"""Build Yelp datasets (K=25/50/100/200) from selected contexts."""

import argparse
from pathlib import Path
from typing import List, Optional

from addm.data.pipelines import build_datasets, DatasetBuildConfig

DATA_ROOT = Path("data/raw")
DEFAULT_SELECTION = Path("data/processed/yelp/core_100.json")
DEFAULT_OUTPUT_DIR = Path("data/processed/yelp")
DEFAULT_SCALES = [25, 50, 100, 200]


def _parse_scales(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Yelp datasets from selected contexts")
    parser.add_argument("--data", required=True, choices=["yelp"], help="Raw dataset name")
    parser.add_argument("--selection", default=str(DEFAULT_SELECTION), help="Selection JSON path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--scales", default=",".join(map(str, DEFAULT_SCALES)), help="Comma-separated K values")
    parser.add_argument("--no-user", action="store_true", help="Skip user metadata enrichment")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    raw_dir = DATA_ROOT / args.data
    config = DatasetBuildConfig(
        raw_dir=raw_dir,
        selection_path=Path(args.selection),
        output_dir=Path(args.output_dir),
        scales=_parse_scales(args.scales),
        include_user=not args.no_user,
        verbose=args.verbose,
    )
    build_datasets(config)


if __name__ == "__main__":
    main()
