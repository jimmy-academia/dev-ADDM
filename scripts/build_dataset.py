"""Build datasets (K=25/50/100/200) from selected restaurant contexts.

Usage:
    # Build from topic-based selection (default)
    python scripts/build_dataset.py --data yelp

    # Build from custom selection
    python scripts/build_dataset.py --data yelp --selection data/selection/yelp/custom.json
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

from addm.data.pipelines import build_datasets, DatasetBuildConfig

DATA_ROOT = Path("data/raw")
DEFAULT_SCALES = [25, 50, 100, 200]


def _parse_scales(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _get_default_paths(data: str) -> Tuple[Path, Path]:
    """Derive default paths from dataset name."""
    return (
        Path(f"data/selection/{data}/topic_100.json"),  # selection
        Path(f"data/context/{data}"),                    # output
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build datasets from selected contexts")
    parser.add_argument("--data", required=True, help="Dataset name (e.g., yelp)")
    parser.add_argument("--selection", default=None,
                        help="Override selection JSON path (default: data/selection/{data}/topic_100.json)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory (default: data/context/{data})")
    parser.add_argument("--scales", default=",".join(map(str, DEFAULT_SCALES)),
                        help="Comma-separated K values (default: 25,50,100,200)")
    parser.add_argument("--no-user", action="store_true", help="Skip user metadata enrichment")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    # Derive paths from --data, allow overrides
    default_selection, default_output = _get_default_paths(args.data)
    selection_path = Path(args.selection) if args.selection else default_selection
    output_dir = Path(args.output_dir) if args.output_dir else default_output
    raw_dir = DATA_ROOT / args.data

    config = DatasetBuildConfig(
        raw_dir=raw_dir,
        selection_path=selection_path,
        output_dir=output_dir,
        scales=_parse_scales(args.scales),
        include_user=not args.no_user,
        verbose=args.verbose,
    )
    build_datasets(config)


if __name__ == "__main__":
    main()
