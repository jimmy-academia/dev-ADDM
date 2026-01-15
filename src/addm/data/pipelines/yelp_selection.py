"""Stratified selection of core Yelp businesses."""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


TARGET_CATEGORIES = [
    "Italian",
    "Coffee & Tea",
    "Pizza",
    "Steakhouses",
    "Chinese",
]


@dataclass
class SelectionConfig:
    raw_dir: Path
    output_path: Path
    target_city: str = "Philadelphia"
    min_reviews: int = 50
    categories: Optional[List[str]] = None
    per_category: int = 20
    seed: int = 42


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_candidates(business_path: Path, city: str, categories: List[str], min_reviews: int) -> List[Dict]:
    candidates = []
    for row in _iter_jsonl(business_path):
        if row.get("city", "").lower() != city.lower():
            continue
        cat_list = (row.get("categories") or "").split(", ")
        if not any(c in cat_list for c in categories):
            continue
        if int(row.get("review_count", 0)) < min_reviews:
            continue
        row["category_list"] = cat_list
        candidates.append(row)
    return candidates


def stratify_by_quadrant(candidates: List[Dict], cuisine: str, n_target: int) -> List[Dict]:
    subset = [b for b in candidates if cuisine in b.get("category_list", [])]
    if not subset:
        return []
    reviews = [b.get("review_count", 0) for b in subset]
    stars = [b.get("stars", 0) for b in subset]
    med_rev = statistics.median(reviews)
    med_stars = statistics.median(stars)

    quadrants: Dict[str, List[Dict]] = {
        "HighVol_HighStars": [],
        "HighVol_LowStars": [],
        "LowVol_HighStars": [],
        "LowVol_LowStars": [],
    }
    for b in subset:
        is_high_vol = b.get("review_count", 0) > med_rev
        is_high_star = b.get("stars", 0) > med_stars
        if is_high_vol and is_high_star:
            key = "HighVol_HighStars"
        elif is_high_vol and not is_high_star:
            key = "HighVol_LowStars"
        elif not is_high_vol and is_high_star:
            key = "LowVol_HighStars"
        else:
            key = "LowVol_LowStars"
        quadrants[key].append(b)

    n_per_quad = max(1, n_target // 4)
    selected = []
    for bucket in quadrants.values():
        if not bucket:
            continue
        sample = random.sample(bucket, min(len(bucket), n_per_quad))
        selected.extend(sample)
    return selected


def select_core_100(config: SelectionConfig, verbose: bool = True) -> List[Dict]:
    categories = config.categories or TARGET_CATEGORIES
    business_path = config.raw_dir / "yelp_academic_dataset_business.json"
    if not business_path.exists():
        raise FileNotFoundError(f"Missing business file: {business_path}")

    if verbose:
        print(f"Scanning {business_path} for {config.target_city}...")

    candidates = load_candidates(business_path, config.target_city, categories, config.min_reviews)
    if not candidates:
        if verbose:
            print("No candidates found.")
        return []

    random.seed(config.seed)
    core = []
    seen = set()

    for cuisine in categories:
        if verbose:
            print(f"Selecting for: {cuisine}")
        selection = stratify_by_quadrant(candidates, cuisine, config.per_category)
        added = 0
        for row in selection:
            bid = row.get("business_id")
            if bid in seen:
                continue
            row["stratification_tag"] = cuisine
            core.append(row)
            seen.add(bid)
            added += 1
        if verbose:
            print(f"  -> Added {added} unique contexts")

    target_total = config.per_category * len(categories)
    if len(core) < target_total:
        needed = target_total - len(core)
        if verbose:
            print(f"Need {needed} more to reach {target_total}...")
        pool = [b for b in candidates if b.get("business_id") not in seen]
        if pool:
            core.extend(random.sample(pool, min(len(pool), needed)))

    return core


def write_selection(output_path: Path, items: List[Dict], verbose: bool = True) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(items, handle, ensure_ascii=True, indent=2)
    if verbose:
        print(f"Saved {len(items)} contexts to {output_path}")


def run_selection(config: SelectionConfig, verbose: bool = True) -> Path:
    items = select_core_100(config, verbose=verbose)
    write_selection(config.output_path, items, verbose=verbose)
    return config.output_path
