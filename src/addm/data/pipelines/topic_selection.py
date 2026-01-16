"""Topic-aware selection of restaurants for ADDM benchmark.

Produces:
  1. Full ranked list (all ~1,200 restaurants) for expandability
  2. Balanced selection (default 100) with quadrant stratification

The selection prioritizes restaurants that appear across multiple topics/groups,
ensuring good coverage for the benchmark evaluation tasks.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import math
import random
from statistics import median
from collections import defaultdict
from typing import Any


@dataclass
class TopicSelectionConfig:
    """Configuration for topic-aware restaurant selection."""

    keyword_hits_dir: Path
    output_dir: Path
    target_count: int = 100
    balanced: bool = True  # Balance across quadrants
    seed: int = 42


# Topic groups for computing group coverage
GROUPS = {
    "G1": ["G1_allergy", "G1_dietary", "G1_hygiene"],
    "G2": ["G2_romance", "G2_business", "G2_group"],
    "G3": ["G3_price_worth", "G3_hidden_costs", "G3_time_value"],
    "G4": ["G4_server", "G4_kitchen", "G4_environment"],
    "G5": ["G5_capacity", "G5_execution", "G5_consistency"],
    "G6": ["G6_uniqueness", "G6_comparison", "G6_loyalty"],
}


def load_keyword_hits(keyword_hits_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all keyword hit files and merge by business_id.

    Args:
        keyword_hits_dir: Directory containing G*_*.json files

    Returns:
        Dict mapping business_id to restaurant data with topic coverage
    """
    restaurants: dict[str, dict[str, Any]] = {}

    for topic_file in sorted(keyword_hits_dir.glob("G*_*.json")):
        topic = topic_file.stem  # e.g., "G1_allergy"

        with open(topic_file) as f:
            topic_data = json.load(f)

        for r in topic_data:
            biz_id = r["business_id"]
            if biz_id not in restaurants:
                restaurants[biz_id] = {
                    "business_id": biz_id,
                    "name": r["name"],
                    "city": r["city"],
                    "state": r["state"],
                    "stars": r["stars"],
                    "review_count": r["review_count"],
                    "categories": r["categories"],
                    "topic_coverage": {},
                }
            # Record keyword hits for this topic
            restaurants[biz_id]["topic_coverage"][topic] = r["keyword_hits"]

    return restaurants


def compute_group_coverage(topics: dict[str, int]) -> int:
    """Count groups where restaurant appears in 2+ topics.

    Args:
        topics: Dict mapping topic names to keyword hit counts

    Returns:
        Number of groups with strong coverage (2+ topics)
    """
    topic_names = set(topics.keys())
    count = 0
    for group_topics in GROUPS.values():
        covered = len(topic_names & set(group_topics))
        if covered >= 2:
            count += 1
    return count


def rank_all_restaurants(restaurants: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Score, assign quadrants, and rank all restaurants.

    Scoring formula:
        score = (topic_count * 100) + (group_coverage * 20) + log(total_hits + 1)

    Args:
        restaurants: Dict mapping business_id to restaurant data

    Returns:
        List of restaurants sorted by score (descending), with ranks assigned
    """
    # Compute derived metrics
    for data in restaurants.values():
        data["topic_count"] = len(data["topic_coverage"])
        data["group_coverage"] = compute_group_coverage(data["topic_coverage"])
        data["total_hits"] = sum(data["topic_coverage"].values())

        # Compute selection score
        data["selection_score"] = (
            data["topic_count"] * 100
            + data["group_coverage"] * 20
            + math.log(data["total_hits"] + 1)
        )

    # Compute quadrant thresholds (median of ALL restaurants)
    all_stars = [r["stars"] for r in restaurants.values()]
    all_volumes = [r["review_count"] for r in restaurants.values()]
    stars_median = median(all_stars)
    volume_median = median(all_volumes)

    # Assign quadrant to each restaurant
    for data in restaurants.values():
        high_stars = data["stars"] > stars_median
        high_volume = data["review_count"] > volume_median
        vol_label = "HighVol" if high_volume else "LowVol"
        stars_label = "HighStars" if high_stars else "LowStars"
        data["quadrant"] = f"{vol_label}_{stars_label}"

    # Sort by score descending, assign global rank
    ranked = sorted(restaurants.values(), key=lambda x: -x["selection_score"])
    for i, r in enumerate(ranked):
        r["global_rank"] = i + 1

    return ranked


def select_top_n(
    ranked_list: list[dict[str, Any]],
    target: int,
    balanced: bool = True,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select top N restaurants with optional quadrant balancing.

    Args:
        ranked_list: Full ranked list of restaurants
        target: Number of restaurants to select
        balanced: If True, balance selection across quadrants (~N/4 each)
        seed: Random seed for reproducibility

    Returns:
        Selected restaurants sorted by global rank
    """
    if not balanced:
        return ranked_list[:target]

    random.seed(seed)

    # Group by quadrant
    quadrants: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ranked_list:
        quadrants[r["quadrant"]].append(r)

    # Select ~N/4 from each quadrant
    per_quadrant = target // 4
    quadrant_order = [
        "HighVol_HighStars",
        "HighVol_LowStars",
        "LowVol_HighStars",
        "LowVol_LowStars",
    ]

    selected = []
    for q in quadrant_order:
        # Take top candidates from this quadrant (already sorted by score)
        candidates = quadrants[q][:per_quadrant * 2]  # Take 2x for shuffling
        random.shuffle(candidates)
        selected.extend(candidates[:per_quadrant])

    # Fill remaining slots from top-ranked not yet selected
    selected_ids = {r["business_id"] for r in selected}
    for r in ranked_list:
        if len(selected) >= target:
            break
        if r["business_id"] not in selected_ids:
            selected.append(r)

    # Sort by global rank for output
    return sorted(selected, key=lambda x: x["global_rank"])


def run_selection(
    config: TopicSelectionConfig, verbose: bool = True
) -> tuple[Path, Path]:
    """Run full topic-aware selection pipeline.

    Args:
        config: Selection configuration
        verbose: If True, print progress information

    Returns:
        Tuple of (ranked_all_path, selected_path)
    """
    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all keyword hits
    if verbose:
        print(f"Loading keyword hits from {config.keyword_hits_dir}...")
    restaurants = load_keyword_hits(config.keyword_hits_dir)
    if verbose:
        print(f"  Found {len(restaurants)} unique restaurants")

    # Rank all restaurants
    if verbose:
        print("Ranking restaurants by topic coverage...")
    ranked = rank_all_restaurants(restaurants)

    # Show distribution
    if verbose:
        from collections import Counter

        topic_dist = Counter(r["topic_count"] for r in ranked)
        print(f"  Topic coverage distribution: {dict(sorted(topic_dist.items()))}")

        quad_dist = Counter(r["quadrant"] for r in ranked)
        print(f"  Quadrant distribution: {dict(quad_dist)}")

    # Save full ranked list
    ranked_all_path = config.output_dir / "topic_ranked_all.json"
    with open(ranked_all_path, "w") as f:
        json.dump(ranked, f, indent=2)
    if verbose:
        print(f"  Saved full ranking to {ranked_all_path}")

    # Select top N
    if verbose:
        balance_str = "balanced" if config.balanced else "unbalanced"
        print(f"Selecting top {config.target_count} ({balance_str})...")
    selected = select_top_n(
        ranked,
        target=config.target_count,
        balanced=config.balanced,
        seed=config.seed,
    )

    # Show selected distribution
    if verbose:
        from collections import Counter

        sel_quad = Counter(r["quadrant"] for r in selected)
        print(f"  Selected quadrant distribution: {dict(sel_quad)}")

    # Save selected list
    selected_path = config.output_dir / f"topic_{config.target_count}.json"
    with open(selected_path, "w") as f:
        json.dump(selected, f, indent=2)
    if verbose:
        print(f"  Saved selection to {selected_path}")

    # Summary statistics
    if verbose:
        print("\nSelection summary:")
        print(f"  Total ranked: {len(ranked)}")
        print(f"  Selected: {len(selected)}")
        print(f"  Top selected rank: {selected[0]['global_rank']}")
        print(f"  Bottom selected rank: {selected[-1]['global_rank']}")

        # Per-topic coverage in selection
        topics = set()
        for r in selected:
            topics.update(r["topic_coverage"].keys())
        print(f"\nPer-topic coverage in selection:")
        for topic in sorted(topics):
            count = sum(1 for r in selected if topic in r["topic_coverage"])
            print(f"  {topic}: {count}/{config.target_count}")

    return ranked_all_path, selected_path
