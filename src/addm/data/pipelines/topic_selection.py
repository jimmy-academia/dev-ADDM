"""Topic-aware selection of restaurants for ADDM benchmark.

Uses layered greedy selection to ensure balanced cell coverage:
- Cells = (topic, severity) where severity is "critical" or "high"
- 18 topics × 2 severities = 36 cells
- Algorithm fills level n (all cells have ≥n restaurants) before level n+1
- Multi-topic restaurants naturally prioritized (fill more cells per pick)

Outputs:
  1. topic_all.json - All restaurants ranked by cell count
  2. topic_{N}.json - Selected N restaurants with balanced cell coverage
"""

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


@dataclass
class TopicSelectionConfig:
    """Configuration for topic-aware restaurant selection."""

    keyword_hits_dir: Path
    output_dir: Path
    target_count: int = 100


# All 18 topics
TOPICS = [
    "G1_allergy", "G1_dietary", "G1_hygiene",
    "G2_romance", "G2_business", "G2_group",
    "G3_price_worth", "G3_hidden_costs", "G3_time_value",
    "G4_server", "G4_kitchen", "G4_environment",
    "G5_capacity", "G5_execution", "G5_consistency",
    "G6_uniqueness", "G6_comparison", "G6_loyalty",
]

# Severity levels
SEVERITIES = ["critical", "high"]


def load_keyword_hits(keyword_hits_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all keyword hit files and merge by business_id.

    Each restaurant can appear in critical_list OR high_list per topic (not both).
    We track which cells (topic, severity) each restaurant covers.

    Args:
        keyword_hits_dir: Directory containing G*_*.json files

    Returns:
        Dict mapping business_id to restaurant data with cells and topic_scores
    """
    restaurants: dict[str, dict[str, Any]] = {}

    for topic_file in sorted(keyword_hits_dir.glob("G*_*.json")):
        topic = topic_file.stem  # e.g., "G1_allergy"

        with open(topic_file) as f:
            topic_data = json.load(f)

        # Process critical_list
        for r in topic_data.get("critical_list", []):
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
                    "topic_scores": {},
                    "cells": set(),
                }

            # Add cell: (topic, "critical")
            restaurants[biz_id]["cells"].add((topic, "critical"))

            restaurants[biz_id]["topic_scores"][topic] = {
                "severity": "critical",
                "max_critical": r.get("max_critical_score", 0),
                "total_critical": r.get("total_critical_score", 0),
                "critical_count": r.get("critical_review_count", 0),
            }

        # Process high_list
        for r in topic_data.get("high_list", []):
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
                    "topic_scores": {},
                    "cells": set(),
                }

            # Add cell: (topic, "high")
            restaurants[biz_id]["cells"].add((topic, "high"))

            restaurants[biz_id]["topic_scores"][topic] = {
                "severity": "high",
                "max_high": r.get("max_high_score", 0),
                "total_high": r.get("total_high_score", 0),
                "high_count": r.get("high_review_count", 0),
            }

    # Compute derived metrics
    for biz_id, r in restaurants.items():
        r["cell_count"] = len(r["cells"])
        r["topic_count"] = len(r["topic_scores"])
        r["topics_covered"] = list(r["topic_scores"].keys())

        # Total value score (sum of max scores across all topics)
        r["total_value_score"] = sum(
            scores.get("max_critical", 0) + scores.get("max_high", 0)
            for scores in r["topic_scores"].values()
        )

    return restaurants


def greedy_select_layered(
    restaurants: dict[str, dict[str, Any]],
    target: int = 100,
) -> list[dict[str, Any]]:
    """Layered greedy selection ensuring cell coverage at each level.

    Algorithm:
    - Level n means: ensure ALL 36 cells have at least n restaurants
    - At each level, pick restaurant that fills the most "needy" cells
    - Multi-topic restaurants naturally score higher
    - Continue until target reached or all cells saturated

    Args:
        restaurants: Dict mapping business_id to restaurant data
        target: Number of restaurants to select

    Returns:
        List of selected restaurants
    """
    # All 36 cells
    all_cells = {(topic, severity) for topic in TOPICS for severity in SEVERITIES}

    selected = []
    selected_ids: set[str] = set()
    cell_counts: dict[tuple[str, str], int] = {cell: 0 for cell in all_cells}

    # Level n = ensure ALL cells have at least n restaurants
    for level in range(1, target + 1):
        if len(selected) >= target:
            break

        while len(selected) < target:
            # Which cells still need their nth restaurant?
            cells_needing_level_n = [
                cell for cell in all_cells
                if cell_counts[cell] < level
            ]

            if not cells_needing_level_n:
                # All cells filled to level n, move to next level
                break

            # Convert to set for faster lookup
            needy_cells_set = set(cells_needing_level_n)

            # Score each unselected restaurant by cells it would help
            best: tuple[str, dict[str, Any]] | None = None
            best_score = 0
            best_total_value = 0  # Tiebreaker

            for biz_id, r in restaurants.items():
                if biz_id in selected_ids:
                    continue

                # How many needy cells does this restaurant fill?
                score = sum(1 for cell in r["cells"] if cell in needy_cells_set)

                # Use total_value_score as tiebreaker
                if score > best_score or (score == best_score and r["total_value_score"] > best_total_value):
                    best_score = score
                    best_total_value = r["total_value_score"]
                    best = (biz_id, r)

            if best is None or best_score == 0:
                # Can't help any needy cells
                break

            # Select best
            biz_id, r = best
            selected.append(r)
            selected_ids.add(biz_id)

            # Update ALL cells this restaurant contributes to
            for cell in r["cells"]:
                cell_counts[cell] += 1

    return selected


def prepare_for_json(restaurants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert restaurants to JSON-serializable format.

    Sets are converted to sorted lists for deterministic output.
    """
    result = []
    for r in restaurants:
        r_copy = r.copy()
        # Convert cells set to sorted list of [topic, severity] pairs
        r_copy["cells"] = sorted([list(cell) for cell in r["cells"]])
        result.append(r_copy)
    return result


def run_selection(
    config: TopicSelectionConfig, verbose: bool = True
) -> tuple[Path, Path]:
    """Run full topic-aware selection pipeline.

    Args:
        config: Selection configuration
        verbose: If True, print progress information

    Returns:
        Tuple of (ranked_path, selected_path)
    """
    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all keyword hits
    if verbose:
        print(f"Loading keyword hits from {config.keyword_hits_dir}...")
    restaurants = load_keyword_hits(config.keyword_hits_dir)
    if verbose:
        print(f"  Found {len(restaurants)} unique restaurants")

    # Show cell distribution
    if verbose:
        from collections import Counter

        cell_dist = Counter(r["cell_count"] for r in restaurants.values())
        print(f"  Cell count distribution: {dict(sorted(cell_dist.items()))}")

        # Count restaurants per cell
        cell_restaurant_counts: dict[tuple[str, str], int] = {}
        for topic in TOPICS:
            for severity in SEVERITIES:
                cell = (topic, severity)
                count = sum(1 for r in restaurants.values() if cell in r["cells"])
                cell_restaurant_counts[cell] = count

        # Show cells with fewest restaurants
        sorted_cells = sorted(cell_restaurant_counts.items(), key=lambda x: x[1])
        print(f"  Cells with fewest restaurants:")
        for cell, count in sorted_cells[:5]:
            print(f"    {cell[0]} ({cell[1]}): {count}")

    # Run greedy selection
    if verbose:
        print(f"\nRunning layered greedy selection for {config.target_count} restaurants...")
    selected = greedy_select_layered(restaurants, target=config.target_count)

    # Compute selection statistics
    if verbose:
        from collections import Counter

        # Cell coverage in selection
        sel_cell_counts: dict[tuple[str, str], int] = {}
        for topic in TOPICS:
            for severity in SEVERITIES:
                cell = (topic, severity)
                count = sum(1 for r in selected if cell in r["cells"])
                sel_cell_counts[cell] = count

        # Level distribution
        min_level = min(sel_cell_counts.values())
        max_level = max(sel_cell_counts.values())
        level_dist = Counter(sel_cell_counts.values())
        print(f"  Selection coverage levels: min={min_level}, max={max_level}")
        print(f"  Level distribution: {dict(sorted(level_dist.items()))}")

        # Cells at minimum level
        min_cells = [cell for cell, count in sel_cell_counts.items() if count == min_level]
        if len(min_cells) <= 10:
            print(f"  Cells at min level ({min_level}): {min_cells}")
        else:
            print(f"  Cells at min level ({min_level}): {len(min_cells)} cells")

    # Rank all restaurants by cell count (for reference)
    all_ranked = sorted(
        restaurants.values(),
        key=lambda r: (r["cell_count"], r["total_value_score"]),
        reverse=True
    )

    # Assign ranks
    for i, r in enumerate(all_ranked):
        r["global_rank"] = i + 1

    # Also assign ranks to selected
    selected_ids = {r["business_id"] for r in selected}
    for r in all_ranked:
        if r["business_id"] in selected_ids:
            # Find and update in selected list
            for s in selected:
                if s["business_id"] == r["business_id"]:
                    s["global_rank"] = r["global_rank"]
                    break

    # Sort selected by global rank for output
    selected = sorted(selected, key=lambda x: x.get("global_rank", 0))

    # Save ranked list
    ranked_path = config.output_dir / "topic_all.json"
    with open(ranked_path, "w") as f:
        json.dump(prepare_for_json(all_ranked), f, indent=2)
    if verbose:
        print(f"\n  Saved full ranking to {ranked_path}")

    # Save selected list
    selected_path = config.output_dir / f"topic_{config.target_count}.json"
    with open(selected_path, "w") as f:
        json.dump(prepare_for_json(selected), f, indent=2)
    if verbose:
        print(f"  Saved selection to {selected_path}")

    # Summary statistics
    if verbose:
        print("\nSelection summary:")
        print(f"  Total restaurants in pool: {len(restaurants)}")
        print(f"  Selected: {len(selected)}")
        if selected:
            print(f"  Top selected rank: {selected[0].get('global_rank', 'N/A')}")
            print(f"  Bottom selected rank: {selected[-1].get('global_rank', 'N/A')}")

        # Per-topic coverage in selection
        print(f"\nPer-topic coverage in selection ({config.target_count} restaurants):")
        for topic in TOPICS:
            critical = sum(1 for r in selected if (topic, "critical") in r["cells"])
            high = sum(1 for r in selected if (topic, "high") in r["cells"])
            print(f"  {topic}: critical={critical}, high={high}")

    return ranked_path, selected_path
