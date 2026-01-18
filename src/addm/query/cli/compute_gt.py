"""CLI: Compute ground truth from judgments using PolicyIR.

Usage:
    python -m addm.query.cli.compute_gt G1_allergy_V2 --k 50
    python -m addm.query.cli.compute_gt G1_allergy_V0 --k 50 --domain yelp
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from addm.query.models.policy import PolicyIR
from addm.query.evaluator import PolicyEvaluator


def resolve_policy_path(policy_id: str) -> Path:
    """Resolve policy ID to YAML file path.

    Args:
        policy_id: e.g., "G1_allergy_V2"

    Returns:
        Path to YAML file
    """
    # Parse policy_id: G1_allergy_V2 -> G1/allergy/V2.yaml
    parts = policy_id.split("_")
    if len(parts) != 3:
        raise ValueError(f"Invalid policy_id format: {policy_id}. Expected: G1_allergy_V2")

    group, topic, version = parts
    return Path(f"src/addm/query/policies/{group}/{topic}/{version}.yaml")


def load_judgment_cache(domain: str) -> Dict[str, Dict[str, Any]]:
    """Load shared judgment cache for a domain.

    Args:
        domain: e.g., "yelp"

    Returns:
        Dict mapping review_id -> judgment
    """
    cache_path = Path(f"data/tasks/{domain}/cache.json")
    if not cache_path.exists():
        print(f"Warning: Cache not found at {cache_path}")
        return {}

    with open(cache_path) as f:
        cache_data = json.load(f)

    # Flatten the cache (old format has task_id -> review_id -> judgment)
    # New format should be just review_id -> judgment
    flattened: Dict[str, Dict[str, Any]] = {}

    for key, value in cache_data.items():
        if isinstance(value, dict):
            # Check if this is a task-keyed entry (old format)
            if any(isinstance(v, dict) and "review_id" in v for v in value.values()):
                # Old format: task_id -> review_id -> judgment
                for review_id, judgment in value.items():
                    if isinstance(judgment, dict):
                        flattened[review_id] = judgment
            elif "review_id" in value:
                # New format: review_id -> judgment
                flattened[key] = value

    return flattened


def get_judgments_for_restaurant(
    restaurant: Dict[str, Any], cache: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Get judgments for all reviews in a restaurant.

    Args:
        restaurant: Restaurant data with reviews
        cache: Judgment cache

    Returns:
        List of judgments for this restaurant's reviews
    """
    judgments = []
    reviews = restaurant.get("reviews", [])

    for review in reviews:
        review_id = review.get("review_id", "")
        if review_id in cache:
            judgment = cache[review_id].copy()
            # Add review metadata if not present
            if "date" not in judgment:
                judgment["date"] = review.get("date", "")
            if "stars" not in judgment:
                judgment["stars"] = review.get("stars", 0)
            judgments.append(judgment)

    return judgments


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute ground truth from judgments using PolicyIR"
    )
    parser.add_argument(
        "policy_id", type=str, help="Policy ID (e.g., G1_allergy_V2)"
    )
    parser.add_argument(
        "--k", type=int, default=50, help="Dataset K value (default: 50)"
    )
    parser.add_argument(
        "--domain", type=str, default="yelp", help="Domain (default: yelp)"
    )
    parser.add_argument(
        "--output", type=str, help="Output path (default: data/groundtruth/{domain}/{policy_id}_K{k}.json)"
    )

    args = parser.parse_args()

    # Load policy
    policy_path = resolve_policy_path(args.policy_id)
    if not policy_path.exists():
        print(f"Error: Policy not found at {policy_path}")
        return

    print(f"Loading policy: {args.policy_id}")
    policy = PolicyIR.load(policy_path)
    evaluator = PolicyEvaluator(policy)

    # Load dataset
    dataset_path = Path(f"data/context/{args.domain}/dataset_K{args.k}.jsonl")
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path) as f:
        restaurants = [json.loads(line) for line in f]

    print(f"Loaded {len(restaurants)} restaurants")

    # Load judgment cache
    print(f"Loading judgment cache for {args.domain}")
    cache = load_judgment_cache(args.domain)
    print(f"Loaded {len(cache)} judgments from cache")

    # Compute ground truth for each restaurant
    output: Dict[str, Any] = {
        "policy_id": args.policy_id,
        "domain": args.domain,
        "k": args.k,
        "computed_at": datetime.now().isoformat(),
        "restaurants": {},
    }

    verdict_counts: Dict[str, int] = {}

    for restaurant in restaurants:
        business = restaurant.get("business", {})
        business_id = business.get("business_id", "")
        name = business.get("name", "Unknown")
        categories = business.get("categories", "")

        # Get judgments for this restaurant
        judgments = get_judgments_for_restaurant(restaurant, cache)

        # Compute ground truth
        restaurant_meta = {"categories": categories, "name": name}
        result = evaluator.evaluate(judgments, restaurant_meta)

        # Store result
        verdict = result.verdict
        output["restaurants"][business_id] = {
            "name": name,
            "categories": categories,
            "n_reviews": len(restaurant.get("reviews", [])),
            "n_judgments": len(judgments),
            "ground_truth": result.to_dict(),
        }

        # Count verdicts
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        print(f"{name}: {verdict}")

    # Summary
    output["summary"] = {
        "total_restaurants": len(restaurants),
        "verdict_distribution": verdict_counts,
    }

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"data/groundtruth/{args.domain}/{args.policy_id}_K{args.k}.json")

    # Save ground truth
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved ground truth to {output_path}")
    print(f"Distribution: {verdict_counts}")


if __name__ == "__main__":
    main()
