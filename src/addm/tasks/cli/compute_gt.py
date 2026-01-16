"""
CLI: Compute ground truth from cached judgments.

Usage:
    python -m addm.tasks.cli.compute_gt --task G1a --domain yelp
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from addm.tasks.base import load_task
from addm.tasks.extraction import JudgmentCache
from addm.tasks.executor import compute_ground_truth


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Compute ground truth from cached judgments")
    parser.add_argument("--task", type=str, required=True, help="Task ID (e.g., G1a)")
    parser.add_argument("--domain", type=str, default="yelp", help="Domain (default: yelp)")
    parser.add_argument("--k", type=int, default=50, help="Dataset K value (default: 50)")

    args = parser.parse_args()

    # Load task config
    task = load_task(args.task, args.domain)
    print(f"Computing GT for task: {task.task_id} ({task.domain})")

    # Load dataset to get restaurant metadata
    dataset_path = Path(f"data/processed/{args.domain}/dataset_K{args.k}.jsonl")
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    with open(dataset_path) as f:
        restaurants = [json.loads(line) for line in f]

    # Load judgment cache
    cache = JudgmentCache(task.cache_path)
    all_judgments = cache.get_all(task.task_id)
    print(f"Loaded {len(all_judgments)} judgments from cache")

    # Build review_id -> judgment lookup
    judgment_by_id = all_judgments

    # Compute GT for each restaurant
    output: Dict[str, Any] = {
        "task_id": task.task_id,
        "domain": task.domain,
        "k": args.k,
        "computed_at": datetime.now().isoformat(),
        "restaurants": {},
    }

    verdicts = {"Low Risk": 0, "High Risk": 0, "Critical Risk": 0}

    for restaurant in restaurants:
        business = restaurant.get("business", {})
        business_id = business.get("business_id", "")
        name = business.get("name", "Unknown")
        categories = business.get("categories", "")

        # Gather judgments for this restaurant's reviews
        reviews = restaurant.get("reviews", [])
        restaurant_judgments: List[Dict[str, Any]] = []

        for review in reviews:
            review_id = review.get("review_id", "")
            if review_id in judgment_by_id:
                restaurant_judgments.append(judgment_by_id[review_id])

        # Compute GT using generic executor
        restaurant_meta = {"categories": categories, "name": name}
        gt = compute_ground_truth(restaurant_judgments, restaurant_meta, task.parsed_prompt)

        # Store result
        verdict = gt.get("verdict", "Unknown")
        final_score = gt.get("FINAL_RISK_SCORE", 0)

        output["restaurants"][business_id] = {
            "name": name,
            "categories": categories,
            "n_reviews": len(reviews),
            "n_judgments": len(restaurant_judgments),
            "ground_truth": gt,
        }

        if verdict in verdicts:
            verdicts[verdict] += 1
        print(f"{name}: {verdict} (score={final_score:.2f})")

    # Summary
    output["summary"] = {
        "total_restaurants": len(restaurants),
        "verdict_distribution": verdicts,
    }

    # Save ground truth
    gt_path = task.groundtruth_path
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved GT to {gt_path}")
    print(f"Distribution: {verdicts}")


if __name__ == "__main__":
    main()
