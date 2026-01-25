"""
CLI: Compute ground truth from cached judgments.

Usage:
    # Compute GT for a topic (all V1-V4 variants)
    python -m addm.tasks.cli.compute_gt --topic G1_allergy

    # Compute GT for ALL policies (default)
    python -m addm.tasks.cli.compute_gt

    # Single policy
    python -m addm.tasks.cli.compute_gt --policy G1_allergy_V2

    # Legacy task-based GT (single formula module)
    python -m addm.tasks.cli.compute_gt --task G1a --domain yelp --k 50
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from addm.tasks.base import load_task
from addm.tasks.extraction import JudgmentCache, PolicyJudgmentCache
from addm.tasks.policy_gt import (
    compute_gt_from_policy,
    load_overrides,
    load_policy,
)

# All topics, policies, K values, and expansion helpers (imported from shared constants)
from addm.tasks.constants import (
    ALL_TOPICS,
    ALL_POLICIES,
    K_VALUES,
    expand_policies,
    get_topic_from_policy_id,
)


def _get_judgement_cache_path(domain: str) -> Path:
    """Get cache path for L0 judgement extraction."""
    return Path(f"data/answers/{domain}/judgement_cache.json")


def _get_policy_gt_path(policy_id: str, domain: str, k: int) -> Path:
    """Get GT output path for a policy."""
    return Path(f"data/answers/{domain}/{policy_id}_K{k}_groundtruth.json")


def main_task(args: argparse.Namespace) -> None:
    """Legacy task-based GT computation."""
    # Load task config
    task = load_task(args.task, args.domain)
    print(f"Computing GT for task: {task.task_id} ({task.domain})")

    # Load dataset to get restaurant metadata
    dataset_path = Path(f"data/context/{args.domain}/dataset_K{args.k}.jsonl")
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

        # Compute GT using task's formula module
        restaurant_meta = {"categories": categories, "name": name}
        gt = task.compute_ground_truth(restaurant_judgments, restaurant_meta)

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
    gt_path = task.groundtruth_path(args.k)
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved GT to {gt_path}")
    print(f"Distribution: {verdicts}")


def main_policy(args: argparse.Namespace) -> None:
    """Policy-based GT computation."""
    # Parse policy list (comma-separated)
    policy_ids = [p.strip() for p in args.policy.split(",")]
    print(f"Computing GT for {len(policy_ids)} policies")

    # Group policies by topic for efficient cache loading
    from collections import defaultdict
    policies_by_topic: Dict[str, List[str]] = defaultdict(list)
    for pid in policy_ids:
        topic = get_topic_from_policy_id(pid)
        policies_by_topic[topic].append(pid)

    print(f"Topics: {list(policies_by_topic.keys())}")

    # Load dataset
    dataset_path = Path(f"data/context/{args.domain}/dataset_K{args.k}.jsonl")
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    with open(dataset_path) as f:
        restaurants = [json.loads(line) for line in f]

    # Load cache once
    cache_path = _get_judgement_cache_path(args.domain)
    cache = PolicyJudgmentCache(cache_path)

    # Track skipped topics
    skipped_topics = []

    # Process each topic
    for topic, topic_policies in policies_by_topic.items():
        print(f"\n{'='*60}")
        print(f"Topic: {topic} ({len(topic_policies)} policies)")
        print(f"{'='*60}")

        # Load aggregated judgments for this topic
        all_judgments = cache.get_all_aggregated(topic)
        print(f"Loaded {len(all_judgments)} aggregated judgments")

        if not all_judgments:
            print(f"[SKIP] No judgments for {topic} - extraction not complete")
            skipped_topics.append(topic)
            continue

        # Load human judgment overrides for this topic
        overrides = load_overrides(topic)
        if overrides:
            print(f"Loaded {len(overrides)} judgment overrides for {topic}")

        # Process each policy in this topic
        for policy_id in topic_policies:
            print(f"\n--- Policy: {policy_id} ---")

            # Load policy
            try:
                policy = load_policy(policy_id)
            except FileNotFoundError as e:
                print(f"[ERROR] {e}")
                continue

            # Prepare output
            output: Dict[str, Any] = {
                "policy_id": policy_id,
                "topic": topic,
                "domain": args.domain,
                "k": args.k,
                "computed_at": datetime.now().isoformat(),
                "has_scoring": policy.normative.scoring is not None,
                "restaurants": {},
            }

            verdicts: Dict[str, int] = {}
            for v in policy.get_verdicts():
                verdicts[v] = 0

            missing_judgments = 0

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
                    if review_id in all_judgments:
                        # Include review date for V3 recency weighting
                        judgment = all_judgments[review_id].copy()
                        judgment["_review_date"] = review.get("date")
                        restaurant_judgments.append(judgment)
                    else:
                        missing_judgments += 1

                # Compute GT using policy scoring (with human overrides)
                restaurant_meta = {"categories": categories, "name": name}
                gt = compute_gt_from_policy(restaurant_judgments, policy, restaurant_meta, overrides, k=args.k)

                # Store result
                verdict = gt.get("verdict", "Unknown")
                score = gt.get("score", 0)

                output["restaurants"][business_id] = {
                    "name": name,
                    "categories": categories,
                    "n_reviews": len(reviews),
                    "n_judgments": len(restaurant_judgments),
                    "ground_truth": gt,
                }

                if verdict in verdicts:
                    verdicts[verdict] += 1
                else:
                    verdicts[verdict] = 1

                if args.verbose:
                    print(f"{name}: {verdict} (score={score})")

            # Summary
            output["summary"] = {
                "total_restaurants": len(restaurants),
                "verdict_distribution": verdicts,
                "missing_judgments": missing_judgments,
            }

            # Save ground truth
            gt_path = _get_policy_gt_path(policy_id, args.domain, args.k)
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(gt_path, "w") as f:
                json.dump(output, f, indent=2)

            print(f"Saved: {gt_path}")
            print(f"Distribution: {verdicts}")
            if missing_judgments > 0:
                print(f"[WARN] {missing_judgments} reviews missing judgments")

    # Final summary
    if skipped_topics:
        print(f"\n{'='*60}")
        print(f"[SUMMARY] Skipped {len(skipped_topics)} topics (no judgments):")
        for t in skipped_topics:
            print(f"  - {t}")
        print("Run extraction first: python -m addm.tasks.cli.extract --k 200 --all")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Compute ground truth from cached judgments")

    # Target (mutually exclusive)
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--task", type=str, help="Task ID (e.g., G1a) - legacy mode")
    target_group.add_argument(
        "--topic",
        type=str,
        help="Topic (e.g., G1_allergy) - computes all V1-V4 variants for K=25,50,100,200",
    )
    target_group.add_argument(
        "--group",
        type=str,
        help="Group (e.g., G1) - computes all topics × V1-V4 variants (12 policies)",
    )
    target_group.add_argument(
        "--policy",
        type=str,
        help="Policy ID(s), comma-separated (e.g., G1_allergy_V2 or G1_allergy_V1,V2,V3,V4)",
    )
    target_group.add_argument(
        "--all", action="store_true", help="Compute GT for ALL policies (72 total) - default"
    )

    # Common options
    parser.add_argument("--domain", type=str, default="yelp", help="Domain (default: yelp)")
    parser.add_argument("--k", type=int, default=200, help="Dataset K value (default: 200, ignored when using --topic)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.task:
        main_task(args)
    elif args.topic or args.group or args.policy or args.all:
        # Use centralized expand_policies() for all policy-based modes
        try:
            policies = expand_policies(
                policy=args.policy,
                topic=args.topic,
                group=args.group,
            )
        except ValueError as e:
            print(f"[ERROR] {e}")
            return

        args.policy = ",".join(policies)

        # For --topic, iterate over all K values
        if args.topic:
            print(f"\n{'='*70}")
            print(f"Generating GT for topic: {args.topic}")
            print(f"K values: {K_VALUES}")
            print(f"Policies: {args.policy}")
            print(f"{'='*70}\n")

            for k_value in K_VALUES:
                print(f"\n{'#'*70}")
                print(f"# K = {k_value}")
                print(f"{'#'*70}\n")
                args.k = k_value
                main_policy(args)
        # For --group, also iterate over all K values
        elif args.group:
            print(f"\n{'='*70}")
            print(f"Generating GT for group: {args.group}")
            print(f"K values: {K_VALUES}")
            print(f"Policies ({len(policies)}): {args.policy[:80]}...")
            print(f"{'='*70}\n")

            for k_value in K_VALUES:
                print(f"\n{'#'*70}")
                print(f"# K = {k_value}")
                print(f"{'#'*70}\n")
                args.k = k_value
                main_policy(args)
        else:
            main_policy(args)
    else:
        # Default: compute GT for all policies
        print("No target specified, defaulting to --all (72 policies)")
        args.policy = ",".join(ALL_POLICIES)
        main_policy(args)


if __name__ == "__main__":
    main()
