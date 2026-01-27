#!/usr/bin/env python3
"""Select diverse samples from ground truth for testing.

Selects restaurants covering all verdict categories (Low/High/Critical Risk)
to ensure balanced evaluation of methods.

Usage:
    # Select samples for a T* policy (default: 2 per category = 6 total)
    .venv/bin/python scripts/select_diverse_samples.py --policy T1P1 --k 25

    # Select for all T1 policies
    .venv/bin/python scripts/select_diverse_samples.py --tier T1 --k 25 --output samples.json

    # Select for all policies and all K values
    .venv/bin/python scripts/select_diverse_samples.py --all --output data/answers/yelp/verdict_sample_ids.json

    # Custom count per category
    .venv/bin/python scripts/select_diverse_samples.py --policy T1P1 --k 25 --per-category 3
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from addm.tasks.constants import ALL_POLICIES, K_VALUES, TIERS


def get_gt_policy_id(policy_id: str) -> str:
    """Map T* policy to its ground truth policy ID.

    T* policies map to G* ground truth files:
    - P1, P4-P7: Uses V1 (base rules, same logic)
    - P2: Uses V2 (extended rules)
    - P3: Uses its own GT (T*P3, different logic - ALL vs ANY)
    """
    import re

    # Check if T* format (e.g., T1P1, T2P3)
    match = re.match(r'^T(\d)P(\d)$', policy_id)
    if not match:
        return policy_id  # Not T* format, return as-is

    tier_num = match.group(1)
    variant = int(match.group(2))

    # Map tier to G* topic
    tier_to_topic = {
        "1": "G1_allergy",
        "2": "G3_price_worth",
        "3": "G4_environment",
        "4": "G5_execution",
        "5": "G4_server",
    }
    topic = tier_to_topic.get(tier_num, f"G{tier_num}_unknown")

    # Map variant to version
    if variant == 2:
        return f"{topic}_V2"  # P2 uses V2 (extended rules)
    elif variant == 3:
        return policy_id  # P3 uses its own GT (T*P3)
    else:
        return f"{topic}_V1"  # P1, P4-P7 use V1 (base rules)


def load_ground_truth(policy_id: str, k: int) -> Optional[Dict]:
    """Load ground truth data for a policy.

    Args:
        policy_id: Policy identifier (e.g., T1P1, G1_allergy_V2)
        k: Context size (25, 50, 100, 200)

    Returns:
        Ground truth dict or None if not found
    """
    gt_policy_id = get_gt_policy_id(policy_id)
    gt_path = Path(f"data/answers/yelp/{gt_policy_id}_K{k}_groundtruth.json")
    if gt_path.exists():
        with open(gt_path) as f:
            return json.load(f)
    return None


def categorize_by_verdict(gt_data: Dict) -> Dict[str, List[str]]:
    """Categorize business IDs by verdict.

    Dynamically discovers verdict labels from the GT data itself.

    Args:
        gt_data: Ground truth data dict

    Returns:
        Dict mapping verdict to list of business IDs
    """
    by_verdict: Dict[str, List[str]] = {}

    for biz_id, data in gt_data.get("restaurants", {}).items():
        verdict = data.get("ground_truth", {}).get("verdict", "Unknown")
        if verdict not in by_verdict:
            by_verdict[verdict] = []
        by_verdict[verdict].append(biz_id)

    return by_verdict


def select_diverse_samples(
    gt_data: Dict,
    per_category: int = 2,
    seed: int = 42,
) -> Tuple[List[str], Dict[str, int]]:
    """Select diverse samples covering all verdict categories.

    Dynamically discovers verdict categories from GT data.
    Selects `per_category` samples from each non-empty category.

    Args:
        gt_data: Ground truth data dict
        per_category: Number of samples to select from each category (default: 2)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (list of business IDs, dict of counts per verdict)
    """
    import random
    random.seed(seed)

    by_verdict = categorize_by_verdict(gt_data)
    verdicts = list(by_verdict.keys())

    # Shuffle each category for randomness
    for v in by_verdict:
        random.shuffle(by_verdict[v])

    selected: List[str] = []
    counts = {v: 0 for v in verdicts}

    # Take up to per_category from each category
    for verdict in verdicts:
        available = by_verdict.get(verdict, [])
        take = min(per_category, len(available))
        for i in range(take):
            selected.append(available[i])
            counts[verdict] += 1

    return selected, counts


def select_for_policies(
    policies: List[str],
    k: int,
    per_category: int = 2,
    seed: int = 42,
) -> Dict[str, Dict]:
    """Select diverse samples for specified policies.

    Args:
        policies: List of policy IDs
        k: Context size
        per_category: Samples per category
        seed: Random seed

    Returns:
        Dict mapping policy_id to selection info
    """
    results = {}

    for policy_id in policies:
        gt_data = load_ground_truth(policy_id, k)
        if gt_data is None:
            gt_policy = get_gt_policy_id(policy_id)
            results[policy_id] = {
                "error": f"Ground truth not found: {gt_policy}_K{k}_groundtruth.json",
                "sample_ids": [],
                "counts": {},
            }
            continue

        sample_ids, counts = select_diverse_samples(gt_data, per_category=per_category, seed=seed)

        # Get names for readability
        names = {}
        for biz_id in sample_ids:
            rest_data = gt_data.get("restaurants", {}).get(biz_id, {})
            names[biz_id] = rest_data.get("name", "Unknown")

        results[policy_id] = {
            "sample_ids": sample_ids,
            "counts": counts,
            "names": names,
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Select diverse samples from ground truth for T* policies"
    )
    parser.add_argument(
        "--policy", type=str,
        help="Policy ID (e.g., T1P1) or comma-separated list"
    )
    parser.add_argument(
        "--tier", type=str,
        help="Tier ID (e.g., T1) - selects all P1-P7 variants"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Select for all 35 T* policies"
    )
    parser.add_argument(
        "--k", type=str, default="25",
        help="Context size (25, 50, 100, 200) or 'all' for all K values"
    )
    parser.add_argument(
        "--per-category", type=int, default=2,
        help="Number of samples per verdict category (default: 2, total 6)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--format", choices=["table", "ids", "json"], default="table",
        help="Output format: table (human readable), ids (comma-separated), json"
    )
    parser.add_argument(
        "--output", type=str,
        help="Output file path (required for --all or --tier with multiple K)"
    )

    args = parser.parse_args()

    # Determine policies to process
    if args.all:
        policies = ALL_POLICIES
    elif args.tier:
        tier = args.tier.upper()
        if tier not in TIERS:
            print(f"Error: Unknown tier '{args.tier}'. Valid: {TIERS}")
            sys.exit(1)
        policies = [f"{tier}P{v}" for v in range(1, 8)]
    elif args.policy:
        policies = [p.strip() for p in args.policy.split(",")]
    else:
        print("Error: Must specify --policy, --tier, or --all")
        sys.exit(1)

    # Parse --k value
    all_k = args.k.lower() == "all"
    k_value = None if all_k else int(args.k)

    # Multi-policy or multi-K mode: output to file
    if len(policies) > 1 or all_k:
        if not args.output:
            print("Error: --output is required when processing multiple policies or K values")
            sys.exit(1)

        k_values = K_VALUES if all_k else [k_value]
        combined = {}

        for policy_id in policies:
            combined[policy_id] = {}
            for k in k_values:
                gt_data = load_ground_truth(policy_id, k)
                if gt_data is None:
                    combined[policy_id][f"K{k}"] = ""
                    continue
                sample_ids, _ = select_diverse_samples(
                    gt_data, per_category=args.per_category, seed=args.seed
                )
                combined[policy_id][f"K{k}"] = ",".join(sample_ids)

        with open(args.output, "w") as f:
            json.dump(combined, f, indent=2)

        n_k = len(k_values)
        n_per = args.per_category * 3  # 3 categories
        print(f"Saved {len(policies)} policies × {n_k} K values ({n_per} samples each) to {args.output}")
        return

    # Single policy mode
    policy_id = policies[0]
    gt_data = load_ground_truth(policy_id, k_value)
    if gt_data is None:
        gt_policy = get_gt_policy_id(policy_id)
        print(f"Error: Ground truth not found: {gt_policy}_K{k_value}_groundtruth.json")
        sys.exit(1)

    sample_ids, counts = select_diverse_samples(
        gt_data, per_category=args.per_category, seed=args.seed
    )

    if args.format == "ids":
        # Just print comma-separated IDs
        print(",".join(sample_ids))

    elif args.format == "json":
        names = {}
        verdicts = {}
        for biz_id in sample_ids:
            rest_data = gt_data.get("restaurants", {}).get(biz_id, {})
            names[biz_id] = rest_data.get("name", "Unknown")
            verdicts[biz_id] = rest_data.get("ground_truth", {}).get("verdict", "Unknown")

        output = {
            "policy_id": policy_id,
            "gt_policy": get_gt_policy_id(policy_id),
            "k": k_value,
            "per_category": args.per_category,
            "sample_ids": sample_ids,
            "counts": counts,
            "names": names,
            "verdicts": verdicts,
        }
        print(json.dumps(output, indent=2))

    else:  # table format
        gt_policy = get_gt_policy_id(policy_id)
        print(f"Diverse samples for {policy_id} (GT: {gt_policy}) K={k_value}")
        print("=" * 70)

        by_verdict = categorize_by_verdict(gt_data)

        # Dynamic display of available counts
        avail_str = " ".join(f"{v[:3]}={len(items)}" for v, items in by_verdict.items())
        print(f"Total available: {avail_str}")

        # Dynamic display of selected counts
        sel_str = " ".join(f"{v[:3]}={c}" for v, c in counts.items())
        print(f"Selected {len(sample_ids)} samples ({args.per_category}/category): {sel_str}")
        print()

        for biz_id in sample_ids:
            rest_data = gt_data.get("restaurants", {}).get(biz_id, {})
            name = rest_data.get("name", "Unknown")
            verdict = rest_data.get("ground_truth", {}).get("verdict", "Unknown")

            # Verdict abbreviation (first letter of each word)
            v_abbr = "".join(w[0] for w in verdict.split()) if verdict != "Unknown" else "?"

            print(f"  [{v_abbr}] {name[:40]:<40} ({biz_id[:16]}...)")

        print()
        print("Sample IDs (for --sample-ids):")
        print(f"  {','.join(sample_ids)}")


if __name__ == "__main__":
    main()
