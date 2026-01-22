#!/usr/bin/env python3
"""Select diverse samples from ground truth for testing.

Selects restaurants covering all verdict categories (Low/High/Critical Risk)
to ensure balanced evaluation of methods.

Usage:
    # Select 5 samples for a policy
    .venv/bin/python scripts/select_diverse_samples.py --policy G1_allergy_V2 --k 50 --n 5

    # Output comma-separated IDs (for --sample-ids flag)
    .venv/bin/python scripts/select_diverse_samples.py --policy G1_allergy_V2 --k 50 --n 5 --format ids

    # Select for all policies and save to file
    .venv/bin/python scripts/select_diverse_samples.py --all --k 50 --n 5 --output samples.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from addm.tasks.constants import ALL_POLICIES


def load_ground_truth(policy_id: str, k: int) -> Optional[Dict]:
    """Load ground truth data for a policy.

    Args:
        policy_id: Policy identifier (e.g., G1_allergy_V2)
        k: Context size (25, 50, 100, 200)

    Returns:
        Ground truth dict or None if not found
    """
    gt_path = Path(f"data/answers/yelp/{policy_id}_K{k}_groundtruth.json")
    if gt_path.exists():
        with open(gt_path) as f:
            return json.load(f)
    return None


def categorize_by_verdict(gt_data: Dict) -> Dict[str, List[str]]:
    """Categorize business IDs by verdict.

    Args:
        gt_data: Ground truth data dict

    Returns:
        Dict mapping verdict to list of business IDs
    """
    by_verdict: Dict[str, List[str]] = {
        "Critical Risk": [],
        "High Risk": [],
        "Low Risk": [],
    }

    for biz_id, data in gt_data.get("restaurants", {}).items():
        verdict = data.get("ground_truth", {}).get("verdict", "Low Risk")
        if verdict in by_verdict:
            by_verdict[verdict].append(biz_id)
        else:
            # Unknown verdict, put in Low Risk
            by_verdict["Low Risk"].append(biz_id)

    return by_verdict


def select_diverse_samples(
    gt_data: Dict,
    n: int = 5,
    seed: int = 42,
) -> Tuple[List[str], Dict[str, int]]:
    """Select diverse samples covering all verdict categories.

    Tries to select from each category proportionally:
    - First, ensure at least 1 from each non-empty category
    - Then fill remaining slots from categories with more samples

    Args:
        gt_data: Ground truth data dict
        n: Total number of samples to select
        seed: Random seed for reproducibility

    Returns:
        Tuple of (list of business IDs, dict of counts per verdict)
    """
    import random
    random.seed(seed)

    by_verdict = categorize_by_verdict(gt_data)

    # Shuffle each category for randomness
    for v in by_verdict:
        random.shuffle(by_verdict[v])

    selected: List[str] = []
    counts = {"Critical Risk": 0, "High Risk": 0, "Low Risk": 0}

    # First pass: take 1 from each non-empty category
    for verdict in ["Critical Risk", "High Risk", "Low Risk"]:
        if by_verdict[verdict] and len(selected) < n:
            selected.append(by_verdict[verdict].pop(0))
            counts[verdict] += 1

    # Second pass: fill remaining slots proportionally
    remaining = n - len(selected)

    if remaining > 0:
        # Create a pool of remaining items
        pool = []
        for verdict, items in by_verdict.items():
            for item in items:
                pool.append((verdict, item))

        random.shuffle(pool)

        for verdict, biz_id in pool[:remaining]:
            selected.append(biz_id)
            counts[verdict] += 1

    return selected, counts


def select_for_all_policies(k: int, n: int, seed: int = 42) -> Dict[str, Dict]:
    """Select diverse samples for all policies.

    Args:
        k: Context size
        n: Number of samples per policy
        seed: Random seed

    Returns:
        Dict mapping policy_id to selection info
    """
    results = {}

    for policy_id in ALL_POLICIES:
        gt_data = load_ground_truth(policy_id, k)
        if gt_data is None:
            results[policy_id] = {
                "error": f"Ground truth not found for {policy_id} K={k}",
                "sample_ids": [],
                "counts": {},
            }
            continue

        sample_ids, counts = select_diverse_samples(gt_data, n=n, seed=seed)

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
        description="Select diverse samples from ground truth"
    )
    parser.add_argument(
        "--policy", type=str,
        help="Policy ID (e.g., G1_allergy_V2)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Select for all 72 policies"
    )
    parser.add_argument(
        "--k", type=int, default=50,
        help="Context size (default: 50)"
    )
    parser.add_argument(
        "--n", type=int, default=5,
        help="Number of samples to select (default: 5)"
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
        help="Output file path (for --all with json format)"
    )

    args = parser.parse_args()

    if args.all:
        results = select_for_all_policies(args.k, args.n, args.seed)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved results to {args.output}")
        else:
            # Print summary
            print(f"Diverse samples for all 72 policies (K={args.k}, n={args.n})")
            print("=" * 70)

            total_with_gt = 0
            total_missing_gt = 0

            for policy_id, info in results.items():
                if "error" in info:
                    print(f"{policy_id}: ❌ {info['error']}")
                    total_missing_gt += 1
                else:
                    counts = info["counts"]
                    c = counts.get("Critical Risk", 0)
                    h = counts.get("High Risk", 0)
                    l = counts.get("Low Risk", 0)
                    print(f"{policy_id}: ✓ C={c} H={h} L={l}")
                    total_with_gt += 1

            print("=" * 70)
            print(f"Total: {total_with_gt} with GT, {total_missing_gt} missing GT")

        return

    if not args.policy:
        parser.error("Either --policy or --all is required")

    # Single policy mode
    gt_data = load_ground_truth(args.policy, args.k)
    if gt_data is None:
        print(f"Error: Ground truth not found for {args.policy} K={args.k}")
        print(f"Run: .venv/bin/python -m addm.tasks.cli.compute_gt --policy {args.policy} --k {args.k}")
        sys.exit(1)

    sample_ids, counts = select_diverse_samples(gt_data, n=args.n, seed=args.seed)

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
            "policy_id": args.policy,
            "k": args.k,
            "n": args.n,
            "sample_ids": sample_ids,
            "counts": counts,
            "names": names,
            "verdicts": verdicts,
        }
        print(json.dumps(output, indent=2))

    else:  # table format
        print(f"Diverse samples for {args.policy} K={args.k}")
        print("=" * 70)

        by_verdict = categorize_by_verdict(gt_data)
        print(f"Total available: C={len(by_verdict['Critical Risk'])} "
              f"H={len(by_verdict['High Risk'])} L={len(by_verdict['Low Risk'])}")
        print(f"Selected {len(sample_ids)} samples: C={counts['Critical Risk']} "
              f"H={counts['High Risk']} L={counts['Low Risk']}")
        print()

        for biz_id in sample_ids:
            rest_data = gt_data.get("restaurants", {}).get(biz_id, {})
            name = rest_data.get("name", "Unknown")
            verdict = rest_data.get("ground_truth", {}).get("verdict", "Unknown")
            score = rest_data.get("ground_truth", {}).get("score", 0)

            # Verdict abbreviation
            v_abbr = {"Critical Risk": "C", "High Risk": "H", "Low Risk": "L"}.get(verdict, "?")

            print(f"  [{v_abbr}] {name[:40]:<40} ({biz_id[:16]}...) score={score}")

        print()
        print("Sample IDs (for --sample-ids):")
        print(f"  {','.join(sample_ids)}")


if __name__ == "__main__":
    main()
