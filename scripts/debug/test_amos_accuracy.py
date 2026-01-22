#!/usr/bin/env python3
"""Test AMOS accuracy on balanced sample.

Run with:
    .venv/bin/python scripts/test_amos_accuracy.py --policy G1_allergy_V2 --k 50
"""

import argparse
import asyncio
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from addm.methods.amos.phase1 import generate_formula_seed
from addm.methods.amos.phase2 import FormulaSeedInterpreter
from addm.llm import LLMService


async def load_or_generate_formula_seed(policy_id: str, llm: LLMService) -> dict:
    """Load formula seed from cache or generate if missing."""
    cache_path = Path(f"results/cache/formula_seeds/{policy_id}.json")
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    # Generate via Phase 1
    print(f"Formula seed not found, generating via Phase 1...")
    agenda_path = Path(f"data/query/yelp/{policy_id}_prompt.txt")
    if not agenda_path.exists():
        raise FileNotFoundError(f"Agenda not found: {agenda_path}")

    agenda = agenda_path.read_text()
    seed, usage = await generate_formula_seed(
        agenda=agenda,
        policy_id=policy_id,
        llm=llm,
        cache_dir=Path("results/cache/formula_seeds"),
    )
    print(f"Generated formula seed (usage: {usage})")
    return seed


def load_ground_truth(policy_id: str, k: int) -> dict:
    """Load ground truth data."""
    gt_path = Path(f"data/answers/yelp/{policy_id}_K{k}_groundtruth.json")
    if gt_path.exists():
        with open(gt_path) as f:
            return json.load(f)
    raise FileNotFoundError(f"Ground truth not found: {gt_path}")


def load_dataset(k: int) -> list:
    """Load dataset."""
    dataset_path = Path(f"data/context/yelp/dataset_K{k}.jsonl")
    restaurants = []
    with open(dataset_path) as f:
        for line in f:
            restaurants.append(json.loads(line))
    return restaurants


def select_balanced_sample(gt_data: dict, n_critical: int = 3, n_high: int = 2, n_low: int = 5) -> list:
    """Select balanced sample from ground truth."""
    critical = []
    high = []
    low = []

    for biz_id, data in gt_data.get("restaurants", {}).items():
        verdict = data.get("ground_truth", {}).get("verdict", "Low Risk")
        if verdict == "Critical Risk":
            critical.append(biz_id)
        elif verdict == "High Risk":
            high.append(biz_id)
        else:
            low.append(biz_id)

    # Select samples
    sample_ids = critical[:n_critical] + high[:n_high] + low[:n_low]
    return sample_ids


async def run_amos(restaurant: dict, seed: dict, llm: LLMService) -> dict:
    """Run AMOS on a single restaurant."""
    interpreter = FormulaSeedInterpreter(seed, llm)

    result = await interpreter.execute(
        reviews=restaurant.get("reviews", []),
        business=restaurant.get("business", {}),
        query="",
        sample_id=restaurant.get("business", {}).get("business_id", ""),
    )

    return {
        "verdict": result.get("_namespace", {}).get("VERDICT", "Unknown"),
        "score": result.get("_namespace", {}).get("TOTAL_SCORE", 0),
        "n_incidents": result.get("_namespace", {}).get("N_INCIDENTS", 0),
        "filter_stats": result.get("_filter_stats", {}),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="G1_allergy_V2")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--n-critical", type=int, default=3)
    parser.add_argument("--n-high", type=int, default=2)
    parser.add_argument("--n-low", type=int, default=5)
    args = parser.parse_args()

    print(f"Testing AMOS accuracy on {args.policy} K={args.k}")
    print("=" * 60)

    # Initialize LLM first (needed for potential seed generation)
    llm = LLMService()

    # Load resources
    seed = await load_or_generate_formula_seed(args.policy, llm)
    gt_data = load_ground_truth(args.policy, args.k)
    dataset = load_dataset(args.k)

    # Index dataset by business_id
    dataset_by_id = {
        r.get("business", {}).get("business_id"): r
        for r in dataset
    }

    # Select balanced sample
    sample_ids = select_balanced_sample(gt_data, args.n_critical, args.n_high, args.n_low)
    print(f"Selected {len(sample_ids)} samples")

    # Run AMOS on each sample
    correct = 0
    total = 0
    results = []

    for biz_id in sample_ids:
        if biz_id not in dataset_by_id:
            print(f"  ⚠️ {biz_id} not found in dataset")
            continue

        restaurant = dataset_by_id[biz_id]
        gt_info = gt_data.get("restaurants", {}).get(biz_id, {})
        gt_verdict = gt_info.get("ground_truth", {}).get("verdict", "Low Risk")
        name = gt_info.get("name", "Unknown")

        print(f"\n[{total + 1}] {name} ({biz_id})")
        print(f"    GT: {gt_verdict}")

        amos_result = await run_amos(restaurant, seed, llm)
        amos_verdict = amos_result["verdict"]

        is_correct = amos_verdict == gt_verdict
        correct += 1 if is_correct else 0
        total += 1

        status = "✅" if is_correct else "❌"
        print(f"    AMOS: {amos_verdict} (score={amos_result['score']}, incidents={amos_result['n_incidents']}) {status}")

        results.append({
            "business_id": biz_id,
            "name": name,
            "gt_verdict": gt_verdict,
            "amos_verdict": amos_verdict,
            "is_correct": is_correct,
            "amos_result": amos_result,
        })

    # Summary
    accuracy = correct / total if total > 0 else 0
    print("\n" + "=" * 60)
    print(f"ACCURACY: {correct}/{total} ({accuracy * 100:.1f}%)")
    print("=" * 60)

    if accuracy >= 0.75:
        print("✅ Target accuracy (>75%) achieved!")
    else:
        print("❌ Target accuracy (>75%) NOT achieved")

    # Show mismatches
    mismatches = [r for r in results if not r["is_correct"]]
    if mismatches:
        print(f"\nMismatches ({len(mismatches)}):")
        for m in mismatches:
            print(f"  - {m['name']}: GT={m['gt_verdict']}, AMOS={m['amos_verdict']}")


if __name__ == "__main__":
    asyncio.run(main())
