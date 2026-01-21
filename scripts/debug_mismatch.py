#!/usr/bin/env python3
"""Debug specific mismatch cases.

Run with:
    .venv/bin/python scripts/debug_mismatch.py
"""

import asyncio
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from addm.methods.amos.phase2 import FormulaSeedInterpreter
from addm.llm import LLMService


def load_formula_seed(policy_id: str) -> dict:
    cache_path = Path(f"results/cache/formula_seeds/{policy_id}.json")
    with open(cache_path) as f:
        return json.load(f)


def load_ground_truth(policy_id: str, k: int) -> dict:
    gt_path = Path(f"data/answers/yelp/{policy_id}_K{k}_groundtruth.json")
    with open(gt_path) as f:
        return json.load(f)


def load_restaurant(biz_id: str, k: int) -> dict:
    dataset_path = Path(f"data/context/yelp/dataset_K{k}.jsonl")
    with open(dataset_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("business", {}).get("business_id") == biz_id:
                return entry
    raise ValueError(f"Restaurant not found: {biz_id}")


async def debug_restaurant(biz_id: str, policy_id: str, k: int):
    """Debug AMOS execution for a single restaurant."""
    print(f"\n{'='*60}")
    print(f"Debugging: {biz_id}")
    print(f"{'='*60}")

    # Load data
    seed = load_formula_seed(policy_id)
    gt_data = load_ground_truth(policy_id, k)
    restaurant = load_restaurant(biz_id, k)

    gt_info = gt_data.get("restaurants", {}).get(biz_id, {})
    gt_verdict = gt_info.get("ground_truth", {}).get("verdict", "Unknown")
    gt_incidents = gt_info.get("ground_truth", {}).get("incidents", [])
    name = gt_info.get("name", "Unknown")

    print(f"Name: {name}")
    print(f"GT Verdict: {gt_verdict}")
    print(f"GT Incidents: {len(gt_incidents)}")

    if gt_incidents:
        print("GT Incident Review IDs:")
        for inc in gt_incidents:
            print(f"  - {inc.get('review_id')}: {inc.get('severity')}")

    reviews = restaurant.get("reviews", [])
    print(f"\nTotal reviews: {len(reviews)}")

    # Check which GT incident reviews are in the dataset
    gt_review_ids = {inc.get("review_id") for inc in gt_incidents}
    found_reviews = [r for r in reviews if r.get("review_id") in gt_review_ids]
    print(f"GT incident reviews in dataset: {len(found_reviews)}")

    # Check keyword filter
    keywords = seed.get("filter", {}).get("keywords", [])
    print(f"\nKeywords ({len(keywords)}): {keywords[:10]}...")

    # Test keyword filter manually
    import re
    patterns = [(kw, re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)) for kw in keywords]

    for review in found_reviews:
        review_id = review.get("review_id")
        text = review.get("text", "")
        hits = []
        for kw, pattern in patterns:
            if pattern.search(text):
                hits.append(kw)

        if hits:
            print(f"\n  Review {review_id}: PASSED filter (keywords: {hits})")
        else:
            print(f"\n  Review {review_id}: FAILED filter (no keyword matches)")
            print(f"    Text preview: {text[:300]}...")

    # Run full AMOS and show details
    print("\n" + "-" * 40)
    print("Running AMOS pipeline...")
    print("-" * 40)

    llm = LLMService()
    interpreter = FormulaSeedInterpreter(seed, llm)

    result = await interpreter.execute(
        reviews=reviews,
        business=restaurant.get("business", {}),
        query="",
        sample_id=biz_id,
    )

    print(f"\nFilter stats: {result.get('_filter_stats', {})}")
    print(f"Namespace: {result.get('_namespace', {})}")

    extractions = result.get("_extractions", [])
    print(f"\nExtractions ({len(extractions)}):")

    # Get review texts for extractions with incidents
    review_by_id = {r.get("review_id"): r for r in reviews}

    incident_extractions = [e for e in extractions
                           if e.get("INCIDENT_SEVERITY") in ["mild", "moderate", "severe"]
                           and e.get("ACCOUNT_TYPE") == "firsthand"]

    print(f"\nIncident extractions ({len(incident_extractions)}):")
    for ext in incident_extractions:
        review_id = ext.get("review_id")
        review = review_by_id.get(review_id, {})
        text = review.get("text", "")[:400]

        is_gt = review_id in gt_review_ids
        marker = "⚠️ GT" if is_gt else ""

        print(f"\n  {review_id} {marker}")
        print(f"    ACCOUNT_TYPE: {ext.get('ACCOUNT_TYPE')}")
        print(f"    INCIDENT_SEVERITY: {ext.get('INCIDENT_SEVERITY')}")
        print(f"    ASSURANCE_OF_SAFETY: {ext.get('ASSURANCE_OF_SAFETY')}")
        print(f"    Review text: {text}...")


async def main():
    policy_id = "G1_allergy_V2"
    k = 50

    # Debug false positives (GT=low, AMOS=risk)
    false_positives = [
        "eZ-t73r7ETHjyclRB7SnwQ",  # Bliss: GT=Low, AMOS=Critical
    ]

    for biz_id in false_positives:
        await debug_restaurant(biz_id, policy_id, k)


if __name__ == "__main__":
    asyncio.run(main())
