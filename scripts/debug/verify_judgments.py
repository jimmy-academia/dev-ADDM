#!/usr/bin/env python3
"""Manual verification helper for judgment cache entries.

Usage:
    # Verify specific review
    .venv/bin/python scripts/debug/verify_judgments.py --review-id abc123 --topic G1_allergy

    # Verify from incident list file
    .venv/bin/python scripts/debug/verify_judgments.py --input /tmp/g1_allergy_remaining.json --topic G1_allergy

    # Continue from specific index
    .venv/bin/python scripts/debug/verify_judgments.py --input /tmp/g1_allergy_remaining.json --topic G1_allergy --start 5
"""

import argparse
import json
from pathlib import Path


def load_dataset(k: int = 200) -> dict[str, dict]:
    """Load dataset and index by review_id for fast lookup."""
    dataset_path = Path(f"data/context/yelp/dataset_K{k}.jsonl")
    reviews = {}
    with open(dataset_path) as f:
        for line in f:
            entry = json.loads(line)
            business_id = entry["business_id"]
            for review in entry.get("reviews", []):
                review_id = review["review_id"]
                reviews[review_id] = {
                    "review": review,
                    "business": entry.get("business", {}),
                    "business_id": business_id,
                }
    return reviews


def load_judgment_cache() -> dict:
    """Load the judgment cache."""
    cache_path = Path("data/answers/yelp/judgement_cache.json")
    with open(cache_path) as f:
        return json.load(f)


def load_overrides() -> dict:
    """Load existing overrides."""
    override_path = Path("data/answers/yelp/judgment_overrides.json")
    with open(override_path) as f:
        return json.load(f)


def save_overrides(overrides: dict):
    """Save overrides file."""
    override_path = Path("data/answers/yelp/judgment_overrides.json")
    with open(override_path, "w") as f:
        json.dump(overrides, f, indent=2)


def get_judgment(cache: dict, topic: str, business_id: str) -> dict | None:
    """Get aggregated judgment for a review."""
    key = f"{topic}::{business_id}"
    aggregated = cache.get("aggregated", {})
    return aggregated.get(key)


def display_review(review_data: dict, judgment: dict, topic: str, index: int = 0, total: int = 0):
    """Display review and judgment for verification."""
    review = review_data["review"]
    business = review_data["business"]

    print("\n" + "=" * 80)
    if total > 0:
        print(f"Review {index + 1}/{total}")
    print("=" * 80)
    print(f"Review ID:   {review['review_id']}")
    print(f"Business:    {business.get('name', 'N/A')} ({review_data['business_id']})")
    print(f"Stars:       {review.get('stars', 'N/A')}")
    print(f"Date:        {review.get('date', 'N/A')}")
    print("-" * 80)
    print("REVIEW TEXT:")
    print("-" * 80)
    print(review.get("text", "[No text]"))
    print("-" * 80)
    print(f"CURRENT JUDGMENT ({topic}):")
    print("-" * 80)

    # Display relevant judgment fields based on topic
    if topic.startswith("G1_allergy"):
        fields = ["is_relevant", "incident_severity", "assurance_claim", "staff_response", "account_type"]
    elif topic.startswith("G1_dietary"):
        fields = ["is_relevant", "incident_severity", "accommodation_level", "staff_knowledge", "restriction_type"]
    elif topic.startswith("G1_hygiene"):
        fields = ["is_relevant", "issue_severity", "illness_reported", "observation_type"]
    else:
        # Generic - show all non-underscore fields
        fields = [k for k in judgment.keys() if not k.startswith("_") and k not in ["review_id", "date", "stars", "useful"]]

    confidence = judgment.get("_confidence", {})
    for field in fields:
        val = judgment.get(field, "N/A")
        conf = confidence.get(field, "N/A") if isinstance(confidence, dict) else "N/A"
        if isinstance(conf, float):
            conf = f"{conf:.2f}"
        print(f"  {field}: {val} (confidence: {conf})")

    print("=" * 80)


def verify_single(
    review_id: str,
    topic: str,
    dataset: dict,
    cache: dict,
    overrides: dict,
    index: int = 0,
    total: int = 0,
) -> dict | None:
    """Verify a single review. Returns override entry if needed, None otherwise."""
    # Find the review in dataset
    review_data = dataset.get(review_id)
    if not review_data:
        print(f"[ERROR] Review {review_id} not found in dataset")
        return None

    # Get judgment
    business_id = review_data["business_id"]
    judgment = get_judgment(cache, topic, business_id)
    if not judgment:
        print(f"[ERROR] No judgment found for {topic}::{business_id}")
        return None

    # Verify review_id matches
    if judgment.get("review_id") != review_id:
        # Search for the correct key
        for key, val in cache.get("aggregated", {}).items():
            if key.startswith(topic + "::") and isinstance(val, dict) and val.get("review_id") == review_id:
                judgment = val
                business_id = key.split("::")[1]
                review_data["business_id"] = business_id
                break
        else:
            print(f"[ERROR] Could not find judgment for review {review_id}")
            return None

    display_review(review_data, judgment, topic, index, total)

    print("\nIs this judgment CORRECT?")
    print("  [y] Yes, correct")
    print("  [n] No, needs override")
    print("  [s] Skip (not sure)")
    print("  [q] Quit verification")

    while True:
        choice = input("\nChoice [y/n/s/q]: ").strip().lower()
        if choice in ["y", "n", "s", "q"]:
            break
        print("Invalid choice. Please enter y, n, s, or q.")

    if choice == "q":
        return "QUIT"
    if choice in ["y", "s"]:
        return None

    # Need override - collect corrected values
    print("\n" + "-" * 40)
    print("CORRECTION ENTRY")
    print("-" * 40)

    override = {
        "review_id": review_id,
        "business_id": business_id,
        "original": {},
        "corrected": {},
        "reason": "",
    }

    # For G1_allergy, the most common correction is incident_severity
    if topic.startswith("G1_allergy"):
        current_severity = judgment.get("incident_severity", "unknown")
        print(f"\nCurrent incident_severity: {current_severity}")
        print("Correct value? [none/mild/moderate/severe] (or press Enter to keep):")
        new_val = input("  > ").strip().lower()
        if new_val and new_val in ["none", "mild", "moderate", "severe"]:
            override["original"]["incident_severity"] = current_severity
            override["corrected"]["incident_severity"] = new_val

        # Other fields to potentially correct
        for field in ["staff_response", "assurance_claim"]:
            current = judgment.get(field, "N/A")
            print(f"\nCurrent {field}: {current}")
            print(f"Correct value? (or press Enter to keep):")
            new_val = input("  > ").strip().lower()
            if new_val:
                override["original"][field] = current
                override["corrected"][field] = new_val

    # Get reason
    print("\nReason for correction (required):")
    reason = input("  > ").strip()
    if not reason:
        print("[SKIP] No reason provided, skipping override")
        return None

    override["reason"] = reason

    # Confirm
    print("\n" + "-" * 40)
    print("Override to add:")
    print(json.dumps(override, indent=2))
    confirm = input("\nAdd this override? [y/n]: ").strip().lower()

    if confirm == "y":
        return override
    return None


def main():
    parser = argparse.ArgumentParser(description="Manual judgment verification")
    parser.add_argument("--review-id", help="Single review ID to verify")
    parser.add_argument("--input", help="JSON file with list of reviews to verify")
    parser.add_argument("--topic", required=True, help="Topic name (e.g., G1_allergy)")
    parser.add_argument("--start", type=int, default=0, help="Start index (for continuing)")
    parser.add_argument("--k", type=int, default=200, help="K value for dataset")
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = load_dataset(args.k)
    print(f"  Loaded {len(dataset)} reviews")

    print("Loading judgment cache...")
    cache = load_judgment_cache()

    print("Loading overrides...")
    overrides = load_overrides()

    if args.review_id:
        # Single review mode
        result = verify_single(args.review_id, args.topic, dataset, cache, overrides)
        if result and result != "QUIT":
            if args.topic not in overrides:
                overrides[args.topic] = []
            overrides[args.topic].append(result)
            save_overrides(overrides)
            print(f"\n[SAVED] Override added for {args.review_id}")
    elif args.input:
        # Batch mode
        with open(args.input) as f:
            reviews = json.load(f)

        print(f"\nVerifying {len(reviews)} reviews starting from index {args.start}")

        added = 0
        for i, entry in enumerate(reviews[args.start :], start=args.start):
            review_id = entry.get("review_id")
            if not review_id:
                continue

            result = verify_single(
                review_id, args.topic, dataset, cache, overrides, index=i, total=len(reviews)
            )

            if result == "QUIT":
                print(f"\n[INFO] Quit at index {i}. Use --start {i} to continue.")
                break

            if result:
                if args.topic not in overrides:
                    overrides[args.topic] = []
                overrides[args.topic].append(result)
                save_overrides(overrides)
                added += 1
                print(f"[SAVED] Override added ({added} total)")

        print(f"\n[DONE] Added {added} overrides")
    else:
        parser.error("Either --review-id or --input is required")


if __name__ == "__main__":
    main()
