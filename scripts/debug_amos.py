#!/usr/bin/env python3
"""Debug AMOS pipeline step by step.

Run with:
    .venv/bin/python scripts/debug_amos.py
"""

import asyncio
import json
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from addm.methods.amos.phase2 import FormulaSeedInterpreter, _build_extraction_prompt
from addm.llm import LLMService


# Test constants
TEST_RESTAURANT_ID = "sf-KdXHB5nxfNjl2uUpI6Q"  # Cafe Blue Moose
TEST_RESTAURANT_NAME = "Cafe Blue Moose"

# Known incident review IDs from GT
INCIDENT_REVIEW_IDS = [
    "C_9HEqYieW7TnM86OtQlqw",  # moderate, dismissive staff
    "JTeR9wzcNH5dzClfgtCOmA",  # moderate, dismissive staff
    "tbm6wNuN89MQGl4B2h5qrQ",  # moderate, dismissive staff
]


def load_formula_seed(policy_id: str) -> dict:
    """Load formula seed from cache."""
    cache_path = Path(f"results/cache/formula_seeds/{policy_id}.json")
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    raise FileNotFoundError(f"Formula seed not found: {cache_path}")


def load_reviews(restaurant_id: str, k: int = 50) -> tuple:
    """Load reviews for a restaurant.

    Returns:
        Tuple of (reviews, business_info)
    """
    dataset_path = Path(f"data/context/yelp/dataset_K{k}.jsonl")

    with open(dataset_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("business", {}).get("business_id") == restaurant_id:
                return entry.get("reviews", []), entry.get("business", {})

    raise ValueError(f"Restaurant not found: {restaurant_id}")


def test_keyword_filter(seed: dict, reviews: list) -> list:
    """Test keyword filter on reviews."""
    print("\n" + "="*60)
    print("PHASE 1.1: Testing Keyword Filter")
    print("="*60)

    keywords = seed.get("filter", {}).get("keywords", [])
    print(f"Keywords ({len(keywords)}): {keywords[:10]}...")

    # Create a minimal interpreter just for filtering
    class MockLLM:
        async def call_async_with_usage(self, *args, **kwargs):
            return '{"is_relevant": false}', {}

    interpreter = FormulaSeedInterpreter(seed, MockLLM())
    filtered = interpreter._filter_reviews(reviews)

    print(f"\nTotal reviews: {len(reviews)}")
    print(f"Filtered reviews (keyword match): {len(filtered)}")

    # Show which reviews passed the filter
    print("\nFiltered review IDs:")
    for r in filtered[:10]:
        rid = r.get("review_id", "")
        is_incident = "⚠️ GT INCIDENT" if rid in INCIDENT_REVIEW_IDS else ""
        print(f"  - {rid} {is_incident}")

    # Check if GT incident reviews passed the filter
    print("\nChecking GT incident reviews:")
    for incident_id in INCIDENT_REVIEW_IDS:
        found = any(r.get("review_id") == incident_id for r in filtered)
        status = "✅ PASSED" if found else "❌ MISSED"
        print(f"  {incident_id}: {status}")

        if not found:
            # Find the review and show its text
            for r in reviews:
                if r.get("review_id") == incident_id:
                    text = r.get("text", "")[:300]
                    print(f"    Review text: {text}...")
                    break

    return filtered


async def test_extraction(seed: dict, reviews: list, llm) -> list:
    """Test extraction on filtered reviews."""
    print("\n" + "="*60)
    print("PHASE 1.2: Testing Extraction")
    print("="*60)

    # Get just the incident reviews for detailed testing
    test_reviews = [r for r in reviews if r.get("review_id") in INCIDENT_REVIEW_IDS]

    if not test_reviews:
        print("❌ No GT incident reviews found in filtered set!")
        # Try to find them in all reviews
        all_reviews, _ = load_reviews(TEST_RESTAURANT_ID)
        test_reviews = [r for r in all_reviews if r.get("review_id") in INCIDENT_REVIEW_IDS]
        print(f"Found {len(test_reviews)} incident reviews in full dataset")

    print(f"Testing extraction on {len(test_reviews)} reviews")

    fields = seed.get("extract", {}).get("fields", [])
    print(f"Extraction fields: {[f['name'] for f in fields]}")

    extractions = []
    for review in test_reviews[:3]:  # Test first 3
        review_id = review.get("review_id", "unknown")
        review_text = review.get("text", "")

        print(f"\n--- Review {review_id} ---")
        print(f"Text: {review_text[:500]}...")

        # Build and show the prompt
        prompt = _build_extraction_prompt(fields, review_text, review_id)
        print(f"\n[Extraction prompt length: {len(prompt)} chars]")

        # Call LLM for extraction
        messages = [{"role": "user", "content": prompt}]
        response, usage = await llm.call_async_with_usage(
            messages,
            context={"phase": "debug_extraction", "review_id": review_id},
        )

        print(f"\nLLM Response:\n{response}")

        # Parse response
        try:
            result = json.loads(response.strip().replace("```json", "").replace("```", "").strip())
            extractions.append(result)

            # Analyze key fields
            is_relevant = result.get("is_relevant", False)
            account_type = result.get("ACCOUNT_TYPE", "none")
            severity = result.get("INCIDENT_SEVERITY", "none")

            print(f"\nExtracted:")
            print(f"  is_relevant: {is_relevant}")
            print(f"  ACCOUNT_TYPE: {account_type}")
            print(f"  INCIDENT_SEVERITY: {severity}")

            if is_relevant and account_type == "firsthand" and severity in ["mild", "moderate", "severe"]:
                print(f"  ✅ Would count as incident!")
            else:
                print(f"  ❌ Would NOT count as incident")
                if not is_relevant:
                    print(f"     Reason: is_relevant=false")
                if account_type != "firsthand":
                    print(f"     Reason: account_type={account_type} (needs 'firsthand')")
                if severity not in ["mild", "moderate", "severe"]:
                    print(f"     Reason: severity={severity} (needs mild/moderate/severe)")

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse response: {e}")

    return extractions


def test_compute(seed: dict, extractions: list) -> dict:
    """Test compute logic with extractions."""
    print("\n" + "="*60)
    print("PHASE 1.3: Testing Compute Logic")
    print("="*60)

    # Create mock interpreter with extractions
    class MockLLM:
        async def call_async_with_usage(self, *args, **kwargs):
            return '{"is_relevant": false}', {}

    interpreter = FormulaSeedInterpreter(seed, MockLLM())
    interpreter._extractions = extractions

    # Run compute
    business = {"business_id": TEST_RESTAURANT_ID, "name": TEST_RESTAURANT_NAME}
    interpreter._execute_compute(business)

    print(f"\nComputed namespace:")
    for key, value in interpreter._namespace.items():
        print(f"  {key}: {value}")

    # Check verdict
    verdict = interpreter._namespace.get("VERDICT", "Unknown")
    n_incidents = interpreter._namespace.get("N_INCIDENTS", 0)

    print(f"\n{'='*40}")
    print(f"VERDICT: {verdict}")
    print(f"N_INCIDENTS: {n_incidents}")
    print(f"{'='*40}")

    # Expected for Cafe Blue Moose V2: Critical Risk, score=24, n_incidents=3
    expected = "Critical Risk"
    if verdict == expected:
        print(f"✅ Verdict matches expected ({expected})")
    else:
        print(f"❌ Verdict mismatch! Expected: {expected}, Got: {verdict}")

    return interpreter._namespace


async def run_full_pipeline(seed: dict, reviews: list, llm) -> dict:
    """Run full AMOS pipeline on a restaurant."""
    print("\n" + "="*60)
    print("FULL PIPELINE TEST")
    print("="*60)

    interpreter = FormulaSeedInterpreter(seed, llm)

    business = {"business_id": TEST_RESTAURANT_ID, "name": TEST_RESTAURANT_NAME}

    result = await interpreter.execute(
        reviews=reviews,
        business=business,
        query="",
        sample_id=TEST_RESTAURANT_ID,
    )

    print(f"\nFilter stats: {result.get('_filter_stats', {})}")
    print(f"Namespace: {result.get('_namespace', {})}")
    print(f"Extractions count: {len(result.get('_extractions', []))}")

    for ext in result.get("_extractions", []):
        print(f"\n  Extraction {ext.get('review_id')}:")
        for k, v in ext.items():
            if not k.startswith("_"):
                print(f"    {k}: {v}")

    verdict = result.get("_namespace", {}).get("VERDICT", "Unknown")
    print(f"\n{'='*40}")
    print(f"FINAL VERDICT: {verdict}")
    print(f"{'='*40}")

    return result


async def main():
    print("AMOS Pipeline Debugger")
    print("=" * 60)
    print(f"Test Restaurant: {TEST_RESTAURANT_NAME} ({TEST_RESTAURANT_ID})")
    print(f"GT Incident Review IDs: {INCIDENT_REVIEW_IDS}")

    # Load formula seed
    policy_id = "G1_allergy_V2"
    seed = load_formula_seed(policy_id)
    print(f"\nLoaded formula seed: {policy_id}")

    # Load reviews
    reviews, business = load_reviews(TEST_RESTAURANT_ID, k=50)
    print(f"Loaded {len(reviews)} reviews")
    print(f"Business: {business.get('name')}")

    # Phase 1.1: Test keyword filter
    filtered = test_keyword_filter(seed, reviews)

    # Initialize LLM for extraction tests
    llm = LLMService()

    # Phase 1.2: Test extraction on GT incident reviews
    extractions = await test_extraction(seed, filtered, llm)

    # Phase 1.3: Test compute logic
    if extractions:
        namespace = test_compute(seed, extractions)

    # Full pipeline test
    print("\n\n" + "="*60)
    print("Running full pipeline...")
    print("="*60)
    result = await run_full_pipeline(seed, reviews, llm)


if __name__ == "__main__":
    asyncio.run(main())
