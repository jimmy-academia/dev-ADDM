"""
CLI: Extract L0 judgments from reviews.

Usage:
    python -m addm.tasks.cli.extract --task G1a --domain yelp --k 50 --limit 5
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

from addm.llm import LLMService
from addm.tasks.base import load_task
from addm.tasks.extraction import (
    JudgmentCache,
    build_extraction_prompt,
    parse_extraction_response,
    validate_judgment,
)


async def extract_reviews_for_restaurant(
    restaurant: Dict[str, Any],
    task_id: str,
    l0_schema: Dict[str, Dict[str, str]],
    cache: JudgmentCache,
    llm: LLMService,
    verbose: bool = False,
) -> int:
    """
    Extract L0 judgments for all reviews in a restaurant (CONCURRENT).

    Returns number of new extractions performed.
    """
    reviews = restaurant.get("reviews", [])

    # Filter to reviews that need extraction (not already cached)
    to_extract: List[Dict[str, Any]] = []
    for review in reviews:
        review_id = review.get("review_id", "")
        if review_id and not cache.has(task_id, review_id):
            to_extract.append(review)

    if not to_extract:
        return 0

    # Build all prompts
    messages_batch: List[List[Dict[str, str]]] = []
    for review in to_extract:
        review_text = review.get("text", "")[:2000]
        prompt = build_extraction_prompt(l0_schema, review_text, review.get("review_id", ""))
        messages_batch.append([{"role": "user", "content": prompt}])

    # Call LLM concurrently using batch_call
    responses = await llm.batch_call(messages_batch)

    # Process responses
    new_extractions = 0
    for review, response in zip(to_extract, responses):
        review_id = review.get("review_id", "")
        try:
            judgment = parse_extraction_response(response)
            judgment = validate_judgment(judgment, l0_schema)

            # Add review metadata
            judgment["review_id"] = review_id
            judgment["date"] = review.get("date", "")
            judgment["stars"] = review.get("stars", 0)
            judgment["useful"] = review.get("useful", 0)

            # Cache the judgment
            cache.set(task_id, review_id, judgment)
            new_extractions += 1

            if verbose:
                is_allergy = judgment.get("is_allergy_related", False)
                print(f"  {review_id[:8]}... allergy={is_allergy}")

        except Exception as e:
            print(f"  Error parsing {review_id}: {e}")
            continue

    return new_extractions


async def main_async(args: argparse.Namespace) -> None:
    """Main async entry point."""
    # Load task config
    task = load_task(args.task, args.domain)
    print(f"Loaded task: {task.task_id} ({task.domain})")
    print(f"L0 fields: {list(task.l0_schema.keys())}")

    # Load dataset
    dataset_path = Path(f"data/processed/{args.domain}/dataset_K{args.k}.jsonl")
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    with open(dataset_path) as f:
        restaurants = [json.loads(line) for line in f]

    if args.limit:
        restaurants = restaurants[: args.limit]

    print(f"Processing {len(restaurants)} restaurants from {dataset_path.name}")

    # Initialize LLM
    llm = LLMService()
    if args.dry_run:
        # Mock responses for dry run
        def mock_responder(messages):
            return '{"is_allergy_related": false}'
        llm.configure(provider="mock")
        llm.set_mock_responder(mock_responder)
    else:
        llm.configure(
            provider=args.provider,
            model=args.model,
            temperature=0.0,
            max_concurrent=args.concurrency,
        )

    # Initialize cache
    cache = JudgmentCache(task.cache_path)
    initial_count = cache.count(task.task_id)
    print(f"Cache has {initial_count} existing judgments for {task.task_id}")

    # Extract judgments
    total_new = 0
    for i, restaurant in enumerate(restaurants):
        business = restaurant.get("business", {})
        name = business.get("name", "Unknown")
        n_reviews = len(restaurant.get("reviews", []))

        print(f"[{i+1}/{len(restaurants)}] {name} ({n_reviews} reviews)")

        new_count = await extract_reviews_for_restaurant(
            restaurant,
            task.task_id,
            task.l0_schema,
            cache,
            llm,
            verbose=args.verbose,
        )
        total_new += new_count

        # Save periodically
        if (i + 1) % 5 == 0:
            cache.save()

    # Final save
    cache.save()
    final_count = cache.count(task.task_id)
    print(f"\nDone. Added {total_new} new judgments.")
    print(f"Cache now has {final_count} judgments for {task.task_id}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Extract L0 judgments from reviews")
    parser.add_argument("--task", type=str, required=True, help="Task ID (e.g., G1a)")
    parser.add_argument("--domain", type=str, default="yelp", help="Domain (default: yelp)")
    parser.add_argument("--k", type=int, default=50, help="Dataset K value (default: 50)")
    parser.add_argument("--limit", type=int, help="Limit number of restaurants")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent requests")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no API calls)")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
