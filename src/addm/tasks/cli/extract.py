"""
CLI: Extract L0 judgments from reviews.

Usage:
    python -m addm.tasks.cli.extract --task G1a --domain yelp --k 50 --limit 5
"""

import argparse
import asyncio
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from addm.llm import LLMService
from addm.llm_batch import BatchClient, build_chat_batch_item
from addm.tasks.base import load_task
from addm.tasks.extraction import (
    JudgmentCache,
    build_extraction_prompt,
    parse_extraction_response,
    validate_judgment,
)
from addm.utils.cron import install_cron_job, remove_cron_job


def _build_review_custom_id(task_id: str, business_id: str, review_id: str) -> str:
    return f"review::{task_id}::{business_id}::{review_id}"


def _parse_review_custom_id(custom_id: str) -> Tuple[str, str, str]:
    parts = custom_id.split("::")
    if len(parts) != 4 or parts[0] != "review":
        raise ValueError(f"Invalid custom_id: {custom_id}")
    _, task_id, business_id, review_id = parts
    return task_id, business_id, review_id


def _get_batch_response_text(item: Dict[str, Any]) -> Optional[str]:
    response = item.get("response") or {}
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        return message.get("content")
    return None


def _build_cron_command(args: argparse.Namespace, batch_id: str) -> str:
    repo_root = Path.cwd().resolve()
    cmd = [
        sys.executable,
        "-m",
        "addm.tasks.cli.extract",
        "--task",
        args.task,
        "--domain",
        args.domain,
        "--k",
        str(args.k),
        "--mode",
        "24hrbatch",
        "--batch-id",
        batch_id,
    ]
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    cmd.extend(["--provider", args.provider, "--model", args.model])
    if args.verbose:
        cmd.append("--verbose")
    command = " ".join(shlex.quote(c) for c in cmd)
    return f"cd {shlex.quote(str(repo_root))} && {command}"


def _index_reviews(
    restaurants: List[Dict[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for restaurant in restaurants:
        business = restaurant.get("business", {})
        biz_id = business.get("business_id", "")
        for review in restaurant.get("reviews", []):
            review_id = review.get("review_id", "")
            if biz_id and review_id:
                index[(biz_id, review_id)] = review
    return index


def _get_batch_field(batch: Any, key: str) -> Optional[Any]:
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


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
    dataset_path = Path(f"data/context/{args.domain}/dataset_K{args.k}.jsonl")
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    with open(dataset_path) as f:
        restaurants = [json.loads(line) for line in f]

    if args.limit:
        restaurants = restaurants[: args.limit]

    print(f"Processing {len(restaurants)} restaurants from {dataset_path.name}")

    # Initialize cache
    cache = JudgmentCache(task.cache_path)
    initial_count = cache.count(task.task_id)
    print(f"Cache has {initial_count} existing judgments for {task.task_id}")

    if args.mode == "ondemand":
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
        return

    if args.provider != "openai":
        print("24hrbatch mode only supports provider=openai")
        return

    if args.batch_id:
        batch_client = BatchClient()
        batch = batch_client.get_batch(args.batch_id)
        status = _get_batch_field(batch, "status")
        if status not in {"completed", "failed", "expired", "cancelled"}:
            print(f"Batch {args.batch_id} status: {status}")
            return

        output_file_id = _get_batch_field(batch, "output_file_id")
        error_file_id = _get_batch_field(batch, "error_file_id")
        if error_file_id:
            print(f"[WARN] Batch has error file: {error_file_id}")

        if not output_file_id:
            print(f"[WARN] No output file available for batch {args.batch_id}")
            marker = f"ADDM_BATCH_{args.batch_id}"
            try:
                remove_cron_job(marker)
                print(f"Removed cron job for {args.batch_id}")
            except Exception as exc:
                print(f"[WARN] Failed to remove cron job: {exc}")
            return

        output_bytes = batch_client.download_file(output_file_id)
        review_index = _index_reviews(restaurants)

        total_new = 0
        for line in output_bytes.splitlines():
            if not line:
                continue
            item = json.loads(line)
            custom_id = item.get("custom_id", "")
            try:
                task_id, business_id, review_id = _parse_review_custom_id(custom_id)
            except ValueError:
                continue
            if task_id != task.task_id:
                continue
            response_text = _get_batch_response_text(item)
            if response_text is None:
                continue

            review = review_index.get((business_id, review_id))
            if not review:
                continue

            try:
                judgment = parse_extraction_response(response_text)
                judgment = validate_judgment(judgment, task.l0_schema)

                judgment["review_id"] = review_id
                judgment["date"] = review.get("date", "")
                judgment["stars"] = review.get("stars", 0)
                judgment["useful"] = review.get("useful", 0)

                cache.set(task.task_id, review_id, judgment)
                total_new += 1
            except Exception:
                continue

        cache.save()
        final_count = cache.count(task.task_id)
        print(f"\nDone. Added {total_new} new judgments.")
        print(f"Cache now has {final_count} judgments for {task.task_id}")

        marker = f"ADDM_BATCH_{args.batch_id}"
        try:
            remove_cron_job(marker)
            print(f"Removed cron job for {args.batch_id}")
        except Exception as exc:
            print(f"[WARN] Failed to remove cron job: {exc}")
        return

    # Submit batch
    to_extract: List[Tuple[str, Dict[str, Any]]] = []
    for restaurant in restaurants:
        business = restaurant.get("business", {})
        biz_id = business.get("business_id", "")
        for review in restaurant.get("reviews", []):
            review_id = review.get("review_id", "")
            if review_id and not cache.has(task.task_id, review_id):
                to_extract.append((biz_id, review))

    if not to_extract:
        print("No reviews need extraction.")
        return

    request_items = []
    for biz_id, review in to_extract:
        review_id = review.get("review_id", "")
        review_text = review.get("text", "")[:2000]
        prompt = build_extraction_prompt(task.l0_schema, review_text, review_id)
        messages = [{"role": "user", "content": prompt}]
        custom_id = _build_review_custom_id(task.task_id, biz_id, review_id)
        request_items.append(
            build_chat_batch_item(
                custom_id=custom_id,
                model=args.model,
                messages=messages,
                temperature=0.0,
            )
        )

    batch_client = BatchClient()
    input_file_id = batch_client.upload_batch_file(request_items)
    batch_id = batch_client.submit_batch(input_file_id)
    print(f"Submitted batch: {batch_id}")

    marker = f"ADDM_BATCH_{batch_id}"
    cron_line = f"*/5 * * * * {_build_cron_command(args, batch_id)} # {marker}"
    try:
        install_cron_job(cron_line, marker)
        print(f"Installed cron job for batch {batch_id}")
    except Exception as exc:
        print(f"[WARN] Failed to install cron job: {exc}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Extract L0 judgments from reviews")
    parser.add_argument("--task", type=str, required=True, help="Task ID (e.g., G1a)")
    parser.add_argument("--domain", type=str, default="yelp", help="Domain (default: yelp)")
    parser.add_argument("--k", type=int, default=50, help="Dataset K value (default: 50)")
    parser.add_argument("--limit", type=int, help="Limit number of restaurants")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider")
    parser.add_argument("--model", type=str, default="gpt-5-nano", help="LLM model")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent requests")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no API calls)")
    parser.add_argument(
        "--mode",
        type=str,
        default="ondemand",
        choices=["ondemand", "24hrbatch"],
        help="LLM execution mode",
    )
    parser.add_argument("--batch-id", type=str, default=None, help="Batch ID for fetch-only runs")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
