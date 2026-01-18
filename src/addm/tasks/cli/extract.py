"""
CLI: Extract L0 judgments from reviews.

Usage:
    # Legacy task-based extraction (single model)
    python -m addm.tasks.cli.extract --task G1a --domain yelp --k 50 --limit 5

    # Policy-based extraction (multi-model for GT)
    python -m addm.tasks.cli.extract --topic G1_allergy --k 50 --mode 24hrbatch

    # Or derive topic from policy
    python -m addm.tasks.cli.extract --policy G1_allergy_V2 --k 50 --mode 24hrbatch
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
    REQUIRED_RUNS,
    JudgmentCache,
    PolicyJudgmentCache,
    build_extraction_prompt,
    parse_extraction_response,
    validate_judgment,
)
from addm.tasks.policy_gt import (
    aggregate_judgments,
    build_l0_schema_from_topic,
    compute_term_hash,
    get_topic_from_policy_id,
    load_term_library,
)
from addm.utils.cron import install_cron_job, remove_cron_job


# =============================================================================
# Custom ID encoding/decoding
# =============================================================================


def _build_review_custom_id(task_id: str, business_id: str, review_id: str) -> str:
    """Build custom ID for legacy task-based extraction."""
    return f"review::{task_id}::{business_id}::{review_id}"


def _parse_review_custom_id(custom_id: str) -> Tuple[str, str, str]:
    """Parse legacy custom ID."""
    parts = custom_id.split("::")
    if len(parts) != 4 or parts[0] != "review":
        raise ValueError(f"Invalid custom_id: {custom_id}")
    _, task_id, business_id, review_id = parts
    return task_id, business_id, review_id


def _build_policy_custom_id(
    topic: str, business_id: str, review_id: str, model: str, run: int
) -> str:
    """Build custom ID for policy-based multi-model extraction."""
    return f"policy::{topic}::{business_id}::{review_id}::{model}::run{run}"


def _parse_policy_custom_id(
    custom_id: str,
) -> Tuple[str, str, str, str, int]:
    """Parse policy custom ID.

    Returns: (topic, business_id, review_id, model, run)
    """
    parts = custom_id.split("::")
    if len(parts) != 6 or parts[0] != "policy":
        raise ValueError(f"Invalid policy custom_id: {custom_id}")
    _, topic, business_id, review_id, model, run_str = parts
    # run_str is "run1", "run2", etc.
    run = int(run_str.replace("run", ""))
    return topic, business_id, review_id, model, run


# =============================================================================
# Helpers
# =============================================================================


def _get_batch_response_text(item: Dict[str, Any]) -> Optional[str]:
    response = item.get("response") or {}
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        return message.get("content")
    return None


def _build_cron_command(args: argparse.Namespace, batch_id: str) -> str:
    """Build cron command for legacy task extraction."""
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


def _build_policy_cron_command(args: argparse.Namespace, batch_id: str, topic: str) -> str:
    """Build cron command for policy-based extraction."""
    repo_root = Path.cwd().resolve()
    cmd = [
        sys.executable,
        "-m",
        "addm.tasks.cli.extract",
        "--topic",
        topic,
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
    cmd.extend(["--provider", args.provider])
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


# =============================================================================
# Legacy task-based extraction
# =============================================================================


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


async def main_task_async(args: argparse.Namespace) -> None:
    """Main async entry point for legacy task-based extraction."""
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


# =============================================================================
# Policy-based multi-model extraction
# =============================================================================


def _get_policy_cache_path(domain: str) -> Path:
    """Get cache path for policy-based extraction."""
    return Path(f"data/tasks/{domain}/policy_cache.json")


async def main_policy_async(args: argparse.Namespace, topic: str) -> None:
    """Main async entry point for policy-based multi-model extraction."""
    print(f"Policy-based extraction for topic: {topic}")

    # Load term library and build L0 schema
    library = load_term_library()
    l0_schema = build_l0_schema_from_topic(topic, library)
    term_hash = compute_term_hash(l0_schema)
    print(f"L0 schema fields: {list(l0_schema.keys())}")
    print(f"Term hash: {term_hash}")

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

    # Initialize policy cache
    cache_path = _get_policy_cache_path(args.domain)
    cache = PolicyJudgmentCache(cache_path)

    # Check term hash for version mismatch
    if not cache.check_term_hash(topic, term_hash):
        meta = cache.get_topic_metadata(topic)
        old_hash = meta.get("term_hash", "unknown") if meta else "unknown"
        print(f"[WARN] Term definition changed! Old hash: {old_hash}, new hash: {term_hash}")
        print("       Existing cache may be stale. Use --invalidate to clear.")
        if args.invalidate:
            count = cache.invalidate_topic(topic)
            print(f"       Invalidated {count} cache entries for {topic}")
        else:
            print("       Proceeding with existing cache (add --invalidate to clear)")

    # Store term hash metadata
    cache.set_topic_metadata(topic, term_hash, REQUIRED_RUNS)

    # Multi-model configuration
    models_config = REQUIRED_RUNS
    total_runs = sum(models_config.values())  # 9 runs per review
    print(f"Multi-model config: {models_config} ({total_runs} runs per review)")

    initial_agg = cache.count_aggregated(topic)
    print(f"Cache has {initial_agg} aggregated judgments for {topic}")

    if args.mode == "ondemand":
        # Ondemand mode for testing - runs all models sequentially
        print("[INFO] Running in ondemand mode (for testing)")
        print("       For production, use: --mode 24hrbatch")

        llm = LLMService()
        if args.dry_run:
            def mock_responder(messages):
                return '{"is_allergy_related": false}'
            llm.configure(provider="mock")
            llm.set_mock_responder(mock_responder)
        else:
            llm.configure(
                provider=args.provider,
                model="gpt-5-nano",  # Start with cheapest for ondemand
                temperature=0.0,
                max_concurrent=args.concurrency,
            )

        total_raw = 0
        total_aggregated = 0

        for i, restaurant in enumerate(restaurants):
            business = restaurant.get("business", {})
            biz_id = business.get("business_id", "")
            name = business.get("name", "Unknown")

            print(f"[{i+1}/{len(restaurants)}] {name}")

            for review in restaurant.get("reviews", []):
                review_id = review.get("review_id", "")
                if not review_id:
                    continue

                needed = cache.needs_extraction(topic, review_id, models_config)
                if not needed:
                    continue

                review_text = review.get("text", "")[:2000]
                prompt = build_extraction_prompt(
                    l0_schema, review_text, review_id, task_description="relevant information"
                )

                # Run each needed model/run
                for model, run in needed:
                    # Reconfigure LLM for this model
                    if not args.dry_run:
                        llm.configure(
                            provider=args.provider,
                            model=model,
                            temperature=0.0,
                            max_concurrent=1,
                        )

                    try:
                        response = await llm.call([{"role": "user", "content": prompt}])
                        judgment = parse_extraction_response(response)
                        judgment = validate_judgment(judgment, l0_schema)

                        judgment["review_id"] = review_id
                        judgment["date"] = review.get("date", "")
                        judgment["stars"] = review.get("stars", 0)
                        judgment["useful"] = review.get("useful", 0)

                        cache.set_raw(topic, review_id, model, run, judgment)
                        total_raw += 1

                        if args.verbose:
                            is_rel = judgment.get("is_allergy_related", False)
                            print(f"    {review_id[:8]}... {model} run{run} rel={is_rel}")

                    except Exception as e:
                        print(f"    Error {review_id[:8]} {model} run{run}: {e}")
                        continue

                # Aggregate if quota satisfied
                if cache.is_quota_satisfied(topic, review_id, models_config):
                    raw_judgments = cache.get_raw_by_review(topic, review_id)
                    aggregated = aggregate_judgments(raw_judgments, l0_schema)
                    cache.set_aggregated(topic, review_id, aggregated)
                    total_aggregated += 1

            # Save periodically
            if (i + 1) % 2 == 0:
                cache.save()

        cache.save()
        final_agg = cache.count_aggregated(topic)
        print(f"\nDone. Added {total_raw} raw, {total_aggregated} aggregated.")
        print(f"Cache now has {final_agg} aggregated judgments for {topic}")
        return

    if args.provider != "openai":
        print("24hrbatch mode only supports provider=openai")
        return

    if args.batch_id:
        # Fetch and process batch results
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
            marker = f"ADDM_POLICY_BATCH_{args.batch_id}"
            try:
                remove_cron_job(marker)
                print(f"Removed cron job for {args.batch_id}")
            except Exception as exc:
                print(f"[WARN] Failed to remove cron job: {exc}")
            return

        output_bytes = batch_client.download_file(output_file_id)
        review_index = _index_reviews(restaurants)

        # Process batch results
        total_raw = 0
        reviews_touched: set = set()

        for line in output_bytes.splitlines():
            if not line:
                continue
            item = json.loads(line)
            custom_id = item.get("custom_id", "")

            try:
                batch_topic, business_id, review_id, model, run = _parse_policy_custom_id(
                    custom_id
                )
            except ValueError:
                continue

            if batch_topic != topic:
                continue

            response_text = _get_batch_response_text(item)
            if response_text is None:
                continue

            review = review_index.get((business_id, review_id))
            if not review:
                continue

            try:
                judgment = parse_extraction_response(response_text)
                judgment = validate_judgment(judgment, l0_schema)

                # Add review metadata
                judgment["review_id"] = review_id
                judgment["date"] = review.get("date", "")
                judgment["stars"] = review.get("stars", 0)
                judgment["useful"] = review.get("useful", 0)

                # Store raw judgment
                cache.set_raw(topic, review_id, model, run, judgment)
                total_raw += 1
                reviews_touched.add(review_id)

            except Exception as e:
                if args.verbose:
                    print(f"  Error parsing {review_id}: {e}")
                continue

        print(f"Processed {total_raw} raw extractions for {len(reviews_touched)} reviews")

        # Aggregate completed reviews
        total_aggregated = 0
        for review_id in reviews_touched:
            if cache.is_quota_satisfied(topic, review_id, models_config):
                raw_judgments = cache.get_raw_by_review(topic, review_id)
                aggregated = aggregate_judgments(raw_judgments, l0_schema)
                cache.set_aggregated(topic, review_id, aggregated)
                total_aggregated += 1

        cache.save()
        final_agg = cache.count_aggregated(topic)
        print(f"\nDone. Added {total_aggregated} aggregated judgments.")
        print(f"Cache now has {final_agg} aggregated judgments for {topic}")

        marker = f"ADDM_POLICY_BATCH_{args.batch_id}"
        try:
            remove_cron_job(marker)
            print(f"Removed cron job for {args.batch_id}")
        except Exception as exc:
            print(f"[WARN] Failed to remove cron job: {exc}")
        return

    # Submit batch - generate multi-model requests
    request_items = []
    reviews_to_extract = 0

    for restaurant in restaurants:
        business = restaurant.get("business", {})
        biz_id = business.get("business_id", "")

        for review in restaurant.get("reviews", []):
            review_id = review.get("review_id", "")
            if not review_id:
                continue

            # Check what's needed
            needed = cache.needs_extraction(topic, review_id, models_config)
            if not needed:
                continue

            reviews_to_extract += 1
            review_text = review.get("text", "")[:2000]

            # Build prompt (same for all models)
            prompt = build_extraction_prompt(
                l0_schema, review_text, review_id, task_description="relevant information"
            )
            messages = [{"role": "user", "content": prompt}]

            # Generate request for each needed model/run
            for model, run in needed:
                custom_id = _build_policy_custom_id(topic, biz_id, review_id, model, run)
                request_items.append(
                    build_chat_batch_item(
                        custom_id=custom_id,
                        model=model,
                        messages=messages,
                        temperature=0.0,
                    )
                )

    if not request_items:
        print("No reviews need extraction. All quotas satisfied.")
        return

    print(f"Submitting {len(request_items)} requests for {reviews_to_extract} reviews")

    batch_client = BatchClient()
    input_file_id = batch_client.upload_batch_file(request_items)
    batch_id = batch_client.submit_batch(input_file_id)
    print(f"Submitted batch: {batch_id}")

    marker = f"ADDM_POLICY_BATCH_{batch_id}"
    cron_line = f"*/5 * * * * {_build_policy_cron_command(args, batch_id, topic)} # {marker}"
    try:
        install_cron_job(cron_line, marker)
        print(f"Installed cron job for batch {batch_id}")
    except Exception as exc:
        print(f"[WARN] Failed to install cron job: {exc}")


# =============================================================================
# All topics definition
# =============================================================================

ALL_TOPICS = [
    # G1: Customer Safety
    "G1_allergy",
    "G1_dietary",
    "G1_hygiene",
    # G2: Customer Experience
    "G2_romance",
    "G2_business",
    "G2_group",
    # G3: Customer Value
    "G3_price_worth",
    "G3_hidden_costs",
    "G3_time_value",
    # G4: Owner Operations
    "G4_server",
    "G4_kitchen",
    "G4_environment",
    # G5: Owner Performance
    "G5_capacity",
    "G5_execution",
    "G5_consistency",
    # G6: Owner Strategy
    "G6_uniqueness",
    "G6_comparison",
    "G6_loyalty",
]


# =============================================================================
# Multi-topic batch extraction
# =============================================================================


async def main_all_topics_async(args: argparse.Namespace) -> None:
    """Extract all topics in one batch."""
    topics = ALL_TOPICS
    print(f"Extracting ALL {len(topics)} topics for GT generation")
    print(f"Topics: {', '.join(topics)}")

    # Load term library once
    library = load_term_library()

    # Build schemas and hashes for all topics
    topic_schemas: Dict[str, Dict[str, Dict[str, str]]] = {}
    topic_hashes: Dict[str, str] = {}

    for topic in topics:
        try:
            schema = build_l0_schema_from_topic(topic, library)
            topic_schemas[topic] = schema
            topic_hashes[topic] = compute_term_hash(schema)
        except Exception as e:
            print(f"[WARN] Skipping {topic}: {e}")
            continue

    print(f"Loaded schemas for {len(topic_schemas)} topics")

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
    cache_path = _get_policy_cache_path(args.domain)
    cache = PolicyJudgmentCache(cache_path)

    # Check term hashes and set metadata
    for topic, term_hash in topic_hashes.items():
        if not cache.check_term_hash(topic, term_hash):
            meta = cache.get_topic_metadata(topic)
            old_hash = meta.get("term_hash", "unknown") if meta else "unknown"
            print(f"[WARN] Term hash changed for {topic}: {old_hash} -> {term_hash}")
            if args.invalidate:
                count = cache.invalidate_topic(topic)
                print(f"       Invalidated {count} entries")
        cache.set_topic_metadata(topic, term_hash, REQUIRED_RUNS)

    # Multi-model config
    models_config = REQUIRED_RUNS
    total_runs = sum(models_config.values())
    print(f"Multi-model config: {models_config} ({total_runs} runs per review per topic)")

    if args.mode == "ondemand":
        print("[ERROR] --all requires 24hrbatch mode (too many requests for ondemand)")
        print("        Use: --mode 24hrbatch")
        return

    if args.provider != "openai":
        print("24hrbatch mode only supports provider=openai")
        return

    if args.batch_id:
        # Process batch results for all topics
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
            print(f"[WARN] No output file for batch {args.batch_id}")
            marker = f"ADDM_ALL_BATCH_{args.batch_id}"
            try:
                remove_cron_job(marker)
            except Exception:
                pass
            return

        output_bytes = batch_client.download_file(output_file_id)
        review_index = _index_reviews(restaurants)

        # Process results
        total_raw = 0
        reviews_by_topic: Dict[str, set] = {t: set() for t in topic_schemas}

        for line in output_bytes.splitlines():
            if not line:
                continue
            item = json.loads(line)
            custom_id = item.get("custom_id", "")

            try:
                batch_topic, business_id, review_id, model, run = _parse_policy_custom_id(
                    custom_id
                )
            except ValueError:
                continue

            if batch_topic not in topic_schemas:
                continue

            response_text = _get_batch_response_text(item)
            if response_text is None:
                continue

            review = review_index.get((business_id, review_id))
            if not review:
                continue

            l0_schema = topic_schemas[batch_topic]

            try:
                judgment = parse_extraction_response(response_text)
                judgment = validate_judgment(judgment, l0_schema)
                judgment["review_id"] = review_id
                judgment["date"] = review.get("date", "")
                judgment["stars"] = review.get("stars", 0)
                judgment["useful"] = review.get("useful", 0)

                cache.set_raw(batch_topic, review_id, model, run, judgment)
                total_raw += 1
                reviews_by_topic[batch_topic].add(review_id)

            except Exception:
                continue

        print(f"Processed {total_raw} raw extractions")

        # Aggregate completed reviews per topic
        total_aggregated = 0
        for topic, review_ids in reviews_by_topic.items():
            l0_schema = topic_schemas[topic]
            for review_id in review_ids:
                if cache.is_quota_satisfied(topic, review_id, models_config):
                    raw_judgments = cache.get_raw_by_review(topic, review_id)
                    aggregated = aggregate_judgments(raw_judgments, l0_schema)
                    cache.set_aggregated(topic, review_id, aggregated)
                    total_aggregated += 1

        cache.save()
        print(f"\nDone. Added {total_aggregated} aggregated judgments across all topics.")

        for topic in topic_schemas:
            count = cache.count_aggregated(topic)
            print(f"  {topic}: {count} aggregated")

        marker = f"ADDM_ALL_BATCH_{args.batch_id}"
        try:
            remove_cron_job(marker)
            print(f"Removed cron job for {args.batch_id}")
        except Exception as exc:
            print(f"[WARN] Failed to remove cron job: {exc}")
        return

    # Submit batch for all topics
    request_items = []
    reviews_to_extract = 0

    for restaurant in restaurants:
        business = restaurant.get("business", {})
        biz_id = business.get("business_id", "")

        for review in restaurant.get("reviews", []):
            review_id = review.get("review_id", "")
            if not review_id:
                continue

            review_text = review.get("text", "")[:2000]

            # For each topic
            for topic, l0_schema in topic_schemas.items():
                needed = cache.needs_extraction(topic, review_id, models_config)
                if not needed:
                    continue

                reviews_to_extract += 1
                prompt = build_extraction_prompt(
                    l0_schema, review_text, review_id, task_description="relevant information"
                )
                messages = [{"role": "user", "content": prompt}]

                for model, run in needed:
                    custom_id = _build_policy_custom_id(topic, biz_id, review_id, model, run)
                    request_items.append(
                        build_chat_batch_item(
                            custom_id=custom_id,
                            model=model,
                            messages=messages,
                            temperature=0.0,
                        )
                    )

    if not request_items:
        print("No reviews need extraction. All quotas satisfied.")
        return

    n_reviews = len(restaurants) * sum(1 for _ in restaurants[0].get("reviews", []))
    print(f"Submitting {len(request_items)} requests")
    print(f"  ({len(topic_schemas)} topics × {reviews_to_extract} review-topics × {total_runs} runs)")

    batch_client = BatchClient()
    input_file_id = batch_client.upload_batch_file(request_items)
    batch_id = batch_client.submit_batch(input_file_id)
    print(f"Submitted batch: {batch_id}")

    # Build cron command for --all mode
    repo_root = Path.cwd().resolve()
    cmd = [
        sys.executable,
        "-m",
        "addm.tasks.cli.extract",
        "--all",
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
    cmd.extend(["--provider", args.provider])
    if args.verbose:
        cmd.append("--verbose")
    command = " ".join(shlex.quote(c) for c in cmd)
    cron_cmd = f"cd {shlex.quote(str(repo_root))} && {command}"

    marker = f"ADDM_ALL_BATCH_{batch_id}"
    cron_line = f"*/5 * * * * {cron_cmd} # {marker}"
    try:
        install_cron_job(cron_line, marker)
        print(f"Installed cron job for batch {batch_id}")
    except Exception as exc:
        print(f"[WARN] Failed to install cron job: {exc}")


# =============================================================================
# Main entry point
# =============================================================================


async def main_async(args: argparse.Namespace) -> None:
    """Main async entry point - routes to task or policy extraction."""
    # Determine extraction mode
    if args.all:
        # Extract all topics
        await main_all_topics_async(args)
    elif args.topic:
        # Explicit topic
        await main_policy_async(args, args.topic)
    elif args.policy:
        # Derive topic from policy
        topic = get_topic_from_policy_id(args.policy)
        await main_policy_async(args, topic)
    elif args.task:
        # Legacy task-based extraction
        await main_task_async(args)
    else:
        # Default: extract all topics
        print("No target specified, defaulting to --all (all topics)")
        args.all = True
        await main_all_topics_async(args)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Extract L0 judgments from reviews")

    # Extraction target (mutually exclusive)
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--task", type=str, help="Task ID (e.g., G1a) - legacy mode")
    target_group.add_argument(
        "--topic", type=str, help="Topic ID (e.g., G1_allergy) - single topic"
    )
    target_group.add_argument(
        "--policy", type=str, help="Policy ID (e.g., G1_allergy_V2) - derives topic"
    )
    target_group.add_argument(
        "--all", action="store_true", help="Extract ALL topics (G1-G6) - default if no target"
    )

    # Common options
    parser.add_argument("--domain", type=str, default="yelp", help="Domain (default: yelp)")
    parser.add_argument("--k", type=int, default=200, help="Dataset K value (default: 200)")
    parser.add_argument("--limit", type=int, help="Limit number of restaurants")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider")
    parser.add_argument(
        "--model", type=str, default="gpt-5-nano", help="LLM model (task mode only)"
    )
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent requests")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no API calls)")
    parser.add_argument(
        "--mode",
        type=str,
        default="24hrbatch",
        choices=["ondemand", "24hrbatch"],
        help="LLM execution mode (default: 24hrbatch)",
    )
    parser.add_argument(
        "--batch-id", type=str, default=None, help="Batch ID for fetch-only runs"
    )
    parser.add_argument(
        "--invalidate",
        action="store_true",
        help="Invalidate cache if term hash changed (policy mode)",
    )

    args = parser.parse_args()

    # Validate required arguments
    if not args.task and not args.topic and not args.policy:
        parser.error("Must specify one of: --task, --topic, or --policy")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
