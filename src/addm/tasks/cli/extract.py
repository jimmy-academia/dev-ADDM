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
import uuid
from datetime import datetime
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
from addm.utils.output import output

# Batch size limit (OpenAI max is 50K, use 40K for safety margin)
MAX_BATCH_SIZE = 40000


def _log_batch_error(topic: str, custom_id: str, error: dict) -> None:
    """Log batch error to audit file for diagnostics."""
    log_path = Path(f"data/answers/yelp/batch_errors_{topic}.jsonl")
    entry = {
        "timestamp": datetime.now().isoformat(),
        "custom_id": custom_id,
        "error": error,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _print_batch_submitted(identifier: str, topic: str = None) -> None:
    """Print batch submitted message with status check instructions."""
    output.success(f"Batch submitted: {identifier}")
    output.info("Batch API processes requests within 24 hours.")
    output.info("To check status, re-run this command (it will poll and process when ready).")
    if topic:
        output.console.print(f"  [dim].venv/bin/python -m addm.tasks.cli.extract --topic {topic}[/dim]")
    output.info("Or use the wrapper script:")
    output.console.print(f"  [dim]./scripts/run_g1_allergy.sh[/dim]")


def _get_manifest_path(domain: str, manifest_id: str) -> Path:
    """Get path to batch manifest file."""
    return Path(f"data/answers/{domain}/batch_manifest_{manifest_id}.json")


def _get_manifest_log_path(domain: str, manifest_id: str) -> Path:
    """Get path to batch manifest log file (sidecar)."""
    return Path(f"data/answers/{domain}/batch_manifest_{manifest_id}.log")


def _get_batch_log_path(domain: str, batch_id: str) -> Path:
    """Get path to single-batch log file."""
    return Path(f"data/answers/{domain}/batch_{batch_id[:16]}.log")


def _delete_batch_log(domain: str, batch_id: str) -> None:
    """Delete single-batch log file."""
    log_path = _get_batch_log_path(domain, batch_id)
    if log_path.exists():
        log_path.unlink()


def _save_manifest(domain: str, manifest_id: str, manifest: Dict[str, Any]) -> None:
    """Save batch manifest to disk."""
    path = _get_manifest_path(domain, manifest_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def _load_manifest(domain: str, manifest_id: str) -> Optional[Dict[str, Any]]:
    """Load batch manifest from disk."""
    path = _get_manifest_path(domain, manifest_id)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _delete_manifest(domain: str, manifest_id: str) -> None:
    """Delete batch manifest file and its sidecar log."""
    path = _get_manifest_path(domain, manifest_id)
    if path.exists():
        path.unlink()
    log_path = _get_manifest_log_path(domain, manifest_id)
    if log_path.exists():
        log_path.unlink()


def _find_existing_manifests(domain: str, topic: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Find existing manifest files for a topic.

    Returns list of (manifest_id, manifest_data) tuples.
    """
    import glob
    pattern = f"data/answers/{domain}/batch_manifest_{topic}_*.json"
    results = []
    for path in glob.glob(pattern):
        manifest_id = Path(path).stem.replace("batch_manifest_", "")
        try:
            with open(path) as f:
                manifest = json.load(f)
            results.append((manifest_id, manifest))
        except Exception:
            continue
    return results


def _check_and_report_status(args, topic: str) -> bool:
    """Check for existing batches and report status.

    Returns True if there's an existing job (caller should exit),
    False if no existing job (caller should proceed with new extraction).
    """
    from addm.llm_batch import BatchClient

    # Check for existing manifest files
    manifests = _find_existing_manifests(args.domain, topic)

    if not manifests:
        return False  # No existing job, proceed

    output.print(f"\n{'='*60}")
    output.print(f"EXISTING BATCH JOB DETECTED FOR: {topic}")
    output.print(f"{'='*60}\n")

    batch_client = BatchClient()

    for manifest_id, manifest in manifests:
        output.print(f"\nManifest: {manifest_id}")
        output.print(f"  Created: {manifest.get('created_at', 'unknown')}")
        output.print(f"  Batches: {len(manifest.get('batches', []))}")

        # Check status of each batch
        completed = 0
        in_progress = 0
        failed = 0
        total_requests = 0
        completed_requests = 0
        failed_requests = 0

        for batch_info in manifest.get("batches", []):
            batch_id = batch_info.get("batch_id", "")
            if batch_info.get("processed"):
                completed += 1
                continue

            try:
                batch = batch_client.get_batch(batch_id)
                status = batch.status if hasattr(batch, "status") else batch.get("status", "unknown")
                req_counts = batch.request_counts if hasattr(batch, "request_counts") else {}
                total = getattr(req_counts, "total", 0) if hasattr(req_counts, "total") else req_counts.get("total", 0)
                done = getattr(req_counts, "completed", 0) if hasattr(req_counts, "completed") else req_counts.get("completed", 0)
                fail = getattr(req_counts, "failed", 0) if hasattr(req_counts, "failed") else req_counts.get("failed", 0)

                total_requests += total
                completed_requests += done
                failed_requests += fail

                model = batch_info.get("model", "?")

                if status == "completed":
                    completed += 1
                    output.success(f"  {batch_id[:20]}... [{model}] COMPLETED ({done}/{total})")
                elif status in ("failed", "expired", "cancelled"):
                    failed += 1
                    output.error(f"  {batch_id[:20]}... [{model}] FAILED: {status}")
                else:
                    in_progress += 1
                    output.status(f"  {batch_id[:20]}... [{model}] {status} ({done}/{total})")
            except Exception as e:
                output.error(f"  {batch_id[:20]}... ERROR: {e}")

        output.print(f"\n  Summary: {completed} completed, {in_progress} in-progress, {failed} failed")
        if total_requests > 0:
            output.print(f"  Requests: {completed_requests}/{total_requests} completed, {failed_requests} failed")

        if in_progress > 0:
            output.info("\n  Status: STILL PROCESSING - re-run to check again")
        elif failed > 0 and completed > 0:
            output.warn("\n  Status: PARTIALLY FAILED - some batches failed")
            output.print("  Action: Check failed batches, then re-run to submit missing requests")
        elif failed > 0:
            output.error("\n  Status: ALL FAILED - check errors above")
        else:
            output.success("\n  Status: ALL COMPLETE - run again to process results")

    output.print(f"\n{'='*60}")
    output.print("To check batch status:")
    output.print(f"  .venv/bin/python -m addm.tasks.cli.extract --topic {topic}")
    output.print("\nTo force new extraction (clears existing):")
    output.print(f"  rm data/answers/{args.domain}/batch_manifest_{topic}_*.json")
    output.print(f"{'='*60}\n")

    return True  # Existing job found, caller should exit


def _split_into_batches(items: List[Any], max_size: int) -> List[List[Any]]:
    """Split list into chunks of max_size."""
    return [items[i:i + max_size] for i in range(0, len(items), max_size)]


def _split_by_model_then_size(
    items: List[Dict[str, Any]], max_size: int
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    Split batch items by model first, then by size.

    OpenAI Batch API requires each batch to contain only one model.

    Args:
        items: List of batch request items (each has 'body' with 'model')
        max_size: Maximum items per batch

    Returns:
        List of (model, items) tuples, where each items list is <= max_size
    """
    from collections import defaultdict

    # Group by model
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        model = item.get("body", {}).get("model", "unknown")
        by_model[model].append(item)

    # Split each model's items by size
    result: List[Tuple[str, List[Dict[str, Any]]]] = []
    for model in sorted(by_model.keys()):
        model_items = by_model[model]
        chunks = _split_into_batches(model_items, max_size)
        for chunk in chunks:
            result.append((model, chunk))

    return result


def parse_models_config(models_str: str) -> Dict[str, int]:
    """Parse --models flag into dict.

    Format: 'model:runs,model:runs,...'
    Example: 'gpt-5-nano:1' or 'gpt-5-nano:5,gpt-5-mini:3,gpt-5.1:1'
    """
    config = {}
    for part in models_str.split(","):
        part = part.strip()
        if ":" not in part:
            raise ValueError(f"Invalid model config: {part} (expected model:runs)")
        model, runs = part.rsplit(":", 1)
        config[model.strip()] = int(runs.strip())
    return config


def get_models_config(args: argparse.Namespace) -> Dict[str, int]:
    """Get models config from args or use default."""
    if args.models:
        return parse_models_config(args.models)
    return REQUIRED_RUNS


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


def _check_batch_item_error(item: Dict[str, Any], custom_id: str, topic: str) -> bool:
    """Check if batch item has an error and log it.

    Returns True if there was an error (caller should skip this item),
    False if no error.
    """
    error = item.get("error")
    if error:
        error_code = error.get("code", "unknown")
        error_msg = error.get("message", "no message")
        output.warn(f"  Request failed [{custom_id[:30]}...]: {error_code}")
        _log_batch_error(topic, custom_id, error)
        return True
    return False


def _get_batch_response_text(item: Dict[str, Any]) -> Optional[str]:
    response = item.get("response") or {}
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        return message.get("content")
    return None


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
        review_text = review.get("text", "")[:8000]
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
                output.print(f"  {review_id[:8]}... allergy={is_allergy}")

        except Exception as e:
            output.error(f"  Error parsing {review_id}: {e}")
            continue

    return new_extractions


async def main_task_async(args: argparse.Namespace) -> None:
    """Main async entry point for legacy task-based extraction."""
    # Load task config
    task = load_task(args.task, args.domain)
    output.info(f"Loaded task: {task.task_id} ({task.domain})")
    output.print(f"L0 fields: {list(task.l0_schema.keys())}")

    # Load dataset
    dataset_path = Path(f"data/context/{args.domain}/dataset_K{args.k}.jsonl")
    if not dataset_path.exists():
        output.error(f"Dataset not found: {dataset_path}")
        return

    with open(dataset_path) as f:
        restaurants = [json.loads(line) for line in f]

    if args.limit:
        restaurants = restaurants[: args.limit]

    output.info(f"Processing {len(restaurants)} restaurants from {dataset_path.name}")

    # Initialize cache
    cache = JudgmentCache(task.cache_path)
    initial_count = cache.count(task.task_id)
    output.info(f"Cache has {initial_count} existing judgments for {task.task_id}")

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
                max_concurrent=args.concurrency,
            )

        # Extract judgments
        total_new = 0
        for i, restaurant in enumerate(restaurants):
            business = restaurant.get("business", {})
            name = business.get("name", "Unknown")
            n_reviews = len(restaurant.get("reviews", []))

            output.status(f"[{i+1}/{len(restaurants)}] {name} ({n_reviews} reviews)")

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
        output.success(f"\nDone. Added {total_new} new judgments.")
        output.info(f"Cache now has {final_count} judgments for {task.task_id}")
        return

    if args.provider != "openai":
        output.error("24hrbatch mode only supports provider=openai")
        return

    if args.batch_id:
        batch_client = BatchClient()
        batch = batch_client.get_batch(args.batch_id)
        status = _get_batch_field(batch, "status")
        if status not in {"completed", "failed", "expired", "cancelled"}:
            if not args.quiet:
                output.status(f"Batch {args.batch_id} status: {status}")
            return

        output_file_id = _get_batch_field(batch, "output_file_id")
        error_file_id = _get_batch_field(batch, "error_file_id")
        if error_file_id:
            output.warn(f"Batch has error file: {error_file_id}")

        if not output_file_id:
            output.warn(f"No output file available for batch {args.batch_id}")
            _delete_batch_log(args.domain, args.batch_id)
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
            if _check_batch_item_error(item, custom_id, task.task_id):
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
        output.success(f"\nDone. Added {total_new} new judgments.")
        output.info(f"Cache now has {final_count} judgments for {task.task_id}")

        _delete_batch_log(args.domain, args.batch_id)
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
        output.info("No reviews need extraction.")
        return

    request_items = []
    for biz_id, review in to_extract:
        review_id = review.get("review_id", "")
        review_text = review.get("text", "")[:8000]
        prompt = build_extraction_prompt(task.l0_schema, review_text, review_id)
        messages = [{"role": "user", "content": prompt}]
        custom_id = _build_review_custom_id(task.task_id, biz_id, review_id)
        request_items.append(
            build_chat_batch_item(
                custom_id=custom_id,
                model=args.model,
                messages=messages,
            )
        )

    batch_client = BatchClient()
    input_file_id = batch_client.upload_batch_file(request_items)
    batch_id = batch_client.submit_batch(input_file_id)
    output.success(f"Submitted batch: {batch_id}")

    _print_batch_submitted(batch_id)


# =============================================================================
# Policy-based multi-model extraction
# =============================================================================


def _get_judgement_cache_path(domain: str) -> Path:
    """Get cache path for L0 judgement extraction."""
    return Path(f"data/answers/{domain}/judgement_cache.json")


async def main_policy_async(args: argparse.Namespace, topic: str) -> None:
    """Main async entry point for policy-based multi-model extraction."""
    # Check for existing batch jobs (only if not already processing a manifest/batch)
    if not args.manifest_id and not args.batch_id and args.mode == "24hrbatch":
        if _check_and_report_status(args, topic):
            return  # Existing job found, status reported, exit

    output.info(f"Policy-based extraction for topic: {topic}")

    # Load term library and build L0 schema
    library = load_term_library()
    l0_schema = build_l0_schema_from_topic(topic, library)
    term_hash = compute_term_hash(l0_schema)
    output.print(f"L0 schema fields: {list(l0_schema.keys())}")
    output.print(f"Term hash: {term_hash}")

    # Load dataset
    dataset_path = Path(f"data/context/{args.domain}/dataset_K{args.k}.jsonl")
    if not dataset_path.exists():
        output.error(f"Dataset not found: {dataset_path}")
        return

    with open(dataset_path) as f:
        restaurants = [json.loads(line) for line in f]

    if args.limit:
        restaurants = restaurants[: args.limit]

    output.info(f"Processing {len(restaurants)} restaurants from {dataset_path.name}")

    # Initialize policy cache
    cache_path = _get_judgement_cache_path(args.domain)
    cache = PolicyJudgmentCache(cache_path)

    # Check term hash for version mismatch
    if not cache.check_term_hash(topic, term_hash):
        meta = cache.get_topic_metadata(topic)
        old_hash = meta.get("term_hash", "unknown") if meta else "unknown"
        output.warn(f"Term definition changed! Old hash: {old_hash}, new hash: {term_hash}")
        output.print("       Existing cache may be stale. Use --invalidate to clear.")
        if args.invalidate:
            count = cache.invalidate_topic(topic)
            output.info(f"       Invalidated {count} cache entries for {topic}")
        else:
            output.print("       Proceeding with existing cache (add --invalidate to clear)")

    # Store term hash metadata
    # Multi-model configuration (from --models flag or default)
    models_config = get_models_config(args)
    total_runs = sum(models_config.values())
    output.info(f"Multi-model config: {models_config} ({total_runs} runs per review)")

    cache.set_topic_metadata(topic, term_hash, models_config)

    initial_agg = cache.count_aggregated(topic)
    output.info(f"Cache has {initial_agg} aggregated judgments for {topic}")

    # Show per-model cache status
    raw_by_model = cache.count_raw_by_model(topic)
    total_reviews = sum(len(r.get("reviews", [])) for r in restaurants)
    output.info("Cache status by model:")
    for model, required_runs in models_config.items():
        cached = raw_by_model.get(model, 0)
        expected = total_reviews * required_runs
        if cached >= expected:
            output.print(f"  {model}: {cached}/{expected} ✓ complete")
        else:
            output.print(f"  {model}: {cached}/{expected} (need {expected - cached})")

    if args.mode == "ondemand":
        # Ondemand mode - immediate execution (no 50% batch discount)
        output.info("Running in ondemand mode (immediate, no batch discount)")

        # Check if anything needs extraction (use cache status already computed above)
        raw_by_model = cache.count_raw_by_model(topic)
        all_complete = True
        for model, required_runs in models_config.items():
            cached = raw_by_model.get(model, 0)
            expected = total_reviews * required_runs
            if cached < expected:
                all_complete = False
                break

        if all_complete:
            output.success("All extractions complete - nothing to do!")
            return

        output.print("Iterating restaurants to find incomplete reviews...")

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
                max_concurrent=args.concurrency,
            )

        # Track progress per model
        success_by_model: Dict[str, int] = {model: 0 for model in models_config.keys()}
        error_by_model: Dict[str, int] = {model: 0 for model in models_config.keys()}
        total_aggregated = 0

        for i, restaurant in enumerate(restaurants):
            business = restaurant.get("business", {})
            biz_id = business.get("business_id", "")
            name = business.get("name", "Unknown")

            output.status(f"[{i+1}/{len(restaurants)}] {name}")

            for review in restaurant.get("reviews", []):
                review_id = review.get("review_id", "")
                if not review_id:
                    continue

                needed = cache.needs_extraction(topic, review_id, models_config)
                if not needed:
                    continue

                review_text = review.get("text", "")[:8000]
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
                            max_concurrent=1,
                        )

                    try:
                        response = await llm.call_async([{"role": "user", "content": prompt}])
                        judgment = parse_extraction_response(response)
                        judgment = validate_judgment(judgment, l0_schema)

                        judgment["review_id"] = review_id
                        judgment["date"] = review.get("date", "")
                        judgment["stars"] = review.get("stars", 0)
                        judgment["useful"] = review.get("useful", 0)

                        cache.set_raw(topic, review_id, model, run, judgment)
                        success_by_model[model] += 1
                        output.print(f"  + Extracted: {review_id[:12]}... {model} run{run}")

                        if args.verbose:
                            is_rel = judgment.get("is_allergy_related", False)
                            output.print(f"    {review_id[:8]}... {model} run{run} rel={is_rel}")

                    except Exception as e:
                        error_by_model[model] += 1
                        output.error(f"    Error {review_id[:8]} {model} run{run}: {e}")
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
        output.print(f"Cache saved to {cache.cache_path}")

        # Final summary
        total_success = sum(success_by_model.values())
        total_errors = sum(error_by_model.values())
        output.print()
        output.success(f"Done. Added {total_success} raw extractions, {total_aggregated} aggregated.")

        if total_errors > 0:
            output.warn(f"Errors by model: {error_by_model}")

        # Show final cache status
        output.info("Final cache status:")
        final_raw_by_model = cache.count_raw_by_model(topic)
        for model, required_runs in models_config.items():
            cached = final_raw_by_model.get(model, 0)
            expected = total_reviews * required_runs
            if cached >= expected:
                output.print(f"  {model}: {cached}/{expected} ✓ complete")
            else:
                output.print(f"  {model}: {cached}/{expected} (still need {expected - cached})")

        final_agg = cache.count_aggregated(topic)
        output.info(f"Cache now has {final_agg} aggregated judgments for {topic}")
        return

    if args.provider != "openai":
        output.error("24hrbatch mode only supports provider=openai")
        return

    if args.batch_id:
        # Fetch and process batch results
        batch_client = BatchClient()
        batch = batch_client.get_batch(args.batch_id)
        status = _get_batch_field(batch, "status")
        if status not in {"completed", "failed", "expired", "cancelled"}:
            if not args.quiet:
                output.status(f"Batch {args.batch_id} status: {status}")
            return

        output_file_id = _get_batch_field(batch, "output_file_id")
        error_file_id = _get_batch_field(batch, "error_file_id")
        if error_file_id:
            output.warn(f"Batch has error file: {error_file_id}")

        if not output_file_id:
            output.warn(f"No output file available for batch {args.batch_id}")
            _delete_batch_log(args.domain, args.batch_id)
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

            if _check_batch_item_error(item, custom_id, topic):
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
                    output.error(f"  Error parsing {review_id}: {e}")
                continue

        output.info(f"Processed {total_raw} raw extractions for {len(reviews_touched)} reviews")

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
        output.success(f"\nDone. Added {total_aggregated} aggregated judgments.")
        output.info(f"Cache now has {final_agg} aggregated judgments for {topic}")

        _delete_batch_log(args.domain, args.batch_id)
        return

    # Handle manifest-based multi-batch processing
    if args.manifest_id:
        manifest = _load_manifest(args.domain, args.manifest_id)
        if not manifest:
            # Manifest file deleted
            output.warn(f"Manifest not found: {args.manifest_id} (likely already processed or deleted)")
            _delete_manifest(args.domain, args.manifest_id)  # Clean up any leftover files
            return

        batch_client = BatchClient()
        all_complete = True

        if not args.quiet:
            output.status(f"Checking {len(manifest['batches'])} batches...")

        for batch_info in manifest["batches"]:
            batch_id = batch_info["batch_id"]
            if batch_info.get("processed"):
                continue

            batch = batch_client.get_batch(batch_id)
            status = _get_batch_field(batch, "status")

            if status not in {"completed", "failed", "expired", "cancelled"}:
                if not args.quiet:
                    output.status(f"  Batch {batch_id[:20]}... status: {status}")
                all_complete = False
                continue

            if status != "completed":
                output.warn(f"  Batch {batch_id[:20]}... status: {status}")
                batch_info["processed"] = True
                batch_info["status"] = status
                continue

            # Process this batch
            output_file_id = _get_batch_field(batch, "output_file_id")
            if not output_file_id:
                output.warn(f"  Batch {batch_id[:20]}... no output file")
                batch_info["processed"] = True
                batch_info["status"] = "no_output"
                continue

            # Log batch status
            req_counts = _get_batch_field(batch, "request_counts")
            total_reqs = getattr(req_counts, "total", 0) if req_counts else 0
            failed_reqs = getattr(req_counts, "failed", 0) if req_counts else 0
            completed_reqs = getattr(req_counts, "completed", 0) if req_counts else 0

            if failed_reqs > 0:
                output.warn(f"  Batch {batch_id[:20]}... {failed_reqs}/{total_reqs} requests failed")

            output.status(f"  Processing batch {batch_id[:20]}... ({completed_reqs}/{total_reqs} succeeded)")
            output_bytes = batch_client.download_file(output_file_id)
            review_index = _index_reviews(restaurants)

            batch_raw = 0
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

                if _check_batch_item_error(item, custom_id, topic):
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
                    judgment["review_id"] = review_id
                    judgment["date"] = review.get("date", "")
                    judgment["stars"] = review.get("stars", 0)
                    judgment["useful"] = review.get("useful", 0)

                    cache.set_raw(topic, review_id, model, run, judgment)
                    batch_raw += 1
                except Exception:
                    continue

            output.status(f"    Processed {batch_raw} raw extractions")
            batch_info["processed"] = True
            batch_info["status"] = "completed"
            batch_info["raw_count"] = batch_raw

        # Save updated manifest
        _save_manifest(args.domain, args.manifest_id, manifest)
        cache.save()

        if not all_complete:
            if not args.quiet:
                output.status("\nSome batches still processing. Will check again.")
            return

        # All batches complete - run aggregation
        output.success("\nAll batches complete. Running aggregation...")
        total_aggregated = 0
        all_review_ids = set()

        # Collect all review IDs from restaurants
        for restaurant in restaurants:
            for review in restaurant.get("reviews", []):
                review_id = review.get("review_id", "")
                if review_id:
                    all_review_ids.add(review_id)

        for review_id in all_review_ids:
            if cache.is_quota_satisfied(topic, review_id, models_config):
                if not cache.has_aggregated(topic, review_id):
                    raw_judgments = cache.get_raw_by_review(topic, review_id)
                    aggregated = aggregate_judgments(raw_judgments, l0_schema)
                    cache.set_aggregated(topic, review_id, aggregated)
                    total_aggregated += 1

        cache.save()
        final_agg = cache.count_aggregated(topic)
        output.success(f"\nDone. Added {total_aggregated} aggregated judgments.")
        output.info(f"Cache now has {final_agg} aggregated judgments for {topic}")

        # Cleanup - always delete manifest (cache is source of truth)
        _delete_manifest(args.domain, args.manifest_id)

        # Check if extraction is complete
        missing = 0
        for restaurant in restaurants:
            for review in restaurant.get("reviews", []):
                review_id = review.get("review_id", "")
                if review_id and cache.needs_extraction(topic, review_id, models_config):
                    missing += 1

        if missing > 0:
            output.warn(f"{missing} reviews still need extraction. Re-run to submit missing requests.")
        else:
            output.success("All extractions complete!")
        return

    # Submit batch(es) - generate multi-model requests
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
            review_text = review.get("text", "")[:8000]

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
                    )
                )

    if not request_items:
        output.info("No reviews need extraction. All quotas satisfied.")
        return

    output.info(f"Total requests: {len(request_items)} for {reviews_to_extract} reviews")

    # Split by model first (OpenAI requires single model per batch), then by size
    model_batches = _split_by_model_then_size(request_items, MAX_BATCH_SIZE)
    output.info(f"Splitting into {len(model_batches)} batch(es) by model (max {MAX_BATCH_SIZE} per batch)")
    for model, items in model_batches:
        output.print(f"  {model}: {len(items)} requests")

    if args.dry_run:
        output.info("Dry run - skipping batch submission")
        return

    batch_client = BatchClient()

    if len(model_batches) == 1:
        # Single batch - use simple flow
        model, items = model_batches[0]
        input_file_id = batch_client.upload_batch_file(items)
        batch_id = batch_client.submit_batch(input_file_id)
        output.success(f"Submitted batch: {batch_id} ({model})")

        _print_batch_submitted(batch_id, topic)
    else:
        # Multiple batches - use manifest
        manifest_id = f"{topic}_{uuid.uuid4().hex[:8]}"
        manifest = {
            "manifest_id": manifest_id,
            "topic": topic,
            "domain": args.domain,
            "k": args.k,
            "models_config": models_config,
            "total_requests": len(request_items),
            "created_at": datetime.now().isoformat(),
            "batches": [],
        }

        for i, (model, chunk) in enumerate(model_batches):
            output.status(f"  Submitting batch {i+1}/{len(model_batches)} ({model}: {len(chunk)} requests)...")
            input_file_id = batch_client.upload_batch_file(chunk)
            batch_id = batch_client.submit_batch(input_file_id)
            manifest["batches"].append({
                "batch_id": batch_id,
                "model": model,
                "chunk_index": i,
                "request_count": len(chunk),
                "processed": False,
            })
            output.print(f"    Batch ID: {batch_id}")

        _save_manifest(args.domain, manifest_id, manifest)
        output.success(f"\nCreated manifest: {manifest_id}")

        _print_batch_submitted(manifest_id, topic)


# =============================================================================
# All topics definition (imported from shared constants)
# =============================================================================

from addm.tasks.constants import ALL_TOPICS


# =============================================================================
# Multi-topic batch extraction
# =============================================================================


async def main_all_topics_async(args: argparse.Namespace) -> None:
    """Extract all topics in one batch."""
    topics = ALL_TOPICS
    output.info(f"Extracting ALL {len(topics)} topics for GT generation")
    output.print(f"Topics: {', '.join(topics)}")

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
            output.warn(f"Skipping {topic}: {e}")
            continue

    output.info(f"Loaded schemas for {len(topic_schemas)} topics")

    # Load dataset
    dataset_path = Path(f"data/context/{args.domain}/dataset_K{args.k}.jsonl")
    if not dataset_path.exists():
        output.error(f"Dataset not found: {dataset_path}")
        return

    with open(dataset_path) as f:
        restaurants = [json.loads(line) for line in f]

    if args.limit:
        restaurants = restaurants[: args.limit]

    output.info(f"Processing {len(restaurants)} restaurants from {dataset_path.name}")

    # Initialize cache
    cache_path = _get_judgement_cache_path(args.domain)
    cache = PolicyJudgmentCache(cache_path)

    # Check term hashes and set metadata
    for topic, term_hash in topic_hashes.items():
        if not cache.check_term_hash(topic, term_hash):
            meta = cache.get_topic_metadata(topic)
            old_hash = meta.get("term_hash", "unknown") if meta else "unknown"
            output.warn(f"Term hash changed for {topic}: {old_hash} -> {term_hash}")
            if args.invalidate:
                count = cache.invalidate_topic(topic)
                output.info(f"       Invalidated {count} entries")

    # Multi-model config (from --models flag or default)
    models_config = get_models_config(args)
    total_runs = sum(models_config.values())
    output.info(f"Multi-model config: {models_config} ({total_runs} runs per review per topic)")

    # Store metadata for all topics
    for topic, term_hash in topic_hashes.items():
        cache.set_topic_metadata(topic, term_hash, models_config)

    if args.mode == "ondemand":
        output.error("--all requires 24hrbatch mode (too many requests for ondemand)")
        output.print("        Use: --mode 24hrbatch")
        return

    if args.provider != "openai":
        output.error("24hrbatch mode only supports provider=openai")
        return

    # Handle manifest-based multi-batch processing for --all mode
    if args.manifest_id:
        manifest = _load_manifest(args.domain, args.manifest_id)
        if not manifest:
            output.error(f"Manifest not found: {args.manifest_id}")
            return

        batch_client = BatchClient()
        all_complete = True

        if not args.quiet:
            output.status(f"Checking {len(manifest['batches'])} batches...")

        review_index = _index_reviews(restaurants)
        total_raw = 0
        reviews_by_topic: Dict[str, set] = {t: set() for t in topic_schemas}

        for batch_info in manifest["batches"]:
            batch_id = batch_info["batch_id"]
            if batch_info.get("processed"):
                continue

            batch = batch_client.get_batch(batch_id)
            status = _get_batch_field(batch, "status")

            if status not in {"completed", "failed", "expired", "cancelled"}:
                if not args.quiet:
                    output.status(f"  Batch {batch_id[:20]}... status: {status}")
                all_complete = False
                continue

            if status != "completed":
                output.warn(f"  Batch {batch_id[:20]}... status: {status}")
                batch_info["processed"] = True
                batch_info["status"] = status
                continue

            # Process this batch
            output_file_id = _get_batch_field(batch, "output_file_id")
            if not output_file_id:
                output.warn(f"  Batch {batch_id[:20]}... no output file")
                batch_info["processed"] = True
                batch_info["status"] = "no_output"
                continue

            output.status(f"  Processing batch {batch_id[:20]}...")
            output_bytes = batch_client.download_file(output_file_id)

            batch_raw = 0
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

                if _check_batch_item_error(item, custom_id, batch_topic):
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
                    batch_raw += 1
                    total_raw += 1
                    reviews_by_topic[batch_topic].add(review_id)
                except Exception:
                    continue

            output.status(f"    Processed {batch_raw} raw extractions")
            batch_info["processed"] = True
            batch_info["status"] = "completed"
            batch_info["raw_count"] = batch_raw

        # Save updated manifest
        _save_manifest(args.domain, args.manifest_id, manifest)
        cache.save()

        if not all_complete:
            if not args.quiet:
                output.status("\nSome batches still processing. Will check again.")
            return

        # All batches complete - run aggregation
        output.success(f"\nAll batches complete. Processed {total_raw} total raw extractions.")
        output.status("Running aggregation...")

        total_aggregated = 0
        for topic, review_ids in reviews_by_topic.items():
            l0_schema = topic_schemas[topic]
            for review_id in review_ids:
                if cache.is_quota_satisfied(topic, review_id, models_config):
                    if not cache.has_aggregated(topic, review_id):
                        raw_judgments = cache.get_raw_by_review(topic, review_id)
                        aggregated = aggregate_judgments(raw_judgments, l0_schema)
                        cache.set_aggregated(topic, review_id, aggregated)
                        total_aggregated += 1

        cache.save()
        output.success(f"\nDone. Added {total_aggregated} aggregated judgments across all topics.")

        for topic in topic_schemas:
            count = cache.count_aggregated(topic)
            output.print(f"  {topic}: {count} aggregated")

        # Cleanup - always delete manifest (cache is source of truth)

        _delete_manifest(args.domain, args.manifest_id)

        # Check if extraction is complete for all topics
        missing_total = 0
        for topic, l0_schema in topic_schemas.items():
            missing = 0
            for restaurant in restaurants:
                for review in restaurant.get("reviews", []):
                    review_id = review.get("review_id", "")
                    if review_id and cache.needs_extraction(topic, review_id, models_config):
                        missing += 1
            if missing > 0:
                output.warn(f"  {topic}: {missing} reviews still need extraction")
                missing_total += missing

        if missing_total > 0:
            output.warn(f"\n{missing_total} total reviews need extraction. Re-run to submit missing requests.")
        else:
            output.success("All extractions complete!")
        return

    if args.batch_id:
        # Legacy single-batch processing (kept for backwards compatibility)
        batch_client = BatchClient()
        batch = batch_client.get_batch(args.batch_id)
        status = _get_batch_field(batch, "status")
        if status not in {"completed", "failed", "expired", "cancelled"}:
            if not args.quiet:
                output.status(f"Batch {args.batch_id} status: {status}")
            return

        output_file_id = _get_batch_field(batch, "output_file_id")
        error_file_id = _get_batch_field(batch, "error_file_id")
        if error_file_id:
            output.warn(f"Batch has error file: {error_file_id}")

        if not output_file_id:
            output.warn(f"No output file for batch {args.batch_id}")
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

            if _check_batch_item_error(item, custom_id, batch_topic):
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

        output.info(f"Processed {total_raw} raw extractions")

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
        output.success(f"Done. Added {total_aggregated} aggregated judgments across all topics.")

        for topic in topic_schemas:
            count = cache.count_aggregated(topic)
            output.print(f"  {topic}: {count} aggregated")

        _delete_batch_log(args.domain, args.batch_id)
        return

    # Submit batch(es) for all topics
    request_items = []
    reviews_to_extract = 0

    for restaurant in restaurants:
        business = restaurant.get("business", {})
        biz_id = business.get("business_id", "")

        for review in restaurant.get("reviews", []):
            review_id = review.get("review_id", "")
            if not review_id:
                continue

            review_text = review.get("text", "")[:8000]

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
                        )
                    )

    if not request_items:
        output.info("No reviews need extraction. All quotas satisfied.")
        return

    output.info(f"Total requests: {len(request_items)}")
    output.print(f"  ({len(topic_schemas)} topics × {reviews_to_extract} review-topics × {total_runs} runs)")

    # Split by model first (OpenAI requires single model per batch), then by size
    model_batches = _split_by_model_then_size(request_items, MAX_BATCH_SIZE)
    output.info(f"Splitting into {len(model_batches)} batch(es) by model (max {MAX_BATCH_SIZE} per batch)")
    for model, items in model_batches:
        output.print(f"  {model}: {len(items)} requests")

    if args.dry_run:
        output.info("Dry run - skipping batch submission")
        return

    batch_client = BatchClient()
    repo_root = Path.cwd().resolve()

    # Always use manifest for --all mode (multiple models = multiple batches)
    manifest_id = f"all_topics_{uuid.uuid4().hex[:8]}"
    manifest = {
        "manifest_id": manifest_id,
        "mode": "all_topics",
        "domain": args.domain,
        "k": args.k,
        "models_config": models_config,
        "total_requests": len(request_items),
        "created_at": datetime.now().isoformat(),
        "batches": [],
    }

    for i, (model, chunk) in enumerate(model_batches):
        output.status(f"Submitting batch {i+1}/{len(model_batches)} ({model}: {len(chunk)} requests)...")
        input_file_id = batch_client.upload_batch_file(chunk)
        batch_id = batch_client.submit_batch(input_file_id)
        manifest["batches"].append({
            "batch_id": batch_id,
            "model": model,
            "chunk_index": i,
            "request_count": len(chunk),
            "processed": False,
        })
        output.print(f"    Batch ID: {batch_id}")

        _save_manifest(args.domain, manifest_id, manifest)
        output.success(f"\nCreated manifest: {manifest_id}")

        _print_batch_submitted(manifest_id)


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
        output.info("No target specified, defaulting to --all (all topics)")
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
    parser.add_argument("--show-status", dest="quiet", action="store_false", help="Show in-progress status messages")
    parser.set_defaults(quiet=True)
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no API calls)")
    parser.add_argument(
        "--mode",
        type=str,
        default="24hrbatch",
        choices=["ondemand", "24hrbatch"],
        help="LLM execution mode (default: 24hrbatch)",
    )
    parser.add_argument(
        "--batch-id", type=str, default=None, help="Batch ID for fetch-only runs (legacy single batch)"
    )
    parser.add_argument(
        "--manifest-id", type=str, default=None, help="Manifest ID for multi-batch runs"
    )
    parser.add_argument(
        "--invalidate",
        action="store_true",
        help="Invalidate cache if term hash changed (policy mode)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Model config as model:runs,... (e.g., 'gpt-5-nano:1' or 'gpt-5-nano:5,gpt-5-mini:3,gpt-5.1:1')",
    )

    args = parser.parse_args()

    # No validation needed - defaults to --all if no target specified
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
