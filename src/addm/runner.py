"""Experiment runner."""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Dict, Any, List

from addm.cli import parse_args
from addm.data.registry import DatasetRegistry
from addm.eval.evaluator import Evaluator
from addm.eval.validators import ExactMatchValidator
from addm.experiments.manager import ExperimentManager
from addm.llm import LLMService
from addm.methods import build_method_registry
from addm.utils.async_utils import gather_with_concurrency
from addm.utils.logging import setup_logging
from addm.utils.usage import usage_tracker
from addm.utils.debug_logger import DebugLogger, set_debug_logger


async def _run_async(args) -> Dict[str, Any]:
    log = setup_logging(args.verbose)
    registry = DatasetRegistry()
    dataset = registry.load(args.data_path)
    if args.limit:
        dataset.samples = dataset.samples[: args.limit]

    method_registry = build_method_registry()
    method_cls = method_registry.get(args.method)
    method = method_cls()

    llm = LLMService()
    llm.configure(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
    )

    # Create run paths first so we can set up debug logging
    manager = ExperimentManager(
        run_name=args.run_name,
        results_dir=args.results_dir,
        benchmark=args.benchmark,
        method=args.method,
        dataset=dataset.name,
        model=args.model,
    )
    run_paths = manager.create_run()

    # Generate a unique run_id for usage tracking
    run_id = run_paths.root.name

    # Set up debug logger (writes to run_paths.debug_dir)
    debug_logger = DebugLogger(output_dir=run_paths.root)
    set_debug_logger(debug_logger)

    # Clear usage tracker for this run (filter by run_id at end)
    # Note: We use run_id in context to filter, not clearing globally

    log.info("Running %s on %d samples", args.method, len(dataset.samples))

    async def run_sample(sample):
        result = await method.run_sample(sample, llm)
        result["method"] = args.method
        result["provider"] = args.provider
        result["model"] = args.model
        return result

    if args.sequential:
        outputs = []
        for sample in dataset.samples:
            outputs.append(await run_sample(sample))
    else:
        tasks = [run_sample(sample) for sample in dataset.samples]
        outputs = await gather_with_concurrency(args.max_concurrent, tasks)

    metrics = {}
    if args.eval:
        validator = ExactMatchValidator()
        evaluator = Evaluator(validator)
        metrics = evaluator.evaluate(dataset, outputs)
        log.info("Accuracy: %.3f", metrics.get("accuracy", 0.0))

    # Compute usage summary from results
    usage_summary = {
        "run_id": run_id,
        "total_samples": len(outputs),
        "total_calls": sum(r.get("llm_calls", 0) for r in outputs),
        "total_prompt_tokens": sum(r.get("prompt_tokens", 0) for r in outputs),
        "total_completion_tokens": sum(r.get("completion_tokens", 0) for r in outputs),
        "total_tokens": sum(r.get("total_tokens", 0) for r in outputs),
        "total_cost_usd": sum(r.get("cost_usd", 0.0) for r in outputs),
        "total_latency_ms": sum(r.get("latency_ms", 0.0) for r in outputs),
    }

    # Add by-model breakdown from global tracker
    global_summary = usage_tracker.get_summary()
    if global_summary.get("by_model"):
        usage_summary["by_model"] = global_summary["by_model"]

    # Save all outputs
    config = {
        "method": args.method,
        "dataset": dataset.name,
        "provider": args.provider,
        "model": args.model,
        "limit": args.limit,
        "seed": args.seed,
        "run_name": args.run_name,
        "benchmark": args.benchmark,
        "data_path": str(args.data_path),
    }
    manager.save_config(run_paths.config_path, config)
    manager.save_results(run_paths.results_path, outputs)
    if metrics:
        manager.save_metrics(run_paths.metrics_path, metrics)
    manager.save_usage(run_paths.usage_path, usage_summary)

    # Flush debug logs
    debug_logger.flush()

    log.info("Results saved to %s", run_paths.root)
    log.info(
        "Usage: %d tokens ($%.4f)",
        usage_summary["total_tokens"],
        usage_summary["total_cost_usd"],
    )

    return {
        "paths": run_paths,
        "metrics": metrics,
        "usage": usage_summary,
    }


def run() -> None:
    args = parse_args()
    random.seed(args.seed)
    asyncio.run(_run_async(args))
