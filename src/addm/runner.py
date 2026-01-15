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
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
    )

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

    manager = ExperimentManager(
        run_name=args.run_name,
        results_dir=args.results_dir,
        benchmark=args.benchmark,
        method=args.method,
        dataset=dataset.name,
        model=args.model,
    )
    run_paths = manager.create_run()
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

    log.info("Results saved to %s", run_paths.root)
    return {
        "paths": run_paths,
        "metrics": metrics,
    }


def run() -> None:
    args = parse_args()
    random.seed(args.seed)
    asyncio.run(_run_async(args))
