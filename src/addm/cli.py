"""CLI argument parsing."""

import argparse
from pathlib import Path
from typing import List, Optional


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ADDM experiments")

    parser.add_argument("--data", required=True, help="Path to dataset (jsonl/json)")
    parser.add_argument("--method", default="direct", help="Method name")
    parser.add_argument("--provider", default="openai", help="LLM provider")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--base-url", default="", help="LLM base URL (if needed)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max output tokens")
    parser.add_argument("--max-concurrent", type=int, default=32, help="Max concurrent LLM calls")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for async calls")
    parser.add_argument("--sequential", action="store_true", help="Disable async batching")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--run-name", default="run", help="Run name for output directory")
    parser.add_argument("--benchmark", action="store_true", help="Use benchmark results path")
    parser.add_argument("--results-dir", default="results", help="Base results directory")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--validator", default="exact", help="Validator name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--request-timeout", type=float, default=90.0, help="LLM request timeout")
    parser.add_argument("--max-retries", type=int, default=4, help="LLM retry attempts")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args(argv)
    args.data_path = Path(args.data)
    args.results_dir = Path(args.results_dir)
    return args
