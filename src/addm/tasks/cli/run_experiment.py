"""
CLI: Run experiment evaluation (all methods: direct, rlm, rag, amos).

Usage:
    # Run ALL 72 policies (omit --policy)
    .venv/bin/python -m addm.tasks.cli.run_experiment --k 25 --method amos --dev

    # Run ALL 72 policies with verdict samples
    .venv/bin/python -m addm.tasks.cli.run_experiment --k 25 --method amos --dev --sample

    # Single policy
    .venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 -n 100

    # Multiple policies (comma-separated)
    .venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V0,G1_allergy_V1 --dev

    # Dev mode (saves to results/dev/, no quota)
    .venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 -n 5 --dev

    # Force run even if quota is met
    .venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 -n 100 --force

    # Legacy task-based (loads from data/answers/yelp/G1a_prompt.txt)
    .venv/bin/python -m addm.tasks.cli.run_experiment --task G1a -n 5

Output directories:
    (default):   results/{method}/{policy_id}_K{k}/run_N/results.json (benchmark mode)
    --dev:       results/dev/{YYYYMMDD}_{run_name}/results.json
"""

import argparse
import asyncio
import json
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from addm.eval import (
    VERDICT_TO_ORDINAL,
    compute_ordinal_auprc,
    compute_unified_metrics,  # Legacy (kept for backward compat)
    compute_evaluation_metrics,  # New simplified metrics
    normalize_verdict,
)
from addm.llm import LLMService
from addm.llm_batch import BatchClient, build_chat_batch_item
from addm.methods.rlm import eval_restaurant_rlm
from addm.methods.cot import SYSTEM_PROMPT as COT_SYSTEM_PROMPT
from addm.tasks.base import load_task
from addm.utils.async_utils import gather_with_concurrency
from addm.utils.debug_logger import DebugLogger, get_debug_logger, set_debug_logger
from addm.utils.item_logger import ItemLogger, get_item_logger, set_item_logger
from addm.utils.logging import setup_logging  # Legacy, kept for backward compat
from addm.utils.output import output
from addm.utils.results_manager import ResultsManager, get_results_manager
from addm.utils.usage import compute_cost


# Import ALL_POLICIES and expand_policies from shared constants
from addm.tasks.constants import ALL_POLICIES, expand_policies

# Mapping from policy topic to legacy task ID for ground truth comparison
# Policy naming: G{group}_{topic}_V{version} -> Task: G{group}{variant}
POLICY_TO_TASK = {
    "allergy": "a",    # G1_allergy_V* -> G1a
    "dietary": "e",    # G1_dietary_V* -> G1e
    "hygiene": "i",    # G1_hygiene_V* -> G1i
    # Add more mappings as policies are developed
}


# =============================================================================
# Batch Manifest Functions
# =============================================================================

def _get_manifest_path(method: str, benchmark_id: str) -> Path:
    """Get manifest path: results/{method}/{benchmark_id}/batch_manifest.json"""
    return Path(f"results/{method}/{benchmark_id}/batch_manifest.json")


def _save_manifest(method: str, benchmark_id: str, manifest: dict) -> None:
    """Save batch manifest to track pending job."""
    path = _get_manifest_path(method, benchmark_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def _load_manifest(method: str, benchmark_id: str) -> Optional[dict]:
    """Load batch manifest if it exists."""
    path = _get_manifest_path(method, benchmark_id)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _delete_manifest(method: str, benchmark_id: str) -> None:
    """Delete batch manifest after processing."""
    path = _get_manifest_path(method, benchmark_id)
    if path.exists():
        path.unlink()


def _check_batch_status(method: str, benchmark_id: str) -> tuple[str, Optional[dict]]:
    """Check if existing batch manifest exists and its status.

    Returns:
        ("none", None) - No manifest, proceed with new submission
        ("processing", manifest) - Batch still in progress
        ("ready", manifest) - Batch complete, ready to process
        ("failed", manifest) - Batch failed
    """
    manifest = _load_manifest(method, benchmark_id)
    if not manifest:
        return ("none", None)

    batch_client = BatchClient()
    batch_id = manifest.get("batch_id")
    if not batch_id:
        return ("failed", manifest)

    batch = batch_client.get_batch(batch_id)
    status = _get_batch_field(batch, "status")

    if status == "completed":
        return ("ready", manifest)
    elif status in {"failed", "expired", "cancelled"}:
        return ("failed", manifest)
    else:
        return ("processing", manifest)


def _get_batch_usage(item: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Extract token usage from batch response item."""
    response = item.get("response") or {}
    body = response.get("body") or {}
    usage = body.get("usage") or {}

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": compute_cost(model, prompt_tokens, completion_tokens),
        "latency_ms": None,  # Batch mode - no latency tracking
        "llm_calls": 1,
    }


def policy_to_task_id(policy_id: str) -> Optional[str]:
    """Convert policy ID to legacy task ID for ground truth lookup.

    Args:
        policy_id: Policy ID like "G1_allergy_V2" or path like "G1/allergy/V2"

    Returns:
        Task ID like "G1a" or None if no mapping exists
    """
    # Normalize: "G1/allergy/V2" -> "G1_allergy_V2"
    normalized = policy_id.replace("/", "_")

    # Parse: G{group}_{topic}_V{version}
    parts = normalized.split("_")
    if len(parts) < 3:
        return None

    group = parts[0]  # e.g., "G1"
    topic = parts[1]  # e.g., "allergy"

    variant = POLICY_TO_TASK.get(topic)
    if variant is None:
        return None

    return f"{group}{variant}"  # e.g., "G1a"


def load_policy_prompt(policy_id: str, domain: str = "yelp", k: int = 200) -> str:
    """Load prompt from policy-generated file.

    Args:
        policy_id: Policy ID like "G1_allergy_V2" or "G1/allergy/V2"
        domain: Domain (default: yelp)
        k: Context size (25, 50, 100, 200) - prompts are K-specific

    Returns:
        Prompt text
    """
    # Normalize policy ID: "G1/allergy/V2" -> "G1_allergy_V2"
    normalized = policy_id.replace("/", "_")

    # Try K-specific prompt first
    prompt_path = Path(f"data/query/{domain}/{normalized}_K{k}_prompt.txt")
    if not prompt_path.exists():
        # Fallback to old format without K
        prompt_path = Path(f"data/query/{domain}/{normalized}_prompt.txt")
        if not prompt_path.exists():
            raise FileNotFoundError(f"Policy prompt not found: {prompt_path}")

    return prompt_path.read_text()


def load_ground_truth(task_id: str, domain: str, k: int) -> Dict[str, str]:
    """Load ground truth verdicts by business_id."""
    gt_path = Path(f"data/answers/{domain}/{task_id}_K{k}_groundtruth.json")
    if not gt_path.exists():
        # Fallback to old format without K
        gt_path = Path(f"data/answers/{domain}/{task_id}_groundtruth.json")
        if not gt_path.exists():
            return {}

    with open(gt_path) as f:
        gt_data = json.load(f)

    # Extract verdict by business_id
    verdicts = {}
    for biz_id, data in gt_data.get("restaurants", {}).items():
        gt = data.get("ground_truth", {})
        verdicts[biz_id] = gt.get("verdict", "Unknown")

    return verdicts


PROMPT_TEMPLATE = """Context:

{context}

Agenda:
{agenda}
"""


def _build_sample_custom_id(run_id: str, business_id: str) -> str:
    return f"sample::{run_id}::{business_id}"


def _parse_sample_custom_id(custom_id: str) -> tuple[str, str]:
    parts = custom_id.split("::")
    if len(parts) != 3 or parts[0] != "sample":
        raise ValueError(f"Invalid custom_id: {custom_id}")
    _, run_id, business_id = parts
    return run_id, business_id


def _get_batch_response_text(item: Dict[str, Any]) -> Optional[str]:
    response = item.get("response") or {}
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        return message.get("content")
    return None


def load_system_prompt() -> str:
    """Load the output schema system prompt."""
    path = Path(__file__).parent.parent.parent / "query" / "prompts" / "output_schema.txt"
    if not path.exists():
        raise FileNotFoundError(f"System prompt not found: {path}")
    return path.read_text()


def load_dataset(domain: str, k: int) -> List[Dict[str, Any]]:
    """Load dataset for given K value."""
    dataset_path = Path(f"data/context/{domain}/dataset_K{k}.jsonl")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    restaurants = []
    with open(dataset_path) as f:
        for line in f:
            restaurants.append(json.loads(line))
    return restaurants


def build_prompt(restaurant: Dict[str, Any], agenda: str) -> str:
    """Build prompt for a restaurant."""
    # Context is just str(dict) of the restaurant data
    context = str(restaurant)
    return PROMPT_TEMPLATE.format(context=context, agenda=agenda)


def build_messages(
    restaurant: Dict[str, Any],
    agenda: str,
    system_prompt: str,
) -> List[Dict[str, str]]:
    """Build messages list for LLM call with system prompt."""
    context = str(restaurant)
    user_content = PROMPT_TEMPLATE.format(context=context, agenda=agenda)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def extract_json_response(response: str) -> Dict[str, Any]:
    """Extract structured JSON response from LLM output.

    Returns:
        Parsed JSON dict, or dict with error key if parsing fails.
    """
    try:
        # Handle markdown code blocks
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]
        else:
            json_str = response

        return json.loads(json_str.strip())
    except (json.JSONDecodeError, IndexError) as e:
        return {"verdict": None, "parse_error": str(e)}


def extract_verdict(response: str) -> Optional[str]:
    """Extract verdict from LLM response."""
    # Look for verdict patterns
    patterns = [
        r"(?:verdict|VERDICT)[:\s]*[\"']?([A-Za-z\s]+Risk)[\"']?",
        r"(Low Risk|High Risk|Critical Risk)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            verdict = match.group(1).strip()
            # Normalize
            if "low" in verdict.lower():
                return "Low Risk"
            elif "critical" in verdict.lower():
                return "Critical Risk"
            elif "high" in verdict.lower():
                return "High Risk"
    return None


def extract_risk_score(response: str) -> Optional[float]:
    """Extract FINAL_RISK_SCORE from LLM response."""
    # Look for score patterns
    patterns = [
        r"FINAL_RISK_SCORE[:\s]*([0-9]+\.?[0-9]*)",
        r"final_risk_score[:\s]*([0-9]+\.?[0-9]*)",
        r"[\"']?FINAL_RISK_SCORE[\"']?[:\s]*([0-9]+\.?[0-9]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def load_formula_seed(seed_path: str, policy_id: str) -> Dict[str, Any]:
    """Load Formula Seed from file or directory.

    Args:
        seed_path: Path to .json file or directory containing seeds
        policy_id: Policy identifier (e.g., "G1_allergy_V2")

    Returns:
        Formula Seed dict

    Raises:
        FileNotFoundError: If seed file not found
    """
    path = Path(seed_path)

    if path.is_file():
        # Direct file path
        with open(path) as f:
            return json.load(f)
    elif path.is_dir():
        # Directory - look for {policy_id}.json
        seed_file = path / f"{policy_id}.json"
        if not seed_file.exists():
            raise FileNotFoundError(
                f"Formula Seed not found: {seed_file}\n"
                f"Expected {policy_id}.json in {path}"
            )
        with open(seed_file) as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Seed path not found: {seed_path}")


def _get_batch_field(batch: Any, key: str) -> Optional[Any]:
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


def build_result_from_response(
    restaurant: Dict[str, Any],
    agenda: str,
    response: str,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a result dict from a response string."""
    business = restaurant.get("business", {})
    name = business.get("name", "Unknown")
    business_id = business.get("business_id", "")

    if system_prompt:
        messages = build_messages(restaurant, agenda, system_prompt)
        prompt_chars = sum(len(m["content"]) for m in messages)
    else:
        prompt = build_prompt(restaurant, agenda)
        prompt_chars = len(prompt)

    parsed = extract_json_response(response)
    if "parse_error" not in parsed:
        verdict = parsed.get("verdict")
        if verdict:
            v_lower = verdict.lower()
            if "low" in v_lower:
                verdict = "Low Risk"
            elif "critical" in v_lower:
                verdict = "Critical Risk"
            elif "high" in v_lower:
                verdict = "High Risk"

        return {
            "business_id": business_id,
            "name": name,
            "response": response,
            "parsed": parsed,
            "verdict": verdict,
            "risk_score": None,
            "prompt_chars": prompt_chars,
        }

    verdict = extract_verdict(response)
    risk_score = extract_risk_score(response)
    return {
        "business_id": business_id,
        "name": name,
        "response": response,
        "verdict": verdict,
        "risk_score": risk_score,
        "prompt_chars": prompt_chars,
    }


async def eval_restaurant(
    restaurant: Dict[str, Any],
    agenda: str,
    llm: LLMService,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a single restaurant.

    Args:
        restaurant: Restaurant data dict
        agenda: The agenda/prompt text
        llm: LLM service instance
        system_prompt: Optional system prompt for structured output
    """
    business = restaurant.get("business", {})
    name = business.get("name", "Unknown")
    business_id = business.get("business_id", "")

    # Build messages with or without system prompt
    if system_prompt:
        messages = build_messages(restaurant, agenda, system_prompt)
        prompt_chars = sum(len(m["content"]) for m in messages)
    else:
        prompt = build_prompt(restaurant, agenda)
        messages = [{"role": "user", "content": prompt}]
        prompt_chars = len(prompt)

    try:
        response, usage = await llm.call_async_with_usage(
            messages, context={"sample_id": business_id}
        )

        # Extract usage metrics
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = usage.get("cost_usd", 0.0)
        latency_ms = usage.get("latency_ms")

        # Try JSON parsing first (for structured output)
        parsed = extract_json_response(response)
        if "parse_error" not in parsed:
            # Successfully parsed JSON
            verdict = parsed.get("verdict")
            # Normalize verdict
            if verdict:
                v_lower = verdict.lower()
                if "low" in v_lower:
                    verdict = "Low Risk"
                elif "critical" in v_lower:
                    verdict = "Critical Risk"
                elif "high" in v_lower:
                    verdict = "High Risk"

            return {
                "business_id": business_id,
                "name": name,
                "response": response,
                "parsed": parsed,
                "verdict": verdict,
                "risk_score": None,  # JSON format doesn't use numeric score
                "prompt_chars": prompt_chars,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
                "latency_ms": latency_ms,
                "llm_calls": 1,
            }

        # Fallback to regex extraction (legacy format)
        verdict = extract_verdict(response)
        risk_score = extract_risk_score(response)

        return {
            "business_id": business_id,
            "name": name,
            "response": response,
            "verdict": verdict,
            "risk_score": risk_score,
            "prompt_chars": prompt_chars,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
            "llm_calls": 1,
        }
    except Exception as e:
        return {
            "business_id": business_id,
            "name": name,
            "error": str(e),
        }


# Type alias for progress callback
ProgressCallback = Callable[[int, str, float, str], None]


async def run_experiment(
    task_id: Optional[str] = None,
    policy_id: Optional[str] = None,
    domain: str = "yelp",
    k: int = 50,
    n: int = 100,
    skip: int = 0,
    model: str = "gpt-5-nano",
    verbose: bool = True,
    dev: bool = False,
    benchmark: bool = False,
    force: bool = False,
    method: str = "direct",
    token_limit: Optional[int] = None,
    mode: Optional[str] = None,
    batch_id: Optional[str] = None,
    top_k: int = 20,
    batch_size: int = 10,
    sample_ids: Optional[List[str]] = None,
    phase: Optional[str] = None,
    seed_path: Optional[str] = None,
    run_number: Optional[int] = None,
    suppress_output: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Run experiment evaluation.

    Args:
        task_id: Legacy task ID (e.g., "G1a") - loads from data/answers/
        policy_id: Policy ID (e.g., "G1_allergy_V2") - loads from data/query/
        domain: Domain (default: yelp)
        k: Reviews per restaurant
        n: Number of restaurants (0=all)
        skip: Skip first N restaurants
        model: LLM model to use
        verbose: Print detailed output
        dev: If True, save to results/dev/{YYYYMMDD}_{run_name}/
        benchmark: If True, use quota-controlled benchmark mode
        force: If True, create new run even if quota met
        method: Method to use - "direct" (default), "rlm", "rag", or "amos"
        token_limit: Token budget for RLM (converts to iterations via ~3000 tokens/iter)
        mode: Explicit mode override ("ondemand" or "batch").
              In benchmark mode, auto-selected based on quota state if not specified.
        batch_size: AMOS reviews per LLM call
        sample_ids: If provided, only run on these specific business IDs (filters dataset)
        phase: AMOS phase control: '1' (generate seed only), '2' (use seed_path), '1,2' or None (both)
        seed_path: Path to Formula Seed file or directory for --phase 2
        run_number: Explicit run number (1-5) for benchmark mode. If specified:
            - run 1 = ondemand mode (captures latency)
            - run 2-5 = batch mode (cost efficient)
            If not specified, auto-detects next available run number.

    Either task_id or policy_id must be provided.
    """
    if not task_id and not policy_id:
        raise ValueError("Either task_id or policy_id must be provided")

    # Validate phase/seed compatibility
    if phase == "2" and not seed_path:
        raise ValueError("--phase 2 requires --seed to specify pre-generated Formula Seed")
    if seed_path and phase != "2":
        raise ValueError("--seed is only valid with --phase 2")
    if phase and phase not in ("1", "2", "1,2"):
        raise ValueError("--phase must be '1', '2', or '1,2'")
    if phase and method != "amos":
        raise ValueError("--phase is only supported for --method amos")

    # Determine run mode and load agenda
    system_prompt = None
    if policy_id:
        # Policy mode: load from data/query/ with system prompt
        agenda = load_policy_prompt(policy_id, domain, k=k)
        system_prompt = load_system_prompt()
        run_id = policy_id.replace("/", "_")
        # Use policy ID directly for ground truth (policy-based GT files)
        gt_task_id = run_id
    else:
        # Legacy task mode: load from data/answers/ (no system prompt)
        task = load_task(task_id, domain)
        agenda = task.parsed_prompt.full_text
        run_id = task_id
        gt_task_id = task_id

    # Load dataset
    restaurants = load_dataset(domain, k)

    # Filter by sample_ids if provided (overrides skip/n)
    if sample_ids:
        sample_id_set = set(sample_ids)
        restaurants = [
            r for r in restaurants
            if r.get("business", {}).get("business_id") in sample_id_set
        ]
        # Warn if some sample_ids not found
        found_ids = {r["business"]["business_id"] for r in restaurants}
        missing_ids = sample_id_set - found_ids
        if missing_ids:
            output.warn(f"Sample IDs not found in dataset: {missing_ids}")
    elif n > 0:
        restaurants = restaurants[skip : skip + n]
    else:
        restaurants = restaurants[skip:]

    # Get results manager
    results_manager = get_results_manager()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine run mode (ondemand vs batch)
    effective_mode = mode  # Explicit override takes precedence

    # Determine output directory based on run type
    if dev:
        # Dev mode: results/dev/{YYYYMMDD}_{run_name}/
        output_dir = results_manager.create_dev_run_dir(run_id, timestamp)

        # Default to ondemand for dev runs
        if effective_mode is None:
            effective_mode = "ondemand"

    elif benchmark:
        # Benchmark mode: results/{method}/{policy_id}_K{k}/run_N/
        # Include K in the benchmark identifier for separate tracking
        benchmark_id = f"{run_id}_K{k}"

        if run_number is not None:
            # Explicit run number specified via --run
            # Validate run number range
            if run_number < 1 or run_number > results_manager.DEFAULT_QUOTA:
                raise ValueError(f"--run must be between 1 and {results_manager.DEFAULT_QUOTA}")

            # Determine mode based on run number: 1 = ondemand, 2+ = batch
            if effective_mode is None:
                effective_mode = "ondemand" if run_number == 1 else "batch"

            # Create run directory with explicit number
            output_dir = results_manager.get_benchmark_dir(method, benchmark_id) / f"run_{run_number}"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "item_logs").mkdir(exist_ok=True)
        else:
            # Auto-select mode based on quota state
            if effective_mode is None:
                effective_mode = results_manager.get_next_mode(method, benchmark_id)

                if effective_mode is None and not force:
                    # Quota met - print aggregate and exit
                    output.info(f"Benchmark quota met for {method}/{benchmark_id}")
                    results_manager.print_aggregate_summary(method, benchmark_id, output.print)
                    return {"quota_met": True, "method": method, "policy_id": run_id, "k": k}

            # Create run directory
            output_dir = results_manager.get_or_create_run_dir(method, benchmark_id, force=force)
            if output_dir is None and not force:
                # Quota met
                output.info(f"Benchmark quota met for {method}/{benchmark_id}")
                results_manager.print_aggregate_summary(method, benchmark_id, output.print)
                return {"quota_met": True, "method": method, "policy_id": run_id, "k": k}

    else:
        # Legacy mode: results/legacy/{run_id}/
        output_dir = Path(f"results/legacy/{run_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "item_logs").mkdir(exist_ok=True)

        # Default to ondemand for legacy runs
        if effective_mode is None:
            effective_mode = "ondemand"

    # Configure output manager
    output.configure(quiet=not verbose, mode=effective_mode, suppress_all=suppress_output)

    # Setup loggers
    # Debug logger writes to debug.jsonl (consolidated mode)
    debug_logger = DebugLogger(output_dir, consolidated=True)
    set_debug_logger(debug_logger)

    # Item logger writes to item_logs/{sample_id}.json (streaming, per-sample)
    item_logger = ItemLogger(output_dir)
    set_item_logger(item_logger)

    method_name_map = {
        "direct": "Direct LLM",
        "cot": "CoT (Chain-of-Thought)",
        "react": "ReACT (Reasoning + Acting)",
        "rlm": "RLM (Recursive LLM)",
        "rag": "RAG (Retrieval Augmented Generation)",
        "amos": "AMOS (Agenda-Driven Mining)",
    }
    method_name = method_name_map.get(method, "Direct LLM")
    output.header(f"{method_name} Baseline", run_id)

    # Build token limit display string
    token_display = str(token_limit)
    if method == "rlm":
        token_display += f" (~{token_limit // 3000} iterations)"

    output.print_config({
        "Method": method,
        "Token limit": token_display,
        "Restaurants": len(restaurants),
        "Model": model,
        "K": k,
        "Output": str(output_dir),
    })
    if gt_task_id:
        output.status(f"Ground truth: {gt_task_id}_K{k}_groundtruth.json")

    # Log query template for item logger
    if item_logger := get_item_logger():
        example_context = str(restaurants[0]) if restaurants else "No context"
        item_logger.log_query_template(
            policy_id=run_id,
            query_text=agenda,
            example_context=example_context[:500] + "..." if len(example_context) > 500 else example_context,
            system_prompt=system_prompt,
        )

    if effective_mode == "batch":
        if method in ["rlm", "rag", "react"]:
            raise ValueError("batch mode is only supported for single-call methods (direct, cot)")

        batch_client = BatchClient()
        benchmark_id = f"{run_id}_K{k}"

        # Check for existing batch job via manifest (unless explicit batch_id provided)
        if not batch_id:
            manifest_status, manifest = _check_batch_status(method, benchmark_id)

            if manifest_status == "processing":
                manifest_batch_id = manifest["batch_id"]
                batch = batch_client.get_batch(manifest_batch_id)
                status = _get_batch_field(batch, "status")
                output.batch_status(manifest_batch_id, status)
                output.info("Re-run this command later to check status and download results.")
                return {"status": "processing", "batch_id": manifest_batch_id}

            elif manifest_status == "failed":
                output.warn(f"Previous batch failed. Deleting manifest and resubmitting...")
                _delete_manifest(method, benchmark_id)
                # Fall through to new submission

            elif manifest_status == "ready":
                # Batch complete - use manifest batch_id for result processing
                batch_id = manifest["batch_id"]
                output.info(f"Found completed batch: {batch_id[:20]}...")

        # Process existing batch results (from --batch-id or manifest)
        if batch_id:
            batch = batch_client.get_batch(batch_id)
            status = _get_batch_field(batch, "status")
            if status not in {"completed", "failed", "expired", "cancelled"}:
                output.batch_status(batch_id, status)
                output.info("Re-run this command later to check status.")
                return {"status": "processing", "batch_id": batch_id}

            output_file_id = _get_batch_field(batch, "output_file_id")
            error_file_id = _get_batch_field(batch, "error_file_id")
            if error_file_id:
                output.warn(f"Batch has error file: {error_file_id}")

            if not output_file_id:
                output.warn(f"No output file available for batch {batch_id}")
                _delete_manifest(method, benchmark_id)
                return {}

            output_bytes = batch_client.download_file(output_file_id)
            restaurant_map = {
                r.get("business", {}).get("business_id", ""): r
                for r in restaurants
            }

            results: List[Dict[str, Any]] = []
            for line in output_bytes.splitlines():
                if not line:
                    continue
                item = json.loads(line)
                custom_id = item.get("custom_id", "")
                try:
                    run_tag, business_id = _parse_sample_custom_id(custom_id)
                except ValueError:
                    continue
                if run_tag != run_id:
                    continue

                response_text = _get_batch_response_text(item)
                if response_text is None:
                    continue

                restaurant = restaurant_map.get(business_id)
                if not restaurant:
                    continue

                result = build_result_from_response(
                    restaurant=restaurant,
                    agenda=agenda,
                    response=response_text,
                    system_prompt=system_prompt,
                )
                # Add token usage from batch response
                usage = _get_batch_usage(item, model)
                result.update(usage)
                results.append(result)

            # Delete manifest after successful processing
            _delete_manifest(method, benchmark_id)
        else:
            # Submit new batch
            request_items = []
            for restaurant in restaurants:
                business = restaurant.get("business", {})
                business_id = business.get("business_id", "")

                # Build messages based on method
                if method == "cot":
                    # CoT uses step-by-step reasoning prompt
                    context = str(restaurant)
                    user_content = f"""Query: {agenda}

Context:
{context}

Let's think through this step-by-step:"""
                    messages = [
                        {"role": "system", "content": COT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ]
                elif system_prompt:
                    messages = build_messages(restaurant, agenda, system_prompt)
                else:
                    prompt = build_prompt(restaurant, agenda)
                    messages = [{"role": "user", "content": prompt}]

                custom_id = _build_sample_custom_id(run_id, business_id)
                request_items.append(
                    build_chat_batch_item(
                        custom_id=custom_id,
                        model=model,
                        messages=messages,
                    )
                )

            input_file_id = batch_client.upload_batch_file(request_items)
            batch_id = batch_client.submit_batch(input_file_id)

            # Save manifest for status tracking
            manifest = {
                "batch_id": batch_id,
                "method": method,
                "policy_id": run_id,
                "k": k,
                "n": len(restaurants),
                "model": model,
                "created_at": datetime.now().isoformat(),
            }
            _save_manifest(method, benchmark_id, manifest)

            output.batch_submitted(batch_id)
            output.info("Re-run this command to check status and download results.")
            return {"batch_id": batch_id, "manifest_saved": True}
    else:
        # Configure LLM
        llm = LLMService()
        llm.configure(model=model)

        # Run evaluations (with concurrency limit)
        if method == "rlm":
            # RLM method - uses recursive exploration
            tasks = [
                eval_restaurant_rlm(r, agenda, system_prompt, model=model, token_limit=token_limit)
                for r in restaurants
            ]
            # RLM has lower concurrency due to multiple internal LLM calls
            results = await gather_with_concurrency(4, tasks)
        elif method == "rag":
            # RAG method - retrieval augmented generation
            from addm.methods import build_method_registry
            from addm.data.types import Sample

            registry = build_method_registry()
            rag_class = registry.get("rag")
            rag_method = rag_class(top_k=top_k, system_prompt=system_prompt)

            # Convert restaurants to Sample objects
            samples = [
                Sample(
                    sample_id=r["business"]["business_id"],
                    query=agenda,
                    context=json.dumps(r),
                    metadata={"restaurant_name": r["business"]["name"]},
                )
                for r in restaurants
            ]

            # Run via method interface
            tasks = [rag_method.run_sample(s, llm) for s in samples]
            results_raw = await gather_with_concurrency(32, tasks)

            # Convert RAG results to expected format (build_result_from_response format)
            results = []
            for raw_result in results_raw:
                sample_id = raw_result["sample_id"]
                restaurant = next((r for r in restaurants if r["business"]["business_id"] == sample_id), None)
                if not restaurant:
                    continue

                response = raw_result["output"]
                parsed = extract_json_response(response)

                if "parse_error" not in parsed:
                    verdict = parsed.get("verdict")
                    if verdict:
                        v_lower = verdict.lower()
                        if "low" in v_lower:
                            verdict = "Low Risk"
                        elif "critical" in v_lower:
                            verdict = "Critical Risk"
                        elif "high" in v_lower:
                            verdict = "High Risk"

                    results.append({
                        "business_id": sample_id,
                        "name": restaurant["business"]["name"],
                        "response": response,
                        "parsed": parsed,
                        "verdict": verdict,
                        "risk_score": None,
                        "prompt_chars": raw_result.get("prompt_tokens", 0) * 4,  # Estimate
                        # RAG-specific metrics
                        "embedding_tokens": raw_result.get("embedding_tokens", 0),
                        "embedding_cost_usd": raw_result.get("embedding_cost_usd", 0.0),
                        "reviews_retrieved": raw_result.get("reviews_retrieved", 0),
                        "reviews_total": raw_result.get("reviews_total", 0),
                        "top_review_indices": raw_result.get("top_review_indices", []),
                        "cache_hit_embeddings": raw_result.get("cache_hit_embeddings", False),
                        "cache_hit_retrieval": raw_result.get("cache_hit_retrieval", False),
                    })
                else:
                    verdict = extract_verdict(response)
                    risk_score = extract_risk_score(response)
                    results.append({
                        "business_id": sample_id,
                        "name": restaurant["business"]["name"],
                        "response": response,
                        "verdict": verdict,
                        "risk_score": risk_score,
                        "prompt_chars": raw_result.get("prompt_tokens", 0) * 4,
                        # RAG-specific metrics
                        "embedding_tokens": raw_result.get("embedding_tokens", 0),
                        "embedding_cost_usd": raw_result.get("embedding_cost_usd", 0.0),
                        "reviews_retrieved": raw_result.get("reviews_retrieved", 0),
                        "reviews_total": raw_result.get("reviews_total", 0),
                        "top_review_indices": raw_result.get("top_review_indices", []),
                        "cache_hit_embeddings": raw_result.get("cache_hit_embeddings", False),
                        "cache_hit_retrieval": raw_result.get("cache_hit_retrieval", False),
                    })
        elif method == "amos":
            # AMOS method - Agenda-Driven Mining with Observable Steps
            from addm.methods import build_method_registry
            from addm.data.types import Sample

            registry = build_method_registry()
            amos_class = registry.get("amos")
            amos_method = amos_class(
                policy_id=run_id,
                max_concurrent=256,
                batch_size=batch_size,
                system_prompt=system_prompt,
                progress_callback=progress_callback,
            )

            # Convert restaurants to Sample objects
            samples = [
                Sample(
                    sample_id=r["business"]["business_id"],
                    query=agenda,
                    context=json.dumps(r),
                    metadata={"restaurant_name": r["business"]["name"]},
                )
                for r in restaurants
            ]

            # Handle phase control
            if phase == "2":
                # Phase 2 only: Load pre-generated seed
                seed = load_formula_seed(seed_path, run_id)
                amos_method.set_formula_seed(seed)
                output.info(f"Loaded Formula Seed from: {seed_path}")
            else:
                # Phase 1 or both: Generate seed
                seed = await amos_method._get_formula_seed(agenda, llm)

            # Handle phase=1: exit after Phase 1 without running samples
            if phase == "1":
                # Save Formula Seed to run directory with policy-level naming
                seed_output_path = output_dir / f"{run_id}.json"
                with open(seed_output_path, "w") as f:
                    json.dump(seed, f, indent=2)
                # Also save as formula_seed.json for backwards compatibility
                amos_method.save_formula_seed_to_run_dir(output_dir)

                # Print summary
                filter_spec = seed.get("filter", {})
                extract_spec = seed.get("extract", {})
                compute_spec = seed.get("compute", [])

                output.success(f"Phase 1 complete: {seed.get('task_name', run_id)}")
                output.print(f"  Keywords: {len(filter_spec.get('keywords', []))}")
                output.print(f"  Fields: {len(extract_spec.get('fields', []))}")
                output.print(f"  Compute ops: {len(compute_spec)}")
                output.print(f"  Saved to: {seed_output_path}")

                return {
                    "phase": "1",
                    "policy_id": run_id,
                    "output_dir": str(output_dir),
                    "seed": seed,
                    "seed_summary": {
                        "task_name": seed.get("task_name"),
                        "keywords": len(filter_spec.get("keywords", [])),
                        "fields": len(extract_spec.get("fields", [])),
                        "compute_ops": len(compute_spec),
                    },
                }

            # Run all samples in parallel (Phase 2 only)
            async def process_amos_sample(sample: Sample) -> Dict[str, Any]:
                raw_result = await amos_method.run_sample(sample, llm)
                restaurant = next((r for r in restaurants if r["business"]["business_id"] == raw_result["sample_id"]), None)
                if not restaurant:
                    return None

                # Extract verdict directly from AMOS output (no normalization)
                # Each policy defines its own verdict labels (e.g., "Low Risk", "Recommended", "Good Value")
                # Hardcoded normalization was breaking non-risk tasks (G2-G6)
                verdict = raw_result.get("verdict")

                # AMOS now returns standard output format in `parsed` field
                # This matches the output_schema.txt structure
                parsed = raw_result.get("parsed", {})

                return {
                    "business_id": raw_result["sample_id"],
                    "name": restaurant["business"]["name"],
                    "response": raw_result.get("output", ""),
                    "parsed": parsed,  # Standard output format: {verdict, evidences, justification}
                    "verdict": verdict,
                    "risk_score": raw_result.get("risk_score"),
                    "prompt_chars": raw_result.get("prompt_tokens", 0) * 4,
                    # Token and latency metrics
                    "prompt_tokens": raw_result.get("prompt_tokens", 0),
                    "completion_tokens": raw_result.get("completion_tokens", 0),
                    "total_tokens": raw_result.get("total_tokens", 0),
                    "cost_usd": raw_result.get("cost_usd", 0.0),
                    "latency_ms": raw_result.get("latency_ms", 0.0),
                    "llm_calls": raw_result.get("llm_calls", 0),
                    # AMOS-specific metrics
                    "stats": raw_result.get("stats", {}),
                    "extractions_count": raw_result.get("extractions_count", 0),
                    "phase1_cached": raw_result.get("phase1_cached", False),
                }

            tasks = [process_amos_sample(s) for s in samples]
            raw_results = await gather_with_concurrency(100, tasks)  # 100 concurrent restaurants
            results = [r for r in raw_results if r is not None]

            # Save Formula Seed to run directory for artifact tracking
            amos_method.save_formula_seed_to_run_dir(output_dir)
        elif method == "cot":
            # CoT method - chain-of-thought reasoning
            from addm.methods import build_method_registry
            from addm.data.types import Sample

            registry = build_method_registry()
            cot_class = registry.get("cot")
            cot_method = cot_class(system_prompt=system_prompt)

            # Convert restaurants to Sample objects
            samples = [
                Sample(
                    sample_id=r["business"]["business_id"],
                    query=agenda,
                    context=str(r),  # CoT uses string context
                    metadata={"restaurant_name": r["business"]["name"]},
                )
                for r in restaurants
            ]

            # Run via method interface
            tasks = [cot_method.run_sample(s, llm) for s in samples]
            results_raw = await gather_with_concurrency(32, tasks)

            # Convert CoT results to expected format
            results = []
            for raw_result in results_raw:
                sample_id = raw_result["sample_id"]
                restaurant = next((r for r in restaurants if r["business"]["business_id"] == sample_id), None)
                if not restaurant:
                    continue

                response = raw_result["output"]
                parsed = extract_json_response(response)

                if "parse_error" not in parsed:
                    verdict = parsed.get("verdict")
                    if verdict:
                        v_lower = verdict.lower()
                        if "low" in v_lower:
                            verdict = "Low Risk"
                        elif "critical" in v_lower:
                            verdict = "Critical Risk"
                        elif "high" in v_lower:
                            verdict = "High Risk"

                    results.append({
                        "business_id": sample_id,
                        "name": restaurant["business"]["name"],
                        "response": response,
                        "parsed": parsed,
                        "verdict": verdict,
                        "risk_score": None,
                        "prompt_chars": raw_result.get("prompt_tokens", 0) * 4,
                        "prompt_tokens": raw_result.get("prompt_tokens", 0),
                        "completion_tokens": raw_result.get("completion_tokens", 0),
                        "cost_usd": raw_result.get("cost_usd", 0.0),
                        "latency_ms": raw_result.get("latency_ms", 0.0),
                        "llm_calls": raw_result.get("llm_calls", 1),
                    })
                else:
                    verdict = extract_verdict(response)
                    risk_score = extract_risk_score(response)
                    results.append({
                        "business_id": sample_id,
                        "name": restaurant["business"]["name"],
                        "response": response,
                        "verdict": verdict,
                        "risk_score": risk_score,
                        "prompt_chars": raw_result.get("prompt_tokens", 0) * 4,
                        "prompt_tokens": raw_result.get("prompt_tokens", 0),
                        "completion_tokens": raw_result.get("completion_tokens", 0),
                        "cost_usd": raw_result.get("cost_usd", 0.0),
                        "latency_ms": raw_result.get("latency_ms", 0.0),
                        "llm_calls": raw_result.get("llm_calls", 1),
                    })
        elif method == "react":
            # ReACT method - reasoning and acting with tools
            from addm.methods import build_method_registry
            from addm.data.types import Sample

            registry = build_method_registry()
            react_class = registry.get("react")
            react_method = react_class()

            # Convert restaurants to Sample objects
            samples = [
                Sample(
                    sample_id=r["business"]["business_id"],
                    query=agenda,
                    context=json.dumps(r),  # ReACT needs JSON for tool access
                    metadata={"restaurant_name": r["business"]["name"]},
                )
                for r in restaurants
            ]

            # Run via method interface (lower concurrency due to multi-step)
            tasks = [react_method.run_sample(s, llm) for s in samples]
            results_raw = await gather_with_concurrency(8, tasks)

            # Convert ReACT results to expected format
            results = []
            for raw_result in results_raw:
                sample_id = raw_result["sample_id"]
                restaurant = next((r for r in restaurants if r["business"]["business_id"] == sample_id), None)
                if not restaurant:
                    continue

                response = raw_result["output"]
                parsed = extract_json_response(response)

                if "parse_error" not in parsed:
                    verdict = parsed.get("verdict")
                    if verdict:
                        v_lower = verdict.lower()
                        if "low" in v_lower:
                            verdict = "Low Risk"
                        elif "critical" in v_lower:
                            verdict = "Critical Risk"
                        elif "high" in v_lower:
                            verdict = "High Risk"

                    results.append({
                        "business_id": sample_id,
                        "name": restaurant["business"]["name"],
                        "response": response,
                        "parsed": parsed,
                        "verdict": verdict,
                        "risk_score": None,
                        "prompt_chars": raw_result.get("prompt_tokens", 0) * 4,
                        "prompt_tokens": raw_result.get("prompt_tokens", 0),
                        "completion_tokens": raw_result.get("completion_tokens", 0),
                        "cost_usd": raw_result.get("cost_usd", 0.0),
                        "latency_ms": raw_result.get("latency_ms", 0.0),
                        "llm_calls": raw_result.get("llm_calls", 1),
                        # ReACT-specific metrics
                        "steps_taken": raw_result.get("steps_taken", 0),
                        "max_steps": raw_result.get("max_steps", 5),
                    })
                else:
                    verdict = extract_verdict(response)
                    risk_score = extract_risk_score(response)
                    results.append({
                        "business_id": sample_id,
                        "name": restaurant["business"]["name"],
                        "response": response,
                        "verdict": verdict,
                        "risk_score": risk_score,
                        "prompt_chars": raw_result.get("prompt_tokens", 0) * 4,
                        "prompt_tokens": raw_result.get("prompt_tokens", 0),
                        "completion_tokens": raw_result.get("completion_tokens", 0),
                        "cost_usd": raw_result.get("cost_usd", 0.0),
                        "latency_ms": raw_result.get("latency_ms", 0.0),
                        "llm_calls": raw_result.get("llm_calls", 1),
                        # ReACT-specific metrics
                        "steps_taken": raw_result.get("steps_taken", 0),
                        "max_steps": raw_result.get("max_steps", 5),
                    })
        else:
            # Direct method - single LLM call with full context
            tasks = [eval_restaurant(r, agenda, llm, system_prompt) for r in restaurants]
            results = await gather_with_concurrency(32, tasks)

    # Load ground truth for scoring
    gt_verdicts = {}
    if gt_task_id:
        gt_verdicts = load_ground_truth(gt_task_id, domain, k)
    if not gt_verdicts:
        output.warn(f"No ground truth found for {gt_task_id or run_id}")

    # Score and print results
    correct = 0
    total = 0

    for result in results:
        biz_id = result.get("business_id", "")
        gt_verdict = gt_verdicts.get(biz_id)

        if "error" in result:
            output.error(f"{result.get('name', biz_id)}: {result['error']}")
            # Log error to item_logs
            if item_logger := get_item_logger():
                item_logger.log_item_error(
                    sample_id=biz_id,
                    error=result["error"],
                    ground_truth=gt_verdict,
                )
            continue

        pred_verdict = result["verdict"]
        # Normalize verdicts for comparison (handles quotes, case, whitespace)
        pred_normalized = normalize_verdict(pred_verdict, run_id)
        gt_normalized = normalize_verdict(gt_verdict, run_id)
        is_correct = pred_normalized == gt_normalized

        if gt_verdict:
            total += 1
            if is_correct:
                correct += 1

        result["gt_verdict"] = gt_verdict
        result["correct"] = is_correct

        # Build metrics dict for item_logger
        item_metrics = {
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
            "cost_usd": result.get("cost_usd", 0.0),
            "latency_ms": result.get("latency_ms"),
            "llm_calls": result.get("llm_calls", 1),
        }

        # Log to item_logs/{sample_id}.json (streaming)
        if item_logger := get_item_logger():
            # Collect method-specific extra fields
            extra_fields = {}
            for key in ["stats", "extractions_count", "phase1_cached",
                        "reviews_retrieved", "reviews_total", "top_review_indices",
                        "cache_hit_embeddings", "cache_hit_retrieval",
                        "steps_taken", "max_steps"]:
                if key in result:
                    extra_fields[key] = result[key]

            item_logger.log_item(
                sample_id=biz_id,
                verdict=pred_verdict,
                output=result.get("response", ""),
                parsed=result.get("parsed"),
                ground_truth=gt_verdict,
                correct=is_correct,
                metrics=item_metrics,
                **extra_fields,
            )

        # Display result (mode-aware via output.configure())
        if not output._quiet:
            output.rule()
            output.print_result(
                name=result["name"],
                predicted=pred_verdict,
                ground_truth=gt_verdict,
                score=result.get("risk_score"),
                correct=is_correct,
            )
            if verbose:
                # Print first 1000 chars of response
                resp = result["response"]
                output.print(resp[:1000] + ("..." if len(resp) > 1000 else ""))

    # Print accuracy
    accuracy = correct / total if total > 0 else 0.0
    output.print_accuracy(correct, total)

    # Compute AUPRC using FINAL_RISK_SCORE as y_scores
    y_true = []
    y_scores = []
    for result in results:
        if "error" in result or not result.get("gt_verdict"):
            continue
        gt_ord = VERDICT_TO_ORDINAL.get(result["gt_verdict"])
        # Use risk_score (continuous), fallback to verdict ordinal * 5 if not available
        risk_score = result.get("risk_score")
        if risk_score is None:
            pred_ord = VERDICT_TO_ORDINAL.get(result.get("verdict"))
            risk_score = pred_ord * 5.0 if pred_ord is not None else None
        if gt_ord is not None and risk_score is not None:
            y_true.append(gt_ord)
            y_scores.append(risk_score)

    auprc_metrics = {}
    if len(y_true) >= 2:
        auprc_metrics = compute_ordinal_auprc(np.array(y_true), np.array(y_scores))

        # Display AUPRC metrics as table
        auprc_rows = []
        for key, val in auprc_metrics.items():
            if isinstance(val, float):
                auprc_rows.append([key, f"{val:.3f}"])
            else:
                auprc_rows.append([key, str(val)])
        if auprc_rows:
            output.print_table("AUPRC Metrics", ["Metric", "Value"], auprc_rows)

    # Compute intermediate metrics (if structured output available)
    intermediate_metrics = None
    has_structured = any(
        r.get("parsed") and isinstance(r.get("parsed"), dict)
        and "evidences" in r.get("parsed", {})
        for r in results
    )
    if has_structured and gt_task_id:
        from addm.eval.intermediate_metrics import (
            compute_intermediate_metrics,
            load_gt_with_incidents,
            build_reviews_data,
        )

        # Load full GT with incidents
        gt_data, _ = load_gt_with_incidents(gt_task_id, domain, k)

        if gt_data:
            # Build reviews data for snippet validation
            reviews_data = build_reviews_data(restaurants)

            intermediate_metrics = compute_intermediate_metrics(
                results, gt_data, reviews_data
            )

    # Compute evaluation metrics (new simplified system)
    eval_metrics = {}
    unified_metrics = {}  # Keep for backward compat in results.json
    if gt_verdicts:
        # Load full GT data for metrics
        from addm.eval.intermediate_metrics import load_gt_with_incidents, build_reviews_data

        gt_data_full, _ = load_gt_with_incidents(gt_task_id, domain, k)
        reviews_data = build_reviews_data(restaurants)

        # Compute new simplified metrics
        eval_metrics = compute_evaluation_metrics(
            results=results,
            gt_data=gt_data_full,
            gt_verdicts=gt_verdicts,
            method=method,
            reviews_data=reviews_data,
            policy_id=run_id,
        )

        # Also compute legacy unified metrics for backward compat
        unified_metrics = compute_unified_metrics(
            results=results,
            gt_data=gt_data_full,
            gt_verdicts=gt_verdicts,
            method=method,
            reviews_data=reviews_data,
            policy_id=run_id,
        )

        # Display new metrics format (7 metrics)
        output.rule()

        # Extract details for notes
        details = eval_metrics.get("details", {})
        evidence_details = details.get("incident_details", {})  # Key unchanged for backward compat
        judgement_details = details.get("judgement_details", {})
        score_details = details.get("score_details", {})
        consistency_details = details.get("consistency_details", {})

        total_samples = len(results)
        total_claimed = evidence_details.get("total_claimed", 0)
        total_matched = evidence_details.get("total_matched", 0)
        total_gt_evidence = evidence_details.get("total_gt_evidence", 0)
        total_snippets = evidence_details.get("total_snippets", 0)
        valid_snippets = evidence_details.get("valid_snippets", 0)
        judgement_total = judgement_details.get("total", 0)
        judgement_correct = judgement_details.get("correct", 0)
        score_total = score_details.get("total", 0)
        score_correct = score_details.get("correct", 0)
        consistency_total = consistency_details.get("total", 0)
        consistency_count = consistency_details.get("consistent", 0)

        # Format metrics with notes
        def fmt_pct(val):
            return f"{val:.1%}" if val is not None else "N/A"

        score_rows = [
            # Tier 1: Final Quality
            ["AUPRC", fmt_pct(eval_metrics.get("auprc")), "(ranking quality)"],
            # Tier 2: Evidence Quality
            ["Evidence Precision", fmt_pct(eval_metrics.get("evidence_precision")),
             f"({total_matched}/{total_claimed} claimed exist in GT)" if total_claimed > 0 else "(no claims)"],
            ["Evidence Recall", fmt_pct(eval_metrics.get("evidence_recall")),
             f"({total_matched}/{total_gt_evidence} GT evidence found)" if total_gt_evidence > 0 else "(no GT evidence)"],
            ["Snippet Validity", fmt_pct(eval_metrics.get("snippet_validity")),
             f"({valid_snippets}/{total_snippets} quotes match source)" if total_snippets > 0 else "(no snippets)"],
            # Tier 3: Reasoning Quality
            ["Judgement Accuracy", fmt_pct(eval_metrics.get("judgement_accuracy")),
             f"({judgement_correct}/{judgement_total} field correctness)" if judgement_total > 0 else "(no matches)"],
            ["Score Accuracy", fmt_pct(eval_metrics.get("score_accuracy")),
             f"({score_correct}/{score_total} scores match GT)" if score_total > 0 else
             (score_details.get("skipped", "(V2/V3 only)") if isinstance(score_details, dict) else "(V2/V3 only)")],
            ["Verdict Consistency", fmt_pct(eval_metrics.get("verdict_consistency")),
             f"({consistency_count}/{consistency_total} evidence+rule→verdict)" if consistency_total > 0 else "(no evidence)"],
        ]
        output.print_table("EVALUATION METRICS (7 total)", ["Metric", "Score", "Notes"], score_rows)

    # Compute aggregated usage from all results
    aggregated_usage = {
        "total_prompt_tokens": sum(r.get("prompt_tokens", 0) for r in results if "error" not in r),
        "total_completion_tokens": sum(r.get("completion_tokens", 0) for r in results if "error" not in r),
        "total_tokens": sum(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in results if "error" not in r),
        "total_cost_usd": sum(r.get("cost_usd", 0.0) for r in results if "error" not in r),
        "total_latency_ms": sum(r.get("latency_ms", 0.0) for r in results if "error" not in r and r.get("latency_ms")),
        "total_llm_calls": sum(r.get("llm_calls", 1) for r in results if "error" not in r),
    }
    # Add embedding costs for RAG/hybrid methods
    embedding_tokens = sum(r.get("embedding_tokens", 0) for r in results if "error" not in r)
    embedding_cost = sum(r.get("embedding_cost_usd", 0.0) for r in results if "error" not in r)
    if embedding_tokens > 0:
        aggregated_usage["total_embedding_tokens"] = embedding_tokens
        aggregated_usage["total_embedding_cost_usd"] = embedding_cost

    # Display usage summary
    output.rule()
    total_tokens = aggregated_usage["total_tokens"]
    total_cost = aggregated_usage["total_cost_usd"]
    total_latency_ms = aggregated_usage["total_latency_ms"]
    total_llm_calls = aggregated_usage["total_llm_calls"]

    # Format latency as seconds
    latency_str = f"{total_latency_ms / 1000:.1f}s" if total_latency_ms else "N/A (batch)"

    usage_rows = [
        ["Tokens", f"{total_tokens:,}", f"({aggregated_usage['total_prompt_tokens']:,} prompt + {aggregated_usage['total_completion_tokens']:,} completion)"],
        ["Cost", f"${total_cost:.4f}", ""],
        ["Runtime", latency_str, f"({total_llm_calls} LLM calls)"],
    ]
    if embedding_tokens > 0:
        usage_rows.insert(1, ["Embedding Tokens", f"{embedding_tokens:,}", f"(${embedding_cost:.4f})"])
    output.print_table("USAGE SUMMARY", ["Metric", "Value", "Details"], usage_rows)

    # Save results (output_dir determined earlier)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"

    # Extract run_N from output_dir for benchmark runs
    run_dir_name = output_dir.name if benchmark else None

    run_output = {
        "run_id": run_dir_name or run_id,
        "task_id": task_id,
        "policy_id": policy_id,
        "gt_task_id": gt_task_id,
        "domain": domain,
        "method": method,
        "model": model,
        "mode": effective_mode,
        "has_latency": effective_mode == "ondemand",
        "k": k,
        "n": len(results),
        "timestamp": timestamp,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "auprc": auprc_metrics,
        "intermediate_metrics": intermediate_metrics,
        # New simplified metrics (recommended - 7 metrics)
        "evaluation_metrics": {
            # Tier 1: Final Quality
            "auprc": eval_metrics.get("auprc"),
            # Tier 2: Evidence Quality
            "evidence_precision": eval_metrics.get("evidence_precision"),
            "evidence_recall": eval_metrics.get("evidence_recall"),
            "snippet_validity": eval_metrics.get("snippet_validity"),
            # Tier 3: Reasoning Quality
            "judgement_accuracy": eval_metrics.get("judgement_accuracy"),
            "score_accuracy": eval_metrics.get("score_accuracy"),
            "verdict_consistency": eval_metrics.get("verdict_consistency"),
            # Sample counts
            "n_samples": eval_metrics.get("n_samples", 0),
            "n_structured": eval_metrics.get("n_structured", 0),
            "n_with_gt": eval_metrics.get("n_with_gt", 0),
        } if eval_metrics else None,
        "evaluation_metrics_full": eval_metrics if eval_metrics else None,
        # Legacy unified scores (deprecated, kept for backward compat)
        "unified_scores": {
            "auprc": unified_metrics.get("auprc", 0.0),
            "process_score": unified_metrics.get("process_score", 0.0),
            "consistency_score": unified_metrics.get("consistency_score", 0.0),
            "false_positive_rate": unified_metrics.get("false_positive_rate", 0.0),
        } if unified_metrics else None,
        "unified_metrics_full": unified_metrics if unified_metrics else None,
        # Aggregated usage (per-sample data in item_logs/)
        "aggregated_usage": aggregated_usage,
        # NOTE: Per-sample results removed - see item_logs/{sample_id}.json for details
    }

    with open(output_path, "w") as f:
        json.dump(run_output, f, indent=2)

    # Flush debug logger if enabled
    if debug_logger := get_debug_logger():
        debug_logger.flush()

    output.rule()
    output.success(f"Results saved to: {output_path}")

    return run_output


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run experiment evaluation")

    # Task or policy targeting (mutually exclusive, optional - omit to run all 72 policies)
    target_group = parser.add_mutually_exclusive_group(required=False)
    target_group.add_argument("--task", type=str, help="Legacy task ID (e.g., G1a)")
    target_group.add_argument(
        "--policy",
        type=str,
        help="Policy ID (e.g., G1_allergy_V2) or comma-separated list. Omit to run all 72 policies.",
    )
    target_group.add_argument(
        "--topic",
        type=str,
        help="Topic (e.g., G1_allergy) - runs all V0-V3 variants (4 policies)",
    )
    target_group.add_argument(
        "--group",
        type=str,
        help="Group (e.g., G1) - runs all topics × variants (12 policies)",
    )

    parser.add_argument("--domain", type=str, default="yelp", help="Domain")
    parser.add_argument("--k", type=int, default=50, help="Reviews per restaurant")
    parser.add_argument("-n", type=int, default=100, help="Number of restaurants (0=all)")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N")
    parser.add_argument("--model", type=str, default="gpt-5-nano", help="Model")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Dev mode: save to results/dev/{YYYYMMDD}_{run_name}/ (no quota)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force run even if benchmark quota is met",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="amos",
        choices=["direct", "cot", "react", "rlm", "rag", "amos"],
        help="Method: amos (default), direct, cot (chain-of-thought), react (reasoning+acting), rlm (recursive LLM), or rag (retrieval)",
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=50000,
        help="Token budget per restaurant. RLM: ~3000 tokens/iter, so 50000 = ~16 iterations",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="RAG: Number of reviews to retrieve (default: 20, ~10%% of K=200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="AMOS: Reviews per LLM call (default: 10). Higher=fewer calls, lower=more parallel.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default=None,
        help="AMOS phase control: '1' (generate seed only), '2' (use --seed), '1,2' or omit (both phases)",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=None,
        help="Path to Formula Seed file or directory. Required for --phase 2. "
             "If directory, looks for {policy_id}.json",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["ondemand", "batch"],
        help="LLM execution mode (auto-selected in benchmark mode if not specified)",
    )
    parser.add_argument("--batch-id", type=str, default=None, help="Batch ID for fetch-only runs")
    parser.add_argument(
        "--sample-ids",
        type=str,
        default=None,
        help="Comma-separated business IDs to filter dataset (e.g., 'id1,id2,id3')",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use pre-computed verdict samples (1 per verdict category) from data/answers/yelp/verdict_sample_ids.json",
    )
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help="Run specific run number only (1-5). Without this, runs up to quota.",
    )

    args = parser.parse_args()

    # Benchmark mode is default (unless --dev is specified)
    benchmark = not args.dev

    # Parse sample_ids from comma-separated string or load from --sample
    sample_ids = None
    all_sample_ids_data = None  # For multi-policy --sample mode
    if args.sample_ids:
        sample_ids = [s.strip() for s in args.sample_ids.split(",") if s.strip()]
    elif args.sample:
        # Load pre-computed verdict samples
        sample_ids_file = Path("data/answers/yelp/verdict_sample_ids.json")
        if not sample_ids_file.exists():
            output.error(f"Verdict samples file not found: {sample_ids_file}")
            output.info("Generate with: .venv/bin/python scripts/select_diverse_samples.py --all --k <K> --output ...")
            sys.exit(1)
        with open(sample_ids_file) as f:
            all_sample_ids_data = json.load(f)

        # For single policy, resolve sample_ids now; for multi-policy, resolve per-policy later
        if args.policy and "," not in args.policy:
            policy_key = args.policy.replace("/", "_")
            k_key = f"K{args.k}"
            if policy_key not in all_sample_ids_data:
                output.error(f"Policy {policy_key} not found in verdict samples")
                sys.exit(1)
            ids_str = all_sample_ids_data[policy_key].get(k_key, "")
            if not ids_str:
                output.error(f"No sample IDs for {policy_key} K={args.k}")
                sys.exit(1)
            sample_ids = [s.strip() for s in ids_str.split(",") if s.strip()]
            output.info(f"Using {len(sample_ids)} verdict samples for {policy_key} K={args.k}")
        elif args.task:
            policy_key = args.task
            k_key = f"K{args.k}"
            if policy_key not in all_sample_ids_data:
                output.error(f"Task {policy_key} not found in verdict samples")
                sys.exit(1)
            ids_str = all_sample_ids_data[policy_key].get(k_key, "")
            if not ids_str:
                output.error(f"No sample IDs for {policy_key} K={args.k}")
                sys.exit(1)
            sample_ids = [s.strip() for s in ids_str.split(",") if s.strip()]
            output.info(f"Using {len(sample_ids)} verdict samples for {policy_key} K={args.k}")
        # else: multi-policy mode - sample_ids resolved per-policy in the loop

    # Parse policies using centralized expand_policies()
    policies = []
    if not args.task:
        try:
            policies = expand_policies(
                policy=args.policy,
                topic=args.topic,
                group=args.group,
            )
        except ValueError as e:
            output.error(str(e))
            sys.exit(1)
        if not args.policy and not args.topic and not args.group:
            output.info(f"Running all {len(policies)} policies")
        elif args.topic:
            output.info(f"Running topic {args.topic}: {len(policies)} policies")
        elif args.group:
            output.info(f"Running group {args.group}: {len(policies)} policies")

    if len(policies) > 1:
        # Multiple policies - run with progress tracking
        # Use AMOSProgressTracker for AMOS method (shows internal step progress)
        # Use MultiPolicyProgress for baseline methods (shows policy-level progress)
        is_amos = args.method == "amos"
        output.info(f"Multi-policy run: method={args.method}, is_amos={is_amos}, policies={len(policies)}")

        # Save output state (individual runs will set suppress_all=True)
        original_suppress_all = output._suppress_all
        if is_amos:
            progress_tracker = output.create_amos_progress(policies)
        else:
            progress_tracker = output.create_multi_policy_progress(policies)

        async def run_single_policy_safe(policy: str) -> Dict[str, Any]:
            """Run a single policy with error handling and progress tracking."""
            progress_tracker.start_policy(policy)
            try:
                # Resolve sample_ids per-policy if using --sample with multiple policies
                policy_sample_ids = sample_ids
                if all_sample_ids_data and not sample_ids:
                    policy_key = policy.replace("/", "_")
                    k_key = f"K{args.k}"
                    ids_str = all_sample_ids_data.get(policy_key, {}).get(k_key, "")
                    if ids_str:
                        policy_sample_ids = [s.strip() for s in ids_str.split(",") if s.strip()]

                # For AMOS, get callback to report internal progress
                callback = progress_tracker.get_callback(policy) if is_amos else None

                result = await run_experiment(
                    task_id=args.task,
                    policy_id=policy,
                    domain=args.domain,
                    k=args.k,
                    n=args.n,
                    skip=args.skip,
                    model=args.model,
                    verbose=False,  # Quiet for parallel runs
                    dev=args.dev,
                    benchmark=benchmark,
                    force=args.force,
                    method=args.method,
                    token_limit=args.token_limit,
                    mode=args.mode,
                    batch_id=args.batch_id,
                    top_k=args.top_k,
                    batch_size=args.batch_size,
                    sample_ids=policy_sample_ids,
                    phase=args.phase,
                    seed_path=args.seed,
                    run_number=args.run,
                    suppress_output=True,  # Suppress all output for progress bar
                    progress_callback=callback,
                )
                progress_tracker.complete_policy(policy, result)
                return result
            except Exception as e:
                result = {"error": str(e), "policy_id": policy}
                progress_tracker.complete_policy(policy, result)
                return result

        async def run_multiple_policies():
            with progress_tracker:
                tasks = [run_single_policy_safe(policy) for policy in policies]
                results = await gather_with_concurrency(len(policies), tasks)
            return dict(zip(policies, results))

        run_result = asyncio.run(run_multiple_policies())

        # Print summary table (detailed for AMOS, simple for baseline)
        if is_amos:
            try:
                progress_tracker.print_detailed_summary(n=args.n, k=args.k, method=args.method)
            except Exception as e:
                output.error(f"Failed to print detailed summary: {e}")
                import traceback
                traceback.print_exc()
                progress_tracker.print_summary()
        else:
            progress_tracker.print_summary()

        # Print aggregated usage summary for multi-policy runs
        # Restore output state (individual runs set suppress_all=True)
        output._suppress_all = original_suppress_all

        total_tokens = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0
        total_latency_ms = 0.0
        total_llm_calls = 0
        total_embedding_tokens = 0
        total_embedding_cost = 0.0

        for policy_id, result in run_result.items():
            if isinstance(result, dict) and not result.get("error"):
                usage = result.get("aggregated_usage", {})
                total_tokens += usage.get("total_tokens", 0)
                total_prompt_tokens += usage.get("total_prompt_tokens", 0)
                total_completion_tokens += usage.get("total_completion_tokens", 0)
                total_cost += usage.get("total_cost_usd", 0.0)
                total_latency_ms += usage.get("total_latency_ms", 0.0) or 0.0
                total_llm_calls += usage.get("total_llm_calls", 0)
                total_embedding_tokens += usage.get("total_embedding_tokens", 0)
                total_embedding_cost += usage.get("total_embedding_cost_usd", 0.0)

        if total_tokens > 0 or total_cost > 0:
            latency_str = f"{total_latency_ms / 1000:.1f}s" if total_latency_ms else "N/A (batch)"
            usage_rows = [
                ["Tokens", f"{total_tokens:,}", f"({total_prompt_tokens:,} prompt + {total_completion_tokens:,} completion)"],
                ["Cost", f"${total_cost:.4f}", ""],
                ["Runtime", latency_str, f"({total_llm_calls} LLM calls)"],
            ]
            if total_embedding_tokens > 0:
                usage_rows.insert(1, ["Embedding Tokens", f"{total_embedding_tokens:,}", f"(${total_embedding_cost:.4f})"])
            output.print_table("USAGE SUMMARY (all policies)", ["Metric", "Value", "Details"], usage_rows)
    else:
        # Single policy (or task)
        run_result = asyncio.run(
            run_experiment(
                task_id=args.task,
                policy_id=policies[0] if policies else None,
                domain=args.domain,
                k=args.k,
                n=args.n,
                skip=args.skip,
                model=args.model,
                verbose=not args.quiet,
                dev=args.dev,
                benchmark=benchmark,
                force=args.force,
                method=args.method,
                token_limit=args.token_limit,
                mode=args.mode,
                batch_id=args.batch_id,
                top_k=args.top_k,
                batch_size=args.batch_size,
                sample_ids=sample_ids,
                phase=args.phase,
                seed_path=args.seed,
                run_number=args.run,
            )
        )

        # Handle batch submission notification
        if isinstance(run_result, dict):
            batch_id = run_result.get("batch_id")
            manifest_saved = run_result.get("manifest_saved", False)
            if batch_id and manifest_saved:
                # Batch submitted with manifest - re-run same command to check
                pass  # Message already printed in run_experiment()
            elif batch_id and not args.batch_id:
                # Legacy: batch submitted without manifest tracking
                output.info("Batch submitted. Re-run this command with --batch-id to check status.")
                output.console.print(f"  [dim]--batch-id {batch_id}[/dim]")


if __name__ == "__main__":
    main()
