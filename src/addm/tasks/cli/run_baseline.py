"""
CLI: Run direct LLM baseline evaluation.

Usage:
    # Legacy task-based (loads from data/answers/yelp/G1a_prompt.txt)
    .venv/bin/python -m addm.tasks.cli.run_baseline --task G1a -n 5

    # New policy-based (loads from data/query/yelp/G1_allergy_V2_prompt.txt)
    .venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 5
    .venv/bin/python -m addm.tasks.cli.run_baseline --policy G1/allergy/V2 -n 5

    # Dev mode (saves to results/dev/{timestamp}_{id}/)
    .venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 5 --dev

Output directories:
    --dev:     results/dev/{timestamp}_{run_id}/results.json
    (default): results/baseline/{run_id}/results.json
"""

import argparse
import asyncio
import json
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from addm.eval import VERDICT_TO_ORDINAL, compute_ordinal_auprc, compute_unified_metrics
from addm.llm import LLMService
from addm.llm_batch import BatchClient, build_chat_batch_item
from addm.methods.rlm import eval_restaurant_rlm
from addm.tasks.base import load_task
from addm.utils.async_utils import gather_with_concurrency
from addm.utils.debug_logger import DebugLogger, get_debug_logger, set_debug_logger
from addm.utils.logging import ResultLogger, get_result_logger, set_result_logger
from addm.utils.output import output


# Mapping from policy topic to legacy task ID for ground truth comparison
# Policy naming: G{group}_{topic}_V{version} -> Task: G{group}{variant}
POLICY_TO_TASK = {
    "allergy": "a",    # G1_allergy_V* -> G1a
    "dietary": "e",    # G1_dietary_V* -> G1e
    "hygiene": "i",    # G1_hygiene_V* -> G1i
    # Add more mappings as policies are developed
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


def load_policy_prompt(policy_id: str, domain: str = "yelp") -> str:
    """Load prompt from policy-generated file.

    Args:
        policy_id: Policy ID like "G1_allergy_V2" or "G1/allergy/V2"
        domain: Domain (default: yelp)

    Returns:
        Prompt text
    """
    # Normalize policy ID: "G1/allergy/V2" -> "G1_allergy_V2"
    normalized = policy_id.replace("/", "_")

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
        response = await llm.call_async(messages)

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
        }
    except Exception as e:
        return {
            "business_id": business_id,
            "name": name,
            "error": str(e),
        }


async def run_baseline(
    task_id: Optional[str] = None,
    policy_id: Optional[str] = None,
    domain: str = "yelp",
    k: int = 50,
    n: int = 100,
    skip: int = 0,
    model: str = "gpt-5-nano",
    verbose: bool = True,
    dev: bool = False,
    method: str = "direct",
    token_limit: Optional[int] = None,
    mode: str = "ondemand",
    batch_id: Optional[str] = None,
    top_k: int = 20,
    regenerate_seed: bool = False,
) -> Dict[str, Any]:
    """Run baseline evaluation.

    Args:
        task_id: Legacy task ID (e.g., "G1a") - loads from data/answers/
        policy_id: Policy ID (e.g., "G1_allergy_V2") - loads from data/query/
        domain: Domain (default: yelp)
        k: Reviews per restaurant
        n: Number of restaurants (0=all)
        skip: Skip first N restaurants
        model: LLM model to use
        verbose: Print detailed output
        dev: If True, save to results/dev/ instead of results/baseline/
        method: Method to use - "direct" (default), "rlm", "rag", or "amos"
        token_limit: Token budget for RLM (converts to iterations via ~3000 tokens/iter)
        regenerate_seed: Force regenerate Formula Seed for AMOS method

    Either task_id or policy_id must be provided.
    """
    if not task_id and not policy_id:
        raise ValueError("Either task_id or policy_id must be provided")

    # Determine run mode and load agenda
    system_prompt = None
    if policy_id:
        # Policy mode: load from data/query/ with system prompt
        agenda = load_policy_prompt(policy_id, domain)
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
    if n > 0:
        restaurants = restaurants[skip : skip + n]
    else:
        restaurants = restaurants[skip:]

    # Configure output manager
    output.configure(quiet=not verbose, mode=mode)

    # Setup loggers in dev mode (need to determine output_dir early)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if dev:
        output_dir = Path(f"results/dev/{timestamp}_{run_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup result logger
        result_logger = ResultLogger(output_dir / "results.jsonl")
        set_result_logger(result_logger)

        # Setup debug logger (centralized in results/logs/debug/)
        debug_dir = Path(f"results/logs/debug/{run_id}")
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_logger = DebugLogger(debug_dir)
        set_debug_logger(debug_logger)
    else:
        output_dir = Path(f"results/baseline/{run_id}")

    method_name_map = {
        "rlm": "RLM (Recursive LLM)",
        "rag": "RAG (Retrieval Augmented Generation)",
        "direct": "Direct LLM",
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
    })
    if gt_task_id:
        output.status(f"Ground truth: {gt_task_id}_K{k}_groundtruth.json")

    if mode == "24hrbatch":
        if method in ["rlm", "rag"]:
            raise ValueError("24hrbatch mode is only supported for method=direct")

        batch_client = BatchClient()

        if batch_id:
            batch = batch_client.get_batch(batch_id)
            status = _get_batch_field(batch, "status")
            if status not in {"completed", "failed", "expired", "cancelled"}:
                output.batch_status(batch_id, status)
                return {}

            output_file_id = _get_batch_field(batch, "output_file_id")
            error_file_id = _get_batch_field(batch, "error_file_id")
            if error_file_id:
                output.warn(f"Batch has error file: {error_file_id}")

            if not output_file_id:
                output.warn(f"No output file available for batch {batch_id}")
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
                results.append(result)
        else:
            request_items = []
            for restaurant in restaurants:
                business = restaurant.get("business", {})
                business_id = business.get("business_id", "")
                if system_prompt:
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
            output.batch_submitted(batch_id)
            return {"batch_id": batch_id}
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
                max_concurrent=32,
                force_regenerate=regenerate_seed,
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

            # Pre-load Formula Seed (Phase 1) to enable parallel Phase 2 execution
            await amos_method._get_formula_seed(agenda, llm)

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

                return {
                    "business_id": raw_result["sample_id"],
                    "name": restaurant["business"]["name"],
                    "response": raw_result.get("output", ""),
                    "verdict": verdict,
                    "risk_score": raw_result.get("risk_score"),
                    "prompt_chars": raw_result.get("prompt_tokens", 0) * 4,
                    # AMOS-specific metrics
                    "filter_stats": raw_result.get("filter_stats", {}),
                    "extractions_count": raw_result.get("extractions_count", 0),
                    "phase1_cached": raw_result.get("phase1_cached", False),
                    "llm_calls": raw_result.get("llm_calls", 0),
                }

            tasks = [process_amos_sample(s) for s in samples]
            raw_results = await gather_with_concurrency(8, tasks)  # 8 concurrent restaurants
            results = [r for r in raw_results if r is not None]
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
        if "error" in result:
            output.error(f"{result['name']}: {result['error']}")
            continue

        biz_id = result["business_id"]
        gt_verdict = gt_verdicts.get(biz_id)
        pred_verdict = result["verdict"]
        is_correct = pred_verdict == gt_verdict

        if gt_verdict:
            total += 1
            if is_correct:
                correct += 1

        result["gt_verdict"] = gt_verdict
        result["correct"] = is_correct

        risk_score = result.get("risk_score")

        # Log to result logger if enabled
        if result_logger := get_result_logger():
            result_logger.log_result(
                sample_id=biz_id,
                verdict=pred_verdict,
                ground_truth=gt_verdict,
                risk_score=risk_score,
                correct=is_correct,
            )

        # Display result (mode-aware via output.configure())
        if not output._quiet:
            output.rule()
            output.print_result(
                name=result["name"],
                predicted=pred_verdict,
                ground_truth=gt_verdict,
                score=risk_score,
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

            # Display summary
            if intermediate_metrics.get("summary"):
                summary = intermediate_metrics["summary"]
                int_rows = []
                for key in ["n_structured", "incident_precision", "severity_accuracy",
                           "snippet_validity", "verdict_support_rate"]:
                    val = summary.get(key)
                    if val is not None:
                        if isinstance(val, float):
                            int_rows.append([key, f"{val:.3f}"])
                        else:
                            int_rows.append([key, str(val)])
                if int_rows:
                    output.print_table("Intermediate Metrics", ["Metric", "Value"], int_rows)

    # Compute unified 3-score metrics
    unified_metrics = {}
    if gt_verdicts:
        # Load full GT data for unified metrics
        from addm.eval.intermediate_metrics import load_gt_with_incidents, build_reviews_data

        gt_data_full, _ = load_gt_with_incidents(gt_task_id, domain, k)
        reviews_data = build_reviews_data(restaurants)

        unified_metrics = compute_unified_metrics(
            results=results,
            gt_data=gt_data_full,
            gt_verdicts=gt_verdicts,
            method=method,
            reviews_data=reviews_data,
        )

        # Display 3-score summary table
        output.rule()
        score_rows = [
            ["AUPRC", f"{unified_metrics['auprc']:.1%}", "Higher = better ranking"],
            ["Process", f"{unified_metrics['process_score']:.1f}%", ">75% = good"],
            ["Consistency", f"{unified_metrics['consistency_score']:.1f}%", ">75% = good"],
        ]
        output.print_table("UNIFIED EVALUATION SCORES", ["Metric", "Score", "Target"], score_rows)

    # Log metrics to result logger if enabled
    if result_logger := get_result_logger():
        result_logger.log_metrics(
            metrics={"accuracy": accuracy, **auprc_metrics},
            metadata={"run_id": run_id, "n_samples": total},
        )

    # Save results (output_dir determined earlier)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"

    run_output = {
        "run_id": run_id,
        "task_id": task_id,
        "policy_id": policy_id,
        "gt_task_id": gt_task_id,
        "domain": domain,
        "method": method,
        "model": model,
        "k": k,
        "n": len(results),
        "timestamp": timestamp,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "auprc": auprc_metrics,
        "intermediate_metrics": intermediate_metrics,
        "unified_scores": {
            "auprc": unified_metrics.get("auprc", 0.0),
            "process_score": unified_metrics.get("process_score", 0.0),
            "consistency_score": unified_metrics.get("consistency_score", 0.0),
        } if unified_metrics else None,
        "unified_metrics_full": unified_metrics if unified_metrics else None,
        "results": results,
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
    parser = argparse.ArgumentParser(description="Run direct LLM baseline")

    # Task or policy (mutually exclusive, one required)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", type=str, help="Legacy task ID (e.g., G1a)")
    group.add_argument(
        "--policy",
        type=str,
        help="Policy ID (e.g., G1_allergy_V2 or G1/allergy/V2)",
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
        help="Dev mode: save to results/dev/{timestamp}_{id}/",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="direct",
        choices=["direct", "rlm", "rag", "amos"],
        help="Method: direct (default), rlm (recursive LLM), rag (retrieval), or amos (agenda-driven mining)",
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
        "--regenerate-seed",
        action="store_true",
        help="AMOS: Force regenerate Formula Seed even if cached",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="ondemand",
        choices=["ondemand", "24hrbatch"],
        help="LLM execution mode",
    )
    parser.add_argument("--batch-id", type=str, default=None, help="Batch ID for fetch-only runs")

    args = parser.parse_args()

    run_result = asyncio.run(
        run_baseline(
            task_id=args.task,
            policy_id=args.policy,
            domain=args.domain,
            k=args.k,
            n=args.n,
            skip=args.skip,
            model=args.model,
            verbose=not args.quiet,
            dev=args.dev,
            method=args.method,
            token_limit=args.token_limit,
            mode=args.mode,
            batch_id=args.batch_id,
            top_k=args.top_k,
            regenerate_seed=args.regenerate_seed,
        )
    )

    if args.mode == "24hrbatch":
        if not args.batch_id:
            # Batch submitted - print instructions to check status
            batch_id = run_result.get("batch_id") if isinstance(run_result, dict) else None
            if batch_id:
                output.info("Batch submitted. Re-run this command with --batch-id to check status.")
                output.console.print(f"  [dim]--batch-id {batch_id}[/dim]")


if __name__ == "__main__":
    main()
