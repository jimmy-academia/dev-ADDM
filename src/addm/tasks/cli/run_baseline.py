"""
CLI: Run direct LLM baseline evaluation.

Usage:
    # Legacy task-based (loads from data/tasks/yelp/G1a_prompt.txt)
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from addm.llm import LLMService
from addm.tasks.base import load_task
from addm.utils.async_utils import gather_with_concurrency
from addm.eval import compute_ordinal_auprc, VERDICT_TO_ORDINAL


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
    gt_path = Path(f"data/tasks/{domain}/{task_id}_K{k}_groundtruth.json")
    if not gt_path.exists():
        # Fallback to old format without K
        gt_path = Path(f"data/tasks/{domain}/{task_id}_groundtruth.json")
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


def build_messages(restaurant: Dict[str, Any], agenda: str, system_prompt: str) -> List[Dict[str, str]]:
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
    n: int = 1,
    skip: int = 0,
    model: str = "gpt-5-nano",
    verbose: bool = True,
    dev: bool = False,
) -> Dict[str, Any]:
    """Run baseline evaluation.

    Args:
        task_id: Legacy task ID (e.g., "G1a") - loads from data/tasks/
        policy_id: Policy ID (e.g., "G1_allergy_V2") - loads from data/query/
        domain: Domain (default: yelp)
        k: Reviews per restaurant
        n: Number of restaurants (0=all)
        skip: Skip first N restaurants
        model: LLM model to use
        verbose: Print detailed output
        dev: If True, save to results/dev/ instead of results/baseline/

    Either task_id or policy_id must be provided.
    """
    if not task_id and not policy_id:
        raise ValueError("Either task_id or policy_id must be provided")

    # Determine run mode and load agenda
    if policy_id:
        # Policy mode: load from data/query/
        agenda = load_policy_prompt(policy_id, domain)
        run_id = policy_id.replace("/", "_")
        # Map policy to task for ground truth comparison
        gt_task_id = policy_to_task_id(policy_id)
    else:
        # Legacy task mode: load from data/tasks/
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

    print(f"\n{'='*70}")
    print(f"Direct LLM Baseline: {run_id}")
    if policy_id and gt_task_id:
        print(f"Ground truth from: {gt_task_id}")
    print(f"{'='*70}")
    print(f"Restaurants: {len(restaurants)}")
    print(f"Model: {model}")
    print(f"K: {k}")
    print(f"{'='*70}\n")

    # Configure LLM
    llm = LLMService()
    llm.configure(model=model, temperature=0.0)

    # Run evaluations (with concurrency limit)
    tasks = [eval_restaurant(r, agenda, llm) for r in restaurants]
    results = await gather_with_concurrency(32, tasks)

    # Load ground truth for scoring
    gt_verdicts = {}
    if gt_task_id:
        gt_verdicts = load_ground_truth(gt_task_id, domain, k)
    if not gt_verdicts:
        print(f"[WARN] No ground truth found for {gt_task_id or run_id}")

    # Score and print results
    correct = 0
    total = 0

    for result in results:
        if "error" in result:
            print(f"\n[ERROR] {result['name']}: {result['error']}")
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
        print(f"\n{'='*70}")
        print(f"Restaurant: {result['name']}")
        print(f"Predicted: {pred_verdict} (score={risk_score}) | GT: {gt_verdict} | {'✓' if is_correct else '✗'}")
        print(f"{'='*70}")
        if verbose:
            # Print first 1000 chars of response
            resp = result["response"]
            print(resp[:1000] + ("..." if len(resp) > 1000 else ""))

    # Print accuracy
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n{'='*70}")
    print(f"ACCURACY: {correct}/{total} = {accuracy:.1%}")
    print(f"{'='*70}")

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

        print(f"\n{'='*70}")
        print("AUPRC METRICS:")
        for key, val in auprc_metrics.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.3f}")
            else:
                print(f"  {key}: {val}")
        print(f"{'='*70}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if dev:
        # Dev mode: results/dev/{timestamp}_{run_id}/
        results_dir = Path(f"results/dev/{timestamp}_{run_id}")
    else:
        # Benchmark mode: results/baseline/{run_id}/
        results_dir = Path(f"results/baseline/{run_id}")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / f"results.json"

    output = {
        "run_id": run_id,
        "task_id": task_id,
        "policy_id": policy_id,
        "gt_task_id": gt_task_id,
        "domain": domain,
        "model": model,
        "k": k,
        "n": len(results),
        "timestamp": timestamp,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "auprc": auprc_metrics,
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")

    return output


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run direct LLM baseline")

    # Task or policy (mutually exclusive, one required)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", type=str, help="Legacy task ID (e.g., G1a)")
    group.add_argument("--policy", type=str, help="Policy ID (e.g., G1_allergy_V2 or G1/allergy/V2)")

    parser.add_argument("--domain", type=str, default="yelp", help="Domain")
    parser.add_argument("--k", type=int, default=50, help="Reviews per restaurant")
    parser.add_argument("-n", type=int, default=1, help="Number of restaurants (0=all)")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N")
    parser.add_argument("--model", type=str, default="gpt-5-nano", help="Model")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--dev", action="store_true", help="Dev mode: save to results/dev/{timestamp}_{id}/")

    args = parser.parse_args()

    asyncio.run(
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
        )
    )


if __name__ == "__main__":
    main()
