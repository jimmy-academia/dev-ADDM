"""
CLI: Run direct LLM baseline evaluation.

Usage:
    .venv/bin/python -m addm.tasks.cli.run_baseline --task G1a -n 5
    .venv/bin/python -m addm.tasks.cli.run_baseline --task G1a --all
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


def load_dataset(domain: str, k: int) -> List[Dict[str, Any]]:
    """Load dataset for given K value."""
    dataset_path = Path(f"data/processed/{domain}/dataset_K{k}.jsonl")
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
) -> Dict[str, Any]:
    """Evaluate a single restaurant."""
    business = restaurant.get("business", {})
    name = business.get("name", "Unknown")
    business_id = business.get("business_id", "")

    prompt = build_prompt(restaurant, agenda)

    try:
        response = await llm.call_async([{"role": "user", "content": prompt}])
        verdict = extract_verdict(response)
        risk_score = extract_risk_score(response)

        return {
            "business_id": business_id,
            "name": name,
            "response": response,
            "verdict": verdict,
            "risk_score": risk_score,
            "prompt_chars": len(prompt),
        }
    except Exception as e:
        return {
            "business_id": business_id,
            "name": name,
            "error": str(e),
        }


async def run_baseline(
    task_id: str,
    domain: str = "yelp",
    k: int = 50,
    n: int = 1,
    skip: int = 0,
    model: str = "gpt-4o-mini",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run baseline evaluation."""
    # Load task to get agenda
    task = load_task(task_id, domain)
    agenda = task.parsed_prompt.full_text  # Use full prompt as agenda

    # Load dataset
    restaurants = load_dataset(domain, k)
    if n > 0:
        restaurants = restaurants[skip : skip + n]
    else:
        restaurants = restaurants[skip:]

    print(f"\n{'='*70}")
    print(f"Direct LLM Baseline: {task_id}")
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
    gt_verdicts = load_ground_truth(task_id, domain, k)

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
    results_dir = Path(f"results/baseline/{task_id}")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"run_{timestamp}.json"

    output = {
        "task": task_id,
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
    parser.add_argument("--task", type=str, required=True, help="Task ID (e.g., G1a)")
    parser.add_argument("--domain", type=str, default="yelp", help="Domain")
    parser.add_argument("--k", type=int, default=50, help="Reviews per restaurant")
    parser.add_argument("-n", type=int, default=1, help="Number of restaurants (0=all)")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    asyncio.run(
        run_baseline(
            task_id=args.task,
            domain=args.domain,
            k=args.k,
            n=args.n,
            skip=args.skip,
            model=args.model,
            verbose=not args.quiet,
        )
    )


if __name__ == "__main__":
    main()
