"""
Evaluation metrics for ADDM benchmark.

Separate metrics (not combined/adjusted):
1. AUPRC (method verdict) - ranking quality of method's output
2. Primitive accuracy - correctness of intermediate calculations
3. Verdict consistency - does method verdict match computed-from-primitives verdict?
4. AUPRC (computed verdict) - ranking quality from primitives
"""

from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import average_precision_score
from typing import Dict, List, Optional, Any

# Ordinal class mapping (default for G1 risk policies)
VERDICT_TO_ORDINAL = {"Low Risk": 0, "High Risk": 1, "Critical Risk": 2}
CLASS_NAMES = ["Low Risk", "High Risk", "Critical Risk"]

# Cache for loaded policy verdicts
_policy_verdicts_cache: Dict[str, List[str]] = {}


def get_policy_verdicts(policy_id: str) -> List[str]:
    """Load valid verdicts for a policy from YAML.

    Args:
        policy_id: Policy identifier (e.g., "G1_allergy_V2" or "G1/allergy/V2")

    Returns:
        List of valid verdict strings in order (low to high)
    """
    if policy_id in _policy_verdicts_cache:
        return _policy_verdicts_cache[policy_id]

    # Normalize policy_id to path format
    normalized = policy_id.replace("_", "/")
    parts = normalized.split("/")
    if len(parts) >= 3:
        group, topic, version = parts[0], parts[1], parts[2]
        yaml_path = Path(f"src/addm/query/policies/{group}/{topic}/{version}.yaml")

        if yaml_path.exists():
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            verdicts = data.get("normative", {}).get("decision", {}).get("verdicts", [])
            if verdicts:
                _policy_verdicts_cache[policy_id] = verdicts
                return verdicts

    # Fallback to default risk verdicts
    return CLASS_NAMES


def normalize_verdict(verdict: Optional[str], policy_id: Optional[str] = None) -> Optional[str]:
    """Normalize verdict string for comparison.

    Handles:
    - Extra quotes: "'Low Risk'" -> "Low Risk"
    - Whitespace: " Low Risk " -> "Low Risk"
    - Case-insensitive matching to policy's valid verdicts

    Args:
        verdict: Raw verdict string
        policy_id: Optional policy ID to load valid verdicts for matching

    Returns:
        Normalized verdict or None if invalid
    """
    if verdict is None:
        return None

    # Strip whitespace and quotes (single, double, backticks) from either end
    normalized = verdict.strip().strip("'\"`").strip()

    if not normalized:
        return None

    # If policy_id provided, match against valid verdicts (case-insensitive)
    if policy_id:
        valid_verdicts = get_policy_verdicts(policy_id)
        normalized_lower = normalized.lower()
        for valid in valid_verdicts:
            if valid.lower() == normalized_lower:
                return valid  # Return canonical form

    return normalized


def compute_ordinal_auprc(
    y_true_ordinal: np.ndarray,
    y_scores: np.ndarray,
) -> Dict[str, float]:
    """
    Compute cumulative ordinal AUPRC.

    Args:
        y_true_ordinal: GT ordinal labels (0=Low, 1=High, 2=Critical)
        y_scores: Predicted scores (higher = more risky)

    Returns:
        Dict with auprc_ge_high, auprc_ge_critical, ordinal_auprc
    """
    results = {}
    n_samples = len(y_true_ordinal)

    if n_samples < 2:
        return {"ordinal_auprc": 0.0, "n_samples": n_samples}

    # AUPRC for >=High (High+Critical vs Low)
    y_binary_high = (y_true_ordinal >= 1).astype(int)
    if y_binary_high.sum() > 0 and y_binary_high.sum() < len(y_binary_high):
        results["auprc_ge_high"] = average_precision_score(y_binary_high, y_scores)

    # AUPRC for >=Critical (Critical vs rest)
    y_binary_crit = (y_true_ordinal >= 2).astype(int)
    if y_binary_crit.sum() > 0 and y_binary_crit.sum() < len(y_binary_crit):
        results["auprc_ge_critical"] = average_precision_score(y_binary_crit, y_scores)

    # Mean ordinal AUPRC
    auprc_values = [v for k, v in results.items() if k.startswith("auprc_")]
    results["ordinal_auprc"] = np.mean(auprc_values) if auprc_values else 0.0
    results["n_samples"] = n_samples

    return results


def compute_primitive_accuracy(
    predicted: Dict[str, Any],
    ground_truth: Dict[str, Any],
    tolerances: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute primitive accuracy for a single sample.

    Args:
        predicted: Predicted primitive values
        ground_truth: Expected primitive values
        tolerances: Acceptable tolerance per field (default: exact match)

    Returns:
        Accuracy in [0, 1]
    """
    if tolerances is None:
        tolerances = {}

    n_correct = 0
    n_total = 0

    for field, gt_val in ground_truth.items():
        pred_val = predicted.get(field)
        if pred_val is None:
            continue

        tol = tolerances.get(field, 0)
        n_total += 1

        if isinstance(gt_val, (int, float)) and isinstance(pred_val, (int, float)):
            if abs(float(gt_val) - float(pred_val)) <= tol:
                n_correct += 1
        elif gt_val == pred_val:
            n_correct += 1

    return n_correct / n_total if n_total > 0 else 0.0


def compute_verdict_consistency(
    method_verdict: str,
    computed_verdict: str,
) -> int:
    """
    Check if method's verdict matches verdict computed from its primitives.

    Returns:
        1 if consistent, 0 if inconsistent
    """
    return 1 if method_verdict == computed_verdict else 0


def evaluate_results(
    results: List[Dict[str, Any]],
    gt_verdicts: Dict[str, str],
    compute_verdict_fn: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics.

    Args:
        results: List of method outputs with verdict, risk_score, primitives
        gt_verdicts: Ground truth verdicts by business_id
        compute_verdict_fn: Function to compute verdict from primitives

    Returns:
        Dict with all metrics
    """
    # Collect data for AUPRC
    y_true = []
    y_scores_method = []  # From method's risk_score
    y_scores_computed = []  # From computed primitives

    # For accuracy and consistency
    primitive_accuracies = []
    verdict_consistencies = []

    for result in results:
        if "error" in result:
            continue

        biz_id = result.get("business_id")
        gt_verdict = gt_verdicts.get(biz_id)
        if not gt_verdict:
            continue

        gt_ord = VERDICT_TO_ORDINAL.get(gt_verdict)
        if gt_ord is None:
            continue

        # Method verdict and score
        method_verdict = result.get("verdict")
        method_score = result.get("risk_score")

        if method_score is not None:
            y_true.append(gt_ord)
            y_scores_method.append(method_score)

        # TODO: Compute verdict from primitives if compute_verdict_fn provided
        # computed_verdict = compute_verdict_fn(result.get("primitives", {}))
        # verdict_consistencies.append(compute_verdict_consistency(method_verdict, computed_verdict))

    metrics = {}

    # 1. AUPRC from method verdict/score
    if y_true and y_scores_method:
        auprc_method = compute_ordinal_auprc(
            np.array(y_true),
            np.array(y_scores_method)
        )
        metrics["auprc_method"] = auprc_method

    # 2. Primitive accuracy (if available)
    if primitive_accuracies:
        metrics["primitive_accuracy"] = {
            "mean": np.mean(primitive_accuracies),
            "std": np.std(primitive_accuracies),
            "min": min(primitive_accuracies),
            "max": max(primitive_accuracies),
        }

    # 3. Verdict consistency (if available)
    if verdict_consistencies:
        metrics["verdict_consistency"] = {
            "mean": np.mean(verdict_consistencies),
            "n_consistent": sum(verdict_consistencies),
            "n_total": len(verdict_consistencies),
        }

    # 4. AUPRC from computed verdict (if available)
    if y_true and y_scores_computed:
        auprc_computed = compute_ordinal_auprc(
            np.array(y_true),
            np.array(y_scores_computed)
        )
        metrics["auprc_computed"] = auprc_computed

    return metrics
