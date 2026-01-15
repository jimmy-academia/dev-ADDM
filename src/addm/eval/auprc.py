#!/usr/bin/env python3
"""
Cumulative Ordinal AUPRC Scoring for G1.1 (Peanut Allergy Safety)

Implements Adjusted AUPRC = Ordinal_AUPRC × Avg_Primitive_Accuracy

This scoring approach is inspired by process reward models (Lightman et al. 2023,
"Let's Verify Step by Step") which evaluate intermediate reasoning steps rather
than just final answers. By incorporating primitive accuracy, we penalize models
that reach correct conclusions through incorrect reasoning.
"""

import json
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from typing import Dict, List, Tuple, Optional


# Ordinal class mapping
CLASS_ORDER = {'Low Risk': 0, 'High Risk': 1, 'Critical Risk': 2}
CLASS_NAMES = ['Low Risk', 'High Risk', 'Critical Risk']

# Default tolerances for V1 primitives
DEFAULT_TOLERANCES_V1 = {
    'n_total_incidents': 0,
    'incident_score': 0,
    'recency_decay': 0.1,
    'credibility_factor': 0.2,
    'final_risk_score': 1.5,
}

# Default tolerances for V2 primitives
DEFAULT_TOLERANCES_V2 = {
    'n_total_incidents': 0,
    'n_allergy_reviews': 0,
    'trust_score': 0.1,
    'adjusted_incident_score': 1.0,
    'trajectory_multiplier': 0,
    'recency_decay': 0.1,
    'credibility_factor': 0.2,
    'cuisine_impact': 0.2,
    'incident_impact': 2.0,
    'trust_impact': 0.3,
    'positive_credit': 0.2,
    'final_risk_score': 1.5,
}


def compute_primitive_accuracy(
    parsed: Dict,
    ground_truth: Dict,
    tolerances: Dict[str, float],
    scoring_fields: Optional[List[str]] = None
) -> float:
    """
    Compute primitive accuracy for a single restaurant.

    Args:
        parsed: Predicted primitive values (keys are UPPERCASE)
        ground_truth: Expected primitive values (keys are lowercase)
        tolerances: Acceptable tolerance per primitive
        scoring_fields: Which primitives to score (defaults to all in tolerances)

    Returns:
        Accuracy in [0, 1] = fraction of primitives within tolerance
    """
    if scoring_fields is None:
        scoring_fields = list(tolerances.keys())

    n_correct = 0
    n_total = 0

    for field in scoring_fields:
        # Skip final_risk_score as it's captured by AUPRC
        if field == 'final_risk_score':
            continue

        gt_val = ground_truth.get(field)
        pred_val = parsed.get(field.upper())

        if gt_val is None or pred_val is None:
            continue

        tol = tolerances.get(field, 0)
        n_total += 1

        # Check if within tolerance
        if isinstance(gt_val, (int, float)) and isinstance(pred_val, (int, float)):
            if abs(float(gt_val) - float(pred_val)) <= tol:
                n_correct += 1
        elif gt_val == pred_val:
            n_correct += 1

    return n_correct / n_total if n_total > 0 else 0.0


def compute_avg_primitive_accuracy(
    runs: List[Dict],
    tolerances: Optional[Dict[str, float]] = None,
    scoring_fields: Optional[List[str]] = None
) -> Tuple[float, List[float]]:
    """
    Compute average primitive accuracy across all restaurants.

    Args:
        runs: List of run results with 'ground_truth' and 'parsed' keys
        tolerances: Tolerance per primitive (auto-detected from task if None)
        scoring_fields: Which primitives to score (auto-detected if None)

    Returns:
        (avg_accuracy, per_restaurant_accuracies)
    """
    accuracies = []

    for run in runs:
        gt = run.get('ground_truth', {})
        parsed = run.get('parsed', {})

        if not gt or not parsed:
            continue

        # Auto-detect version from ground_truth keys
        if tolerances is None:
            if 'trust_score' in gt:
                tolerances = DEFAULT_TOLERANCES_V2
            else:
                tolerances = DEFAULT_TOLERANCES_V1

        if scoring_fields is None:
            scoring_fields = [k for k in tolerances.keys() if k != 'final_risk_score']

        acc = compute_primitive_accuracy(parsed, gt, tolerances, scoring_fields)
        accuracies.append(acc)

    if not accuracies:
        return 0.0, []

    return float(np.mean(accuracies)), accuracies


def load_ground_truth(gt_file: str) -> Dict[str, dict]:
    """Load semantic ground truth indexed by restaurant name."""
    with open(gt_file, 'r') as f:
        data = json.load(f)
    return {item['restaurant']: item for item in data}


def calculate_ordinal_auprc(
    y_true_ordinal: np.ndarray,
    y_scores: np.ndarray,
    class_names: List[str] = CLASS_NAMES
) -> Dict[str, float]:
    """
    Calculate Cumulative Ordinal AUPRC.

    Args:
        y_true_ordinal: Ground truth ordinal labels (0=Low, 1=High, 2=Critical)
        y_scores: Predicted risk scores (higher = more risky)
        class_names: Names for each ordinal class

    Returns:
        Dictionary with all AUPRC metrics
    """
    n_classes = len(class_names)
    n_samples = len(y_true_ordinal)

    results = {
        'n_samples': n_samples,
        'distribution': {},
    }

    # Count class distribution
    for i, name in enumerate(class_names):
        count = (y_true_ordinal == i).sum()
        results['distribution'][name] = int(count)

    # === CUMULATIVE ORDINAL AUPRC ===
    # For each threshold k, compute AUPRC for P(Y >= k)

    cumulative_auprcs = []
    cumulative_naps = []  # Normalized (chance-corrected)

    for k in range(1, n_classes):  # k=1 (>=High), k=2 (>=Critical)
        # Binary: 1 if class >= k, else 0
        y_binary = (y_true_ordinal >= k).astype(int)
        prevalence = y_binary.mean()

        threshold_name = f">={class_names[k]}"

        if prevalence > 0 and prevalence < 1:
            ap = average_precision_score(y_binary, y_scores)
            # Normalized AP: (AP - random) / (1 - random)
            nap = (ap - prevalence) / (1 - prevalence)
        else:
            ap = 0.0
            nap = 0.0

        results[f'auprc_ge_{k}'] = ap
        results[f'nap_ge_{k}'] = nap
        results[f'prevalence_ge_{k}'] = prevalence
        results[f'threshold_{k}_name'] = threshold_name

        cumulative_auprcs.append(ap)
        cumulative_naps.append(nap)

    # Combined Ordinal AUPRC (mean of cumulative thresholds)
    results['ordinal_auprc'] = np.mean(cumulative_auprcs)
    results['ordinal_nap'] = np.mean(cumulative_naps)

    # === ADDITIONAL METRICS ===

    # Severity ordering: Among risky (High+Critical), is Critical ranked higher?
    mask_risky = y_true_ordinal >= 1
    if mask_risky.sum() > 1:
        y_risky = y_true_ordinal[mask_risky]
        scores_risky = y_scores[mask_risky]
        y_critical_among_risky = (y_risky == 2).astype(int)

        if y_critical_among_risky.sum() > 0 and y_critical_among_risky.sum() < len(y_critical_among_risky):
            results['severity_ordering_ap'] = average_precision_score(y_critical_among_risky, scores_risky)
        else:
            results['severity_ordering_ap'] = np.nan
    else:
        results['severity_ordering_ap'] = np.nan

    return results


def calculate_from_results_file(results_file: str, gt_file: Optional[str] = None) -> Dict[str, float]:
    """
    Calculate ordinal AUPRC and Adjusted AUPRC from a benchmark results file.

    The Adjusted AUPRC incorporates primitive accuracy to penalize models that
    reach correct conclusions through incorrect reasoning:

        Adjusted_AUPRC = Ordinal_AUPRC × Avg_Primitive_Accuracy

    Args:
        results_file: Path to detailed results JSON
        gt_file: Optional path to ground truth file (uses embedded GT if not provided)

    Returns:
        Dictionary with all metrics including:
        - ordinal_auprc: Raw AUPRC based on predicted scores vs GT verdicts
        - avg_primitive_accuracy: Mean accuracy of intermediate calculations
        - adjusted_auprc: ordinal_auprc × avg_primitive_accuracy
    """
    with open(results_file, 'r') as f:
        data = json.load(f)

    y_true_ordinal = []
    y_scores = []
    restaurants = []
    valid_runs = []

    for run in data.get('runs', []):
        if 'results' not in run:
            continue
        if run.get('restaurant') == 'validation_restaurant':
            continue

        # Ground truth verdict
        gt_verdict = run['results']['verdict']['expected']
        if gt_verdict not in CLASS_ORDER:
            continue

        # Predicted score
        pred_score = run['results']['final_risk_score']['predicted']
        if pred_score is None:
            pred_score = 0.0

        y_true_ordinal.append(CLASS_ORDER[gt_verdict])
        y_scores.append(float(pred_score))
        restaurants.append(run.get('restaurant', 'Unknown'))
        valid_runs.append(run)

    y_true_ordinal = np.array(y_true_ordinal)
    y_scores = np.array(y_scores)

    # Calculate standard ordinal AUPRC
    metrics = calculate_ordinal_auprc(y_true_ordinal, y_scores)

    # Calculate primitive accuracy
    avg_prim_acc, per_restaurant_acc = compute_avg_primitive_accuracy(valid_runs)

    # Compute Adjusted AUPRC = Ordinal_AUPRC × Avg_Primitive_Accuracy
    metrics['avg_primitive_accuracy'] = avg_prim_acc
    metrics['adjusted_auprc'] = metrics['ordinal_auprc'] * avg_prim_acc
    metrics['adjusted_nap'] = metrics['ordinal_nap'] * avg_prim_acc

    # Store per-restaurant accuracies for analysis
    metrics['primitive_accuracy_std'] = float(np.std(per_restaurant_acc)) if per_restaurant_acc else 0.0
    metrics['primitive_accuracy_min'] = float(min(per_restaurant_acc)) if per_restaurant_acc else 0.0
    metrics['primitive_accuracy_max'] = float(max(per_restaurant_acc)) if per_restaurant_acc else 0.0

    return metrics


def print_report(metrics: Dict[str, float]) -> None:
    """Print formatted AUPRC report."""
    print("=" * 70)
    print("CUMULATIVE ORDINAL AUPRC REPORT")
    print("=" * 70)
    print(f"Samples: {metrics['n_samples']}")
    print(f"Distribution: {metrics['distribution']}")
    print()

    print("CUMULATIVE THRESHOLDS:")
    print("-" * 50)

    # Threshold 1: >= High (High + Critical vs Low)
    print(f"  Threshold 1: {metrics.get('threshold_1_name', '>=High')}")
    print(f"    Prevalence:  {metrics['prevalence_ge_1']:.3f}")
    print(f"    AUPRC:       {metrics['auprc_ge_1']:.3f}")
    print(f"    nAP:         {metrics['nap_ge_1']:.3f}")
    print()

    # Threshold 2: >= Critical (Critical vs Rest)
    print(f"  Threshold 2: {metrics.get('threshold_2_name', '>=Critical')}")
    print(f"    Prevalence:  {metrics['prevalence_ge_2']:.3f}")
    print(f"    AUPRC:       {metrics['auprc_ge_2']:.3f}")
    print(f"    nAP:         {metrics['nap_ge_2']:.3f}")
    print()

    print("=" * 70)
    print("COMBINED ORDINAL METRICS:")
    print("-" * 50)
    print(f"  Ordinal AUPRC:  {metrics['ordinal_auprc']:.3f}  (mean of cumulative AUPRCs)")
    print(f"  Ordinal nAP:    {metrics['ordinal_nap']:.3f}  (chance-corrected)")

    if not np.isnan(metrics.get('severity_ordering_ap', np.nan)):
        print(f"  Severity Order: {metrics['severity_ordering_ap']:.3f}  (Critical > High among risky)")

    # Print Adjusted AUPRC if available
    if 'avg_primitive_accuracy' in metrics:
        print()
        print("=" * 70)
        print("PROCESS-REWARD ADJUSTED METRICS:")
        print("-" * 50)
        print(f"  Primitive Acc:  {metrics['avg_primitive_accuracy']:.3f}  (mean accuracy of intermediate calculations)")
        print(f"    - Range:      [{metrics.get('primitive_accuracy_min', 0):.3f}, {metrics.get('primitive_accuracy_max', 0):.3f}]")
        print(f"    - Std Dev:    {metrics.get('primitive_accuracy_std', 0):.3f}")
        print()
        print(f"  Adjusted AUPRC: {metrics['adjusted_auprc']:.3f}  = {metrics['ordinal_auprc']:.3f} × {metrics['avg_primitive_accuracy']:.3f}")
        print(f"  Adjusted nAP:   {metrics['adjusted_nap']:.3f}  = {metrics['ordinal_nap']:.3f} × {metrics['avg_primitive_accuracy']:.3f}")
        print()
        print("  Formula: Adjusted_AUPRC = Ordinal_AUPRC × Avg_Primitive_Accuracy")
        print("  Rationale: Penalizes correct conclusions from incorrect reasoning")
        print("  Reference: Lightman et al. 2023, 'Let's Verify Step by Step'")

    print("=" * 70)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python score_auprc.py <results_file.json>")
    else:
        metrics = calculate_from_results_file(sys.argv[1])
        print_report(metrics)
