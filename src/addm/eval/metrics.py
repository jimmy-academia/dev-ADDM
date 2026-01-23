"""
Evaluation metrics for ADDM benchmark.

New metric system (v2 - simplified, interpretable):

Tier 1: Final Quality
- AUPRC: Ordinal ranking quality (>=High, >=Critical)

Tier 3: Process Quality (separate metrics, no weighting)
- Incident Precision: % claimed incidents that exist in GT (by review_id)
- Judgement Accuracy: % correct field values for matched incidents
- Verdict Consistency: Does claimed evidence → claimed verdict?

Removed:
- Process Score (weighted composite - arbitrary weights)
- Accuracy (misleading when class-skewed)
- Incident Recall (partial extraction is fine for efficiency)
"""

from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import average_precision_score
from typing import Dict, List, Optional, Any, Tuple

from addm.eval.constants import (
    VERDICT_TO_ORDINAL,
    CLASS_NAMES,
    SEVERITY_BASE_POINTS,
    MODIFIER_POINTS,
    VERDICT_THRESHOLDS,
)

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


def normalize_judgement(judgement: str, valid_values: List[str]) -> Optional[str]:
    """Extract a known value from a judgement string.

    Handles flexible formatting like "moderate incident" → "moderate",
    "Dismissive" → "dismissive", etc.

    Args:
        judgement: Raw judgement string from method output
        valid_values: List of valid values to look for (lowercase)

    Returns:
        Matched value (lowercase) or None if no match
    """
    if not judgement:
        return None

    judgement_lower = judgement.lower()

    # First try exact match
    if judgement_lower in valid_values:
        return judgement_lower

    # Then try substring match (e.g., "moderate incident" contains "moderate")
    for value in valid_values:
        if value in judgement_lower:
            return value

    return None


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

    # Convert to string if needed (AMOS may return int verdicts)
    verdict_str = str(verdict) if not isinstance(verdict, str) else verdict

    # Strip whitespace and quotes (single, double, backticks) from either end
    normalized = verdict_str.strip().strip("'\"`").strip()

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


def _extract_evidences(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract evidences array from result's parsed field."""
    parsed = result.get("parsed", {})
    if isinstance(parsed, dict):
        return parsed.get("evidences", [])
    return []


def _compute_verdict_from_evidences(evidences: List[Dict[str, Any]]) -> Tuple[str, float]:
    """Compute verdict from evidences using V2 policy scoring rules.

    Args:
        evidences: List of evidence dicts with field and judgement

    Returns:
        Tuple of (computed_verdict, computed_score)
    """
    score = 0.0

    # Valid values for normalization
    severity_values = list(SEVERITY_BASE_POINTS.keys())
    assurance_positive = ["true", "yes", "detected", "false assurance", "no assurance"]
    staff_negative = ["dismissive", "negative", "bad", "rude", "unhelpful"]

    for evidence in evidences:
        field = evidence.get("field", "").upper()
        judgement = evidence.get("judgement", "")

        if field == "INCIDENT_SEVERITY":
            normalized = normalize_judgement(judgement, severity_values)
            if normalized:
                score += SEVERITY_BASE_POINTS.get(normalized, 0)
        elif field in ("ASSURANCE_CLAIM", "ASSURANCE_OF_SAFETY"):
            normalized = normalize_judgement(judgement, assurance_positive)
            if normalized:
                score += MODIFIER_POINTS.get("False assurance", 0)
        elif field == "STAFF_RESPONSE":
            normalized = normalize_judgement(judgement, staff_negative)
            if normalized:
                score += MODIFIER_POINTS.get("Dismissive staff", 0)

    # Determine verdict from score
    if score >= VERDICT_THRESHOLDS["Critical Risk"]:
        verdict = "Critical Risk"
    elif score >= VERDICT_THRESHOLDS["High Risk"]:
        verdict = "High Risk"
    else:
        verdict = "Low Risk"

    return verdict, score


# Rule → Verdict mapping for verdict consistency checks
RULE_TO_VERDICT = {
    "CRITICAL": "Critical Risk",
    "HIGH": "High Risk",
    "LOW": "Low Risk",
}


def compute_score_accuracy(
    results: List[Dict[str, Any]],
    gt_data: Dict[str, Dict[str, Any]],
    policy_id: Optional[str] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Compute score accuracy - does method's computed score match GT?

    Only evaluated for policies with score systems (V2/V3).
    Returns None for V0/V1 (condition-based) policies.

    Args:
        results: Method outputs with parsed.justification.scoring_trace
        gt_data: GT data by business_id with score field
        policy_id: Policy ID to check if scoring applies

    Returns:
        Tuple of (accuracy, details_dict)
        - accuracy: float 0.0-1.0 or None if not applicable/no data
        - details: breakdown with correct, total, per_sample stats
    """
    # Check if policy uses scoring (V2/V3)
    if policy_id:
        version = policy_id.split("_")[-1] if "_" in policy_id else ""
        if version not in ("V2", "V3"):
            return None, {"skipped": "Policy does not use scoring system (V0/V1)"}

    correct = 0
    total = 0
    per_sample = []

    for result in results:
        if "error" in result:
            continue

        biz_id = result.get("business_id")
        gt = gt_data.get(biz_id, {})
        gt_score = gt.get("score")

        if gt_score is None:
            continue

        # Extract method's claimed score
        parsed = result.get("parsed", {})
        if not isinstance(parsed, dict):
            continue

        justification = parsed.get("justification", {})
        if not isinstance(justification, dict):
            continue

        scoring_trace = justification.get("scoring_trace", {})
        if not isinstance(scoring_trace, dict):
            continue

        method_score = scoring_trace.get("total_score")

        if method_score is None:
            continue

        try:
            method_score_float = float(method_score)
            gt_score_float = float(gt_score)
        except (ValueError, TypeError):
            continue

        total += 1
        is_correct = abs(method_score_float - gt_score_float) < 0.01
        if is_correct:
            correct += 1

        per_sample.append({
            "business_id": biz_id,
            "method_score": method_score_float,
            "gt_score": gt_score_float,
            "correct": is_correct,
        })

    if total == 0:
        return None, {"skipped": "No samples with both method score and GT score"}

    return correct / total, {
        "correct": correct,
        "total": total,
        "per_sample": per_sample,
    }


def compute_verdict_consistency_enhanced(
    results: List[Dict[str, Any]],
    policy_id: Optional[str] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Enhanced verdict consistency - checks (evidence + triggered_rule) → verdict.

    For each result:
    1. Compute verdict from method's claimed evidences (score-based)
    2. Get verdict implied by method's claimed triggered_rule
    3. Check if both match method's claimed verdict

    Returns consistent only if:
    - evidence_verdict == method_verdict, AND
    - rule_verdict == method_verdict (if rule provided)

    Args:
        results: Method outputs with parsed.evidences and parsed.justification
        policy_id: Optional policy ID for verdict normalization

    Returns:
        Tuple of (consistency_rate, details_dict)
    """
    consistent = 0
    total = 0
    skipped_no_evidence = 0
    per_sample = []

    for result in results:
        if "error" in result:
            continue

        method_verdict = result.get("verdict")
        if not method_verdict:
            continue

        evidences = _extract_evidences(result)
        if not evidences:
            skipped_no_evidence += 1
            continue

        # 1. Compute verdict from evidences
        computed_verdict, computed_score = _compute_verdict_from_evidences(evidences)

        # 2. Get verdict implied by triggered_rule (if provided)
        parsed = result.get("parsed", {})
        justification = parsed.get("justification", {}) if isinstance(parsed, dict) else {}
        triggered_rule = justification.get("triggered_rule", "") if isinstance(justification, dict) else ""
        rule_verdict = RULE_TO_VERDICT.get(triggered_rule.upper()) if triggered_rule else None

        # 3. Check consistency
        evidence_matches = normalize_verdict(method_verdict, policy_id) == normalize_verdict(computed_verdict, policy_id)
        rule_matches = (rule_verdict is None) or (normalize_verdict(method_verdict, policy_id) == normalize_verdict(rule_verdict, policy_id))

        is_consistent = evidence_matches and rule_matches

        total += 1
        if is_consistent:
            consistent += 1

        per_sample.append({
            "business_id": result.get("business_id"),
            "method_verdict": method_verdict,
            "evidence_verdict": computed_verdict,
            "evidence_score": computed_score,
            "rule_verdict": rule_verdict,
            "triggered_rule": triggered_rule,
            "evidence_matches": evidence_matches,
            "rule_matches": rule_matches,
            "consistent": is_consistent,
        })

    if total == 0:
        return None, {"skipped_no_evidence": skipped_no_evidence, "total": 0}

    return consistent / total, {
        "consistent": consistent,
        "total": total,
        "skipped_no_evidence": skipped_no_evidence,
        "per_sample": per_sample,
    }


def compute_evaluation_metrics(
    results: List[Dict[str, Any]],
    gt_data: Dict[str, Dict[str, Any]],
    gt_verdicts: Dict[str, str],
    method: str = "direct",
    reviews_data: Optional[Dict[str, Dict[str, str]]] = None,
    policy_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute all evaluation metrics (new simplified system - 6 metrics).

    Returns separate, interpretable metrics without weighted composites.

    Tier 1: Final Quality
    - AUPRC: Ordinal ranking quality (>=High, >=Critical)

    Tier 2: Evidence Quality
    - Incident Precision: % claimed incidents that exist in GT
    - Snippet Validity: % snippets that match actual review text

    Tier 3: Reasoning Quality
    - Judgement Accuracy: % correct field values for matched incidents
    - Score Accuracy: Is computed score correct? (V2/V3 only)
    - Verdict Consistency: Does (evidence + triggered_rule) → verdict?

    Args:
        results: List of method outputs with parsed.evidences
        gt_data: {business_id: GT with incidents}
        gt_verdicts: {business_id: verdict}
        method: Method name ("direct", "amos", etc.)
        reviews_data: Optional {business_id: {review_id: text}} for snippet validation
        policy_id: Optional policy ID for verdict normalization

    Returns:
        {
            "auprc": float,  # 0.0-1.0, ordinal ranking quality
            "evidence_precision": float or None,  # 0.0-1.0, % claimed that exist in GT
            "snippet_validity": float or None,  # 0.0-1.0, % snippets that match source
            "judgement_accuracy": float or None,  # 0.0-1.0, field correctness
            "score_accuracy": float or None,  # 0.0-1.0, V2/V3 score correctness
            "verdict_consistency": float or None,  # 0.0-1.0, evidence+rule→verdict
            "details": {
                "auprc_metrics": {...},
                "incident_details": {...},
                "judgement_details": {...},
                "score_details": {...},
                "consistency_details": {...},
            },
            "n_samples": int,
            "n_with_gt": int,
            "n_structured": int,
        }
    """
    from addm.eval.unified_metrics import normalize_amos_output
    from addm.eval.intermediate_metrics import (
        compute_evidence_validity,
        compute_judgement_accuracy,
    )

    # Normalize AMOS outputs if needed
    if method == "amos":
        results = [normalize_amos_output(r) for r in results]

    # Filter to structured results (have parsed.evidences)
    structured_results = [
        r for r in results
        if r.get("parsed") and isinstance(r.get("parsed"), dict)
        and "evidences" in r.get("parsed", {})
    ]

    # 1. Compute AUPRC
    y_true = []
    y_scores = []

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

        # Get risk score (continuous), fallback to verdict ordinal * 5
        risk_score = result.get("risk_score")
        if risk_score is None:
            pred_ord = VERDICT_TO_ORDINAL.get(result.get("verdict"))
            risk_score = pred_ord * 5.0 if pred_ord is not None else None

        if risk_score is not None:
            y_true.append(gt_ord)
            y_scores.append(risk_score)

    auprc_metrics = {}
    auprc = 0.0
    if len(y_true) >= 2:
        auprc_metrics = compute_ordinal_auprc(np.array(y_true), np.array(y_scores))
        auprc = auprc_metrics.get("ordinal_auprc", 0.0)

    # 2. Compute Incident Precision, Incident Recall, and Snippet Validity
    incident_details = {"error": "No structured results or GT data"}
    evidence_precision = None
    evidence_recall = None
    snippet_validity = None

    if structured_results and gt_data:
        incident_details = compute_evidence_validity(
            structured_results, gt_data, reviews_data
        )
        evidence_precision = incident_details.get("evidence_precision")
        evidence_recall = incident_details.get("evidence_recall")
        snippet_validity = incident_details.get("snippet_validity")

    # 3. Compute Judgement Accuracy (policy-agnostic)
    judgement_details = {"error": "No structured results or GT data"}
    judgement_accuracy = None

    if structured_results and gt_data:
        judgement_accuracy, judgement_details = compute_judgement_accuracy(
            structured_results, gt_data
        )

    # 4. Compute Score Accuracy (V2/V3 only)
    score_accuracy, score_details = compute_score_accuracy(
        results, gt_data, policy_id
    )

    # 5. Compute Enhanced Verdict Consistency (with triggered_rule check)
    verdict_consistency, consistency_details = compute_verdict_consistency_enhanced(
        results, policy_id
    )

    return {
        # Tier 1: Final Quality
        "auprc": auprc,
        # Tier 2: Evidence Quality
        "evidence_precision": evidence_precision,
        "evidence_recall": evidence_recall,
        "snippet_validity": snippet_validity,
        # Tier 3: Reasoning Quality
        "judgement_accuracy": judgement_accuracy,
        "score_accuracy": score_accuracy,
        "verdict_consistency": verdict_consistency,
        # Details
        "details": {
            "auprc_metrics": auprc_metrics,
            "incident_details": incident_details,
            "judgement_details": judgement_details,
            "score_details": score_details,
            "consistency_details": consistency_details,
        },
        "n_samples": len(results),
        "n_with_gt": len(y_true),
        "n_structured": len(structured_results),
    }
