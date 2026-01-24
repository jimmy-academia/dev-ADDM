"""Unified 3-score evaluation for ADDM methods.

DEPRECATED: This module is being replaced by the simplified metrics in metrics.py.
Use compute_evaluation_metrics() from addm.eval.metrics for new code.

Legacy functions maintained for backward compatibility:
- compute_unified_metrics() - old 3-score system
- compute_process_score() - weighted composite (removed in new system)
- compute_consistency_score() - now just verdict_consistency
- normalize_amos_output() - still needed for AMOS output normalization
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from addm.eval.metrics import (
    compute_ordinal_auprc,
    normalize_verdict,
    normalize_judgement,
    extract_verdict_from_rule,
)
from addm.eval.constants import (
    VERDICT_TO_ORDINAL,
    SEVERITY_BASE_POINTS,
    MODIFIER_POINTS,
    VERDICT_THRESHOLDS,
)


# DEPRECATED: Process Score weights (kept for backward compatibility)
# New system uses separate metrics without weighting
PROCESS_WEIGHTS = {
    "incident_precision": 0.35,   # Finding correct incidents (most important)
    "severity_accuracy": 0.30,    # Correct severity classification
    "modifier_accuracy": 0.15,    # Correct modifier detection
    "verdict_support_rate": 0.15, # Evidence supports verdict
    "snippet_validity": 0.05,     # Valid quotes (AMOS doesn't have snippets)
}


def normalize_amos_output(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert AMOS output to standard evidence schema.

    AMOS outputs:
    {
        "VERDICT": "Low Risk",
        "SCORE": 5,
        "_extractions": [
            {"review_id": "...", "is_relevant": true, "INCIDENT_SEVERITY": "moderate", ...}
        ]
    }

    Convert to standard schema:
    {
        "verdict": "Low Risk",
        "evidences": [
            {"evidence_id": "E1", "review_id": "...", "field": "INCIDENT_SEVERITY", "judgement": "moderate"}
        ],
        "justification": {"scoring_trace": {"total_score": 5}}
    }

    Args:
        result: AMOS method result dict

    Returns:
        Normalized result with standard evidence schema
    """
    # Try to parse response as JSON if it's a string
    output_data = {}
    response = result.get("response", "")
    if isinstance(response, str) and response.strip():
        try:
            output_data = json.loads(response)
        except json.JSONDecodeError:
            pass

    # Check for AMOS-specific fields
    extractions = output_data.get("_extractions", [])
    namespace = output_data.get("_namespace", {})

    if not extractions and not output_data.get("VERDICT"):
        # Not an AMOS output, return original
        return result

    # Build evidences from extractions - only include actual incidents (severity != "none")
    evidences = []
    evidence_ids = []
    for i, extraction in enumerate(extractions):
        if not extraction.get("is_relevant", False):
            continue

        # Only include extractions with actual incidents (not "none" severity)
        severity = extraction.get("INCIDENT_SEVERITY")
        if not severity or severity.lower() == "none":
            continue  # Skip non-incidents

        review_id = extraction.get("review_id", f"unknown_{i}")
        evidence_id = f"E{i + 1}"
        evidence_ids.append(evidence_id)

        # Extract INCIDENT_SEVERITY field
        evidences.append({
            "evidence_id": evidence_id,
            "review_id": review_id,
            "field": "INCIDENT_SEVERITY",
            "judgement": severity.lower() if isinstance(severity, str) else str(severity),
        })

        # Extract ASSURANCE_CLAIM field (boolean)
        assurance = extraction.get("ASSURANCE_CLAIM")
        if assurance is not None:
            evidences.append({
                "evidence_id": f"{evidence_id}_AC",
                "review_id": review_id,
                "field": "ASSURANCE_CLAIM",
                "judgement": "true" if assurance else "false",
            })

        # Extract STAFF_RESPONSE field
        staff = extraction.get("STAFF_RESPONSE")
        if staff:
            evidences.append({
                "evidence_id": f"{evidence_id}_SR",
                "review_id": review_id,
                "field": "STAFF_RESPONSE",
                "judgement": staff.lower() if isinstance(staff, str) else str(staff),
            })

    # Build normalized parsed structure
    score = (
        output_data.get("FINAL_RISK_SCORE") or
        output_data.get("RISK_SCORE") or
        output_data.get("SCORE") or
        namespace.get("FINAL_RISK_SCORE") or
        namespace.get("SCORE") or
        0
    )

    normalized_parsed = {
        "verdict": output_data.get("VERDICT") or result.get("verdict"),
        "evidences": evidences,
        "justification": {
            "direct_evidence": evidence_ids,
            "scoring_trace": {
                "total_score": score,
            },
        },
    }

    # Return a copy with normalized parsed field
    normalized = result.copy()
    normalized["parsed"] = normalized_parsed
    normalized["_amos_extractions"] = extractions  # Keep original for debugging

    return normalized


def _extract_evidences(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract evidences array from result's parsed field."""
    parsed = result.get("parsed", {})
    if isinstance(parsed, dict):
        return parsed.get("evidences", [])
    return []




def compute_process_score(
    intermediate_metrics: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Compute consolidated Process Score (0-100).

    Uses weighted average of intermediate metrics:
    - incident_precision: 35% (finding correct incidents)
    - severity_accuracy: 30% (correct severity classification)
    - modifier_accuracy: 15% (correct modifier detection)
    - verdict_support_rate: 15% (evidence supports verdict)
    - snippet_validity: 5% (valid quotes)

    Args:
        intermediate_metrics: Dict from compute_intermediate_metrics()

    Returns:
        Tuple of (process_score, component_scores)
    """
    if not intermediate_metrics or "error" in intermediate_metrics:
        return 0.0, {"error": "No intermediate metrics available"}

    summary = intermediate_metrics.get("summary", {})
    evidence_validity = intermediate_metrics.get("evidence_validity", {})
    classification = intermediate_metrics.get("classification_accuracy", {})
    verdict_support = intermediate_metrics.get("verdict_support", {})

    # Extract component values (default to 0 if missing)
    component_scores = {}

    # Incident precision from evidence validity
    incident_precision = evidence_validity.get("incident_precision")
    if incident_precision is not None:
        component_scores["incident_precision"] = incident_precision
    else:
        component_scores["incident_precision"] = 0.0

    # Severity accuracy from classification
    severity_accuracy = classification.get("severity_accuracy")
    if severity_accuracy is not None:
        component_scores["severity_accuracy"] = severity_accuracy
    else:
        component_scores["severity_accuracy"] = 0.0

    # Modifier accuracy from classification
    modifier_accuracy = classification.get("modifier_accuracy")
    if modifier_accuracy is not None:
        component_scores["modifier_accuracy"] = modifier_accuracy
    else:
        component_scores["modifier_accuracy"] = 0.0

    # Verdict support rate
    verdict_support_rate = verdict_support.get("verdict_support_rate")
    if verdict_support_rate is not None:
        component_scores["verdict_support_rate"] = verdict_support_rate
    else:
        component_scores["verdict_support_rate"] = 0.0

    # Snippet validity from evidence validity
    snippet_validity = evidence_validity.get("snippet_validity")
    if snippet_validity is not None:
        component_scores["snippet_validity"] = snippet_validity
    else:
        # For methods without snippets (like AMOS), don't penalize
        # Use weighted average of other components instead
        component_scores["snippet_validity"] = None

    # Compute weighted average
    weighted_sum = 0.0
    total_weight = 0.0

    for metric, weight in PROCESS_WEIGHTS.items():
        value = component_scores.get(metric)
        if value is not None:
            weighted_sum += value * weight
            total_weight += weight

    # Normalize by actual weight used (in case snippet_validity is None)
    if total_weight > 0:
        process_score = (weighted_sum / total_weight) * 100
    else:
        process_score = 0.0

    # Add final score to component breakdown
    component_scores["weighted_sum"] = weighted_sum
    component_scores["total_weight"] = total_weight

    return process_score, component_scores


def compute_consistency_score(
    results: List[Dict[str, Any]],
    method: str = "direct",
    policy_id: Optional[str] = None,
    formula_seed: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """OUTPUT-BASED verdict consistency score (0-100).

    For each result:
    1. Extract verdict from method's triggered_rule (e.g., "SCORE >= 5 → Critical Risk")
    2. Compare to method's claimed verdict
    3. Consistent if they match

    This is OUTPUT-BASED: we check the method's internal consistency (does its
    triggered_rule lead to its claimed verdict?) rather than re-computing verdict
    from evidences.

    Samples with no triggered_rule are excluded from the denominator.

    Args:
        results: List of method outputs
        method: Method name ("direct", "amos", etc.)
        policy_id: Optional policy ID for verdict normalization
        formula_seed: Unused (kept for API compatibility)

    Returns:
        Tuple of (consistency_score, details)
    """
    total = 0
    consistent = 0
    skipped_no_rule = 0
    per_sample = []

    for result in results:
        if "error" in result:
            continue

        biz_id = result.get("business_id", "")
        method_verdict = result.get("verdict")

        if not method_verdict:
            continue

        # Normalize AMOS output if needed
        if method == "amos":
            result = normalize_amos_output(result)

        # Get triggered_rule from justification
        parsed = result.get("parsed", {})
        justification = parsed.get("justification", {}) if isinstance(parsed, dict) else {}
        triggered_rule = justification.get("triggered_rule", "") if isinstance(justification, dict) else ""

        if not triggered_rule:
            skipped_no_rule += 1
            continue

        # Extract verdict from triggered_rule string
        rule_verdict = extract_verdict_from_rule(triggered_rule)

        if not rule_verdict:
            skipped_no_rule += 1
            continue

        total += 1

        # Normalize for comparison (handles quotes, case, whitespace)
        method_norm = normalize_verdict(method_verdict, policy_id)
        rule_norm = normalize_verdict(rule_verdict, policy_id)
        is_consistent = method_norm == rule_norm

        if is_consistent:
            consistent += 1

        per_sample.append({
            "business_id": biz_id,
            "method_verdict": method_verdict,
            "method_verdict_normalized": method_norm,
            "triggered_rule": triggered_rule,
            "rule_verdict": rule_verdict,
            "rule_verdict_normalized": rule_norm,
            "is_consistent": is_consistent,
        })

    consistency_score = (consistent / total * 100) if total > 0 else 0.0

    return consistency_score, {
        "consistent": consistent,
        "total": total,
        "skipped_no_rule": skipped_no_rule,
        "per_sample": per_sample,
    }


def compute_false_positive_rate(
    results: List[Dict[str, Any]],
    gt_verdicts: Dict[str, str],
    method: str = "direct",
    policy_id: Optional[str] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Compute false positive rate on negative-GT samples.

    For samples where GT = lowest verdict (e.g., "Low Risk"):
    - Check if method claimed any evidences
    - False positive = claimed evidence when GT says none exist

    This measures how often the method hallucinates incidents that
    don't exist according to ground truth.

    Args:
        results: List of method outputs
        gt_verdicts: {business_id: verdict} ground truth
        method: Method name ("direct", "amos", etc.)
        policy_id: Optional policy ID for verdict normalization

    Returns:
        Tuple of (fp_rate, details)
    """
    from addm.eval.metrics import get_policy_verdicts, CLASS_NAMES

    # Get lowest verdict for this policy (e.g., "Low Risk")
    verdicts = get_policy_verdicts(policy_id) if policy_id else CLASS_NAMES
    lowest_verdict = verdicts[0]  # First is lowest

    negative_samples = 0
    false_positives = 0
    per_sample = []

    for result in results:
        if "error" in result:
            continue

        biz_id = result.get("business_id", "")
        gt_verdict = gt_verdicts.get(biz_id)

        if not gt_verdict:
            continue

        # Only evaluate negative GT samples (lowest verdict = no incidents)
        if normalize_verdict(gt_verdict, policy_id) != lowest_verdict:
            continue

        negative_samples += 1

        # Normalize AMOS output if needed
        if method == "amos":
            result = normalize_amos_output(result)

        evidences = _extract_evidences(result)

        # False positive = claimed evidence when GT is negative
        is_fp = len(evidences) > 0
        if is_fp:
            false_positives += 1

        per_sample.append({
            "business_id": biz_id,
            "is_false_positive": is_fp,
            "n_evidences_claimed": len(evidences),
        })

    fp_rate = false_positives / negative_samples if negative_samples > 0 else 0.0

    return fp_rate, {
        "false_positives": false_positives,
        "negative_samples": negative_samples,
        "per_sample": per_sample,
    }


def compute_unified_metrics(
    results: List[Dict[str, Any]],
    gt_data: Dict[str, Dict[str, Any]],
    gt_verdicts: Dict[str, str],
    method: str = "direct",
    reviews_data: Optional[Dict[str, Dict[str, str]]] = None,
    policy_id: Optional[str] = None,
    formula_seed: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Main entry point - returns all 3 scores.

    Args:
        results: List of method outputs
        gt_data: {business_id: GT with incidents}
        gt_verdicts: {business_id: verdict}
        method: Method name ("direct", "amos", etc.)
        reviews_data: Optional {business_id: {review_id: text}} for snippet validation
        policy_id: Optional policy ID for verdict normalization
        formula_seed: Optional Formula Seed for policy-aware verdict computation

    Returns:
        Dict with:
        - auprc: Ordinal AUPRC (0.0-1.0)
        - process_score: Process quality score (0-100)
        - consistency_score: Verdict consistency score (0-100)
        - components: Detailed breakdown of each score
    """
    # Normalize AMOS outputs if needed
    if method == "amos":
        results = [normalize_amos_output(r) for r in results]

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

        # Normalize GT verdict for lookup (defensive - should already be normalized)
        gt_verdict_norm = normalize_verdict(gt_verdict, policy_id)
        gt_ord = VERDICT_TO_ORDINAL.get(gt_verdict_norm or gt_verdict)
        if gt_ord is None:
            continue

        # Get risk score (continuous), fallback to verdict ordinal * 5
        risk_score = result.get("risk_score")
        if risk_score is None:
            # Normalize predicted verdict for lookup (handles "HIGH_RISK" -> "High Risk")
            pred_verdict = normalize_verdict(result.get("verdict"), policy_id)
            pred_ord = VERDICT_TO_ORDINAL.get(pred_verdict)
            risk_score = pred_ord * 5.0 if pred_ord is not None else None

        if risk_score is not None:
            y_true.append(gt_ord)
            y_scores.append(risk_score)

    auprc_metrics = {}
    if len(y_true) >= 2:
        auprc_metrics = compute_ordinal_auprc(np.array(y_true), np.array(y_scores))

    auprc = auprc_metrics.get("ordinal_auprc")  # None if no class variation

    # 2. Compute Process Score (requires intermediate_metrics)
    # First compute intermediate metrics
    from addm.eval.intermediate_metrics import compute_intermediate_metrics

    # Filter to structured results
    structured_results = [
        r for r in results
        if r.get("parsed") and isinstance(r.get("parsed"), dict)
        and "evidences" in r.get("parsed", {})
    ]

    if structured_results and gt_data:
        intermediate_metrics = compute_intermediate_metrics(
            structured_results, gt_data, reviews_data
        )
        process_score, process_components = compute_process_score(intermediate_metrics)
    else:
        intermediate_metrics = {"error": "No structured output or GT data"}
        process_score = 0.0
        process_components = {"error": "No structured output available"}

    # 3. Compute Consistency Score
    consistency_score, consistency_details = compute_consistency_score(results, method, policy_id, formula_seed)

    # 4. Compute False Positive Rate
    fp_rate, fp_details = compute_false_positive_rate(results, gt_verdicts, method, policy_id)

    return {
        "auprc": auprc,
        "auprc_metrics": auprc_metrics,
        "process_score": process_score,
        "process_components": process_components,
        "consistency_score": consistency_score,
        "consistency_details": consistency_details,
        "false_positive_rate": fp_rate,
        "fp_details": fp_details,
        "intermediate_metrics": intermediate_metrics,
        "n_samples": len(results),
        "n_structured": len(structured_results),
        "n_with_gt": len(y_true),
    }
