"""
Intermediate process evaluation metrics for ADDM.

Evaluates the quality of intermediate reasoning steps, not just final verdicts:
- Stage 1: Evidence Validity (precision + snippet verification)
- Stage 2: Judgement Accuracy (policy-agnostic field correctness)
- Stage 3: Verdict Support Validity (does claimed evidence support verdict?)

Uses "sufficiency mode" - methods can early-stop once threshold is met,
so we evaluate whether claimed evidence actually justifies the verdict.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from addm.eval.metrics import normalize_verdict, normalize_judgement
from addm.eval.constants import (
    SEVERITY_BASE_POINTS,
    MODIFIER_POINTS,
    VERDICT_THRESHOLDS,
)


def _extract_evidences(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract evidences array from result's parsed field."""
    parsed = result.get("parsed", {})
    if isinstance(parsed, dict):
        return parsed.get("evidences", [])
    return []


def _extract_justification(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract justification from result's parsed field."""
    parsed = result.get("parsed", {})
    if isinstance(parsed, dict):
        return parsed.get("justification", {})
    return {}


def _get_reviews_from_context(context_str: str) -> Dict[str, str]:
    """Extract review_id -> text mapping from context JSON.

    Args:
        context_str: JSON string of restaurant data

    Returns:
        Dict mapping review_id to review text
    """
    try:
        data = json.loads(context_str)
        reviews = data.get("reviews", [])
        return {
            r.get("review_id", ""): r.get("text", "")
            for r in reviews
            if r.get("review_id")
        }
    except (json.JSONDecodeError, TypeError):
        return {}


def compute_evidence_validity(
    results: List[Dict[str, Any]],
    gt_data: Dict[str, Dict[str, Any]],
    reviews_data: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Compute evidence validity metrics.

    Stage 1: Are the claimed evidence items actually valid?

    Samples with no claims are excluded from precision calculation
    (can't evaluate precision if nothing was claimed).

    Args:
        results: Method outputs with parsed.evidences
        gt_data: GT data by business_id with incidents array
        reviews_data: Optional dict mapping business_id -> {review_id: text}
                      If not provided, snippets are not validated

    Returns:
        Dict with:
        - evidence_precision: |claimed ∩ GT| / |claimed|
        - evidence_recall: |claimed ∩ GT| / |GT|
        - snippet_validity: |valid snippets| / |total snippets|
        - samples_with_no_claims: count of samples that claimed no evidence
        - per_sample: detailed breakdown per sample
    """
    total_claimed = 0
    total_matched = 0
    total_gt_evidence = 0  # For recall calculation
    total_snippets = 0
    valid_snippets = 0
    samples_with_no_claims = 0  # Track samples with no claimed evidence
    samples_with_no_gt = 0  # Track samples with no GT incidents
    per_sample = []

    for result in results:
        biz_id = result.get("business_id")
        if not biz_id:
            continue

        gt = gt_data.get(biz_id, {})
        gt_incidents = gt.get("incidents", [])
        gt_review_ids = {inc.get("review_id") for inc in gt_incidents if inc.get("review_id")}

        evidences = _extract_evidences(result)
        claimed_review_ids = {e.get("review_id") for e in evidences if e.get("review_id")}

        # Track GT incidents for recall (even if method claimed nothing)
        n_gt = len(gt_review_ids)
        if n_gt > 0:
            total_gt_evidence += n_gt
        else:
            samples_with_no_gt += 1

        # Track samples with no claims (excluded from precision calculation)
        if not claimed_review_ids:
            samples_with_no_claims += 1
            # Still compute recall for this sample if GT has incidents
            sample_recall = 0.0 if n_gt > 0 else None
            per_sample.append({
                "business_id": biz_id,
                "claimed_incidents": 0,
                "gt_incidents": n_gt,
                "matched_incidents": 0,
                "evidence_precision": None,  # Can't compute precision with no claims
                "evidence_recall": sample_recall,
                "snippet_valid": 0,
                "snippet_total": 0,
                "snippet_validity": None,
                "has_claims": False,
            })
            continue

        # Incident matching
        matched = claimed_review_ids & gt_review_ids
        sample_precision = len(matched) / len(claimed_review_ids)
        sample_recall = len(matched) / n_gt if n_gt > 0 else None

        total_claimed += len(claimed_review_ids)
        total_matched += len(matched)

        # Snippet validation (if reviews_data provided)
        sample_snippet_valid = 0
        sample_snippet_total = 0

        if reviews_data and biz_id in reviews_data:
            reviews = reviews_data[biz_id]
            for evidence in evidences:
                snippet = evidence.get("snippet", "")
                review_id = evidence.get("review_id", "")
                if snippet and review_id:
                    sample_snippet_total += 1
                    review_text = reviews.get(review_id, "")
                    # Check if snippet is substring of review
                    if snippet in review_text:
                        sample_snippet_valid += 1
                    elif review_text:
                        # Try fuzzy match (allow minor whitespace differences)
                        normalized_snippet = " ".join(snippet.split())
                        normalized_review = " ".join(review_text.split())
                        if normalized_snippet in normalized_review:
                            sample_snippet_valid += 1

            total_snippets += sample_snippet_total
            valid_snippets += sample_snippet_valid

        sample_snippet_validity = (
            sample_snippet_valid / sample_snippet_total
            if sample_snippet_total > 0
            else None
        )

        per_sample.append({
            "business_id": biz_id,
            "claimed_incidents": len(claimed_review_ids),
            "gt_incidents": n_gt,
            "matched_incidents": len(matched),
            "evidence_precision": sample_precision,
            "evidence_recall": sample_recall,
            "snippet_valid": sample_snippet_valid,
            "snippet_total": sample_snippet_total,
            "snippet_validity": sample_snippet_validity,
            "has_claims": True,
        })

    return {
        "evidence_precision": total_matched / total_claimed if total_claimed > 0 else None,
        "evidence_recall": total_matched / total_gt_evidence if total_gt_evidence > 0 else None,
        "total_claimed": total_claimed,
        "total_matched": total_matched,
        "total_gt_evidence": total_gt_evidence,
        "snippet_validity": valid_snippets / total_snippets if total_snippets > 0 else None,
        "total_snippets": total_snippets,
        "valid_snippets": valid_snippets,
        "samples_with_no_claims": samples_with_no_claims,
        "samples_with_no_gt": samples_with_no_gt,
        "per_sample": per_sample,
    }


def _fields_equivalent(method_field: str, gt_field: str) -> bool:
    """Check if method field and GT field are semantically equivalent.

    Handles variations like:
    - INCIDENT_SEVERITY ↔ severity
    - DATE_OUTCOME ↔ date_outcome
    - ATTENTIVENESS ↔ attentiveness

    Args:
        method_field: Field name from method output (uppercase)
        gt_field: Field name from GT (may be any case)

    Returns:
        True if fields are equivalent
    """
    # Normalize both to lowercase for comparison
    method_lower = method_field.lower().replace("_", "")
    gt_lower = gt_field.lower().replace("_", "")

    # Exact match after normalization
    if method_lower == gt_lower:
        return True

    # Check if one contains the other
    if method_lower in gt_lower or gt_lower in method_lower:
        return True

    return False


def _values_equivalent(method_value: str, gt_value: str) -> bool:
    """Check if method value and GT value are semantically equivalent.

    Handles case differences and minor variations.

    Args:
        method_value: Value from method output (lowercase expected)
        gt_value: Value from GT (lowercase expected)

    Returns:
        True if values match
    """
    return method_value.lower().strip() == gt_value.lower().strip()


def compute_judgement_accuracy(
    results: List[Dict[str, Any]],
    gt_data: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Compute policy-agnostic judgement accuracy.

    For each matched incident (by review_id):
    1. Get GT's severity_field and severity_value
    2. Find corresponding field in method output
    3. Compare values (case-insensitive, normalized)

    Also checks modifiers if both method and GT have them.

    This is policy-agnostic: works for G1 allergy (severity_field=incident_severity),
    G2 romance (severity_field=date_outcome), G4 server (severity_field=attentiveness), etc.

    Args:
        results: Method outputs with parsed.evidences
        gt_data: GT data by business_id with incidents array

    Returns:
        Tuple of (accuracy, details_dict)
        - accuracy: float 0.0-1.0 or None if no matches
        - details: breakdown with correct, total, per_field stats
    """
    correct = 0
    total = 0
    per_field = {}

    for result in results:
        biz_id = result.get("business_id")
        if not biz_id:
            continue

        gt = gt_data.get(biz_id, {})
        gt_incidents = gt.get("incidents", [])

        # Build GT lookup by review_id
        gt_by_review = {
            inc.get("review_id"): inc
            for inc in gt_incidents
            if inc.get("review_id")
        }

        evidences = _extract_evidences(result)
        for evidence in evidences:
            review_id = evidence.get("review_id")
            if not review_id or review_id not in gt_by_review:
                continue  # Not a valid match

            gt_incident = gt_by_review[review_id]

            # Get GT's severity field and value (policy-agnostic)
            gt_severity_field = gt_incident.get("severity_field", "").upper()
            gt_severity_value = gt_incident.get("severity_value", "").lower()

            # Fallback: if new fields not present, use legacy fields
            if not gt_severity_field:
                gt_severity_field = "INCIDENT_SEVERITY"
            if not gt_severity_value:
                gt_severity_value = gt_incident.get("severity", "").lower()

            # Get method's field and judgement
            method_field = evidence.get("field", "").upper()
            method_judgement = evidence.get("judgement", "").lower()

            # Track per-field stats
            if method_field not in per_field:
                per_field[method_field] = {"correct": 0, "total": 0}

            # Check if fields match (or are equivalent)
            if method_field == gt_severity_field or _fields_equivalent(method_field, gt_severity_field):
                total += 1
                per_field[method_field]["total"] += 1

                # Normalize method judgement to extract core value
                # For severity, check against known severity values
                severity_values = list(SEVERITY_BASE_POINTS.keys())
                normalized_judgement = normalize_judgement(method_judgement, severity_values)

                # If we got a match from severity normalization, use it
                if normalized_judgement:
                    method_value = normalized_judgement
                else:
                    # Otherwise use raw lowercase value (for non-severity fields)
                    method_value = method_judgement.strip()

                if _values_equivalent(method_value, gt_severity_value):
                    correct += 1
                    per_field[method_field]["correct"] += 1

            # Also check modifiers if present in both
            gt_modifiers = set(gt_incident.get("modifiers", []))
            if method_field in ("ASSURANCE_CLAIM", "ASSURANCE_OF_SAFETY"):
                has_false_assurance = "False assurance" in gt_modifiers
                assurance_positive = ["true", "yes", "detected", "false assurance", "no assurance"]
                method_detected = normalize_judgement(method_judgement, assurance_positive) is not None

                total += 1
                per_field[method_field]["total"] = per_field.get(method_field, {}).get("total", 0) + 1

                if method_detected == has_false_assurance:
                    correct += 1
                    per_field[method_field]["correct"] = per_field.get(method_field, {}).get("correct", 0) + 1

            elif method_field == "STAFF_RESPONSE":
                has_dismissive = "Dismissive staff" in gt_modifiers
                staff_negative = ["dismissive", "negative", "bad", "rude", "unhelpful"]
                method_detected = normalize_judgement(method_judgement, staff_negative) is not None

                total += 1
                per_field[method_field]["total"] = per_field.get(method_field, {}).get("total", 0) + 1

                if method_detected == has_dismissive:
                    correct += 1
                    per_field[method_field]["correct"] = per_field.get(method_field, {}).get("correct", 0) + 1

    # Compute per-field accuracy
    for field_stats in per_field.values():
        field_stats["accuracy"] = (
            field_stats["correct"] / field_stats["total"]
            if field_stats["total"] > 0
            else None
        )

    accuracy = correct / total if total > 0 else None

    return accuracy, {
        "correct": correct,
        "total": total,
        "per_field": per_field,
    }


# Keep old function name as alias for backward compatibility
def compute_classification_accuracy(
    results: List[Dict[str, Any]],
    gt_data: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """DEPRECATED: Use compute_judgement_accuracy() instead.

    This wrapper maintains backward compatibility with code that expects
    the old return format.
    """
    accuracy, details = compute_judgement_accuracy(results, gt_data)

    # Convert to old format for backward compatibility
    return {
        "severity_accuracy": accuracy,
        "severity_correct": details.get("correct", 0),
        "severity_total": details.get("total", 0),
        "modifier_accuracy": None,  # Old function tracked separately
        "modifier_correct": 0,
        "modifier_total": 0,
        "per_field": details.get("per_field", {}),
    }


def compute_verdict_support(
    results: List[Dict[str, Any]],
    gt_data: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute verdict support validity.

    Stage 3: Does the claimed scoring actually support the verdict?

    Uses the method's own claimed evidence and classifications to
    recompute what score/verdict they would produce, and checks
    if it matches the method's claimed verdict.

    Args:
        results: Method outputs with parsed.evidences and parsed.justification
        gt_data: GT data by business_id (used for scoring rules reference)

    Returns:
        Dict with:
        - verdict_support_rate: |correctly supported| / |total|
        - score_consistency: agreement between computed and claimed scores
    """
    total = 0
    supported = 0
    score_matches = 0
    score_total = 0
    per_sample = []

    for result in results:
        biz_id = result.get("business_id")
        method_verdict = result.get("verdict")
        if not biz_id or not method_verdict:
            continue

        evidences = _extract_evidences(result)
        justification = _extract_justification(result)
        direct_evidence = justification.get("direct_evidence", [])
        triggered_rule = justification.get("triggered_rule", "")
        scoring_trace = justification.get("scoring_trace", {})

        # Filter to only direct evidence items, or use all if direct_evidence is empty
        if direct_evidence:
            direct_evidence_items = [
                e for e in evidences
                if e.get("evidence_id") in direct_evidence
            ]
        else:
            # Fallback: use all evidences if direct_evidence not specified
            direct_evidence_items = evidences

        # Valid values for normalization
        severity_values = list(SEVERITY_BASE_POINTS.keys())
        assurance_positive = ["true", "yes", "detected", "false assurance", "no assurance"]
        staff_negative = ["dismissive", "negative", "bad", "rude", "unhelpful"]

        # Recompute score from method's claimed classifications
        computed_score = 0
        for evidence in direct_evidence_items:
            field = evidence.get("field", "").upper()
            judgement = evidence.get("judgement", "")

            if field == "INCIDENT_SEVERITY":
                normalized = normalize_judgement(judgement, severity_values)
                if normalized:
                    computed_score += SEVERITY_BASE_POINTS.get(normalized, 0)
            elif field in ("ASSURANCE_CLAIM", "ASSURANCE_OF_SAFETY"):
                if normalize_judgement(judgement, assurance_positive):
                    computed_score += MODIFIER_POINTS.get("False assurance", 0)
            elif field == "STAFF_RESPONSE":
                if normalize_judgement(judgement, staff_negative):
                    computed_score += MODIFIER_POINTS.get("Dismissive staff", 0)

        # Determine computed verdict from score
        if computed_score >= VERDICT_THRESHOLDS["Critical Risk"]:
            computed_verdict = "Critical Risk"
        elif computed_score >= VERDICT_THRESHOLDS["High Risk"]:
            computed_verdict = "High Risk"
        else:
            computed_verdict = "Low Risk"

        # Check if method's verdict is supported by its own evidence
        total += 1
        # Normalize for comparison (handles quotes, case, whitespace)
        verdict_matches = normalize_verdict(method_verdict) == normalize_verdict(computed_verdict)
        if verdict_matches:
            supported += 1

        # Check score consistency if method provided scoring_trace
        claimed_score = scoring_trace.get("total_score")
        if claimed_score is not None:
            # Convert to float if it's a string (LLM may return as string in JSON)
            try:
                claimed_score_float = float(claimed_score)
            except (ValueError, TypeError):
                claimed_score_float = None

            if claimed_score_float is not None:
                score_total += 1
                if abs(computed_score - claimed_score_float) < 0.01:
                    score_matches += 1

        per_sample.append({
            "business_id": biz_id,
            "method_verdict": method_verdict,
            "computed_verdict": computed_verdict,
            "computed_score": computed_score,
            "claimed_score": claimed_score,
            "verdict_supported": verdict_matches,
            "direct_evidence_count": len(direct_evidence_items),
        })

    return {
        "verdict_support_rate": supported / total if total > 0 else None,
        "total": total,
        "supported": supported,
        "score_consistency": score_matches / score_total if score_total > 0 else None,
        "score_matches": score_matches,
        "score_total": score_total,
        "per_sample": per_sample,
    }


def compute_intermediate_metrics(
    results: List[Dict[str, Any]],
    gt_data: Dict[str, Dict[str, Any]],
    reviews_data: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Compute all intermediate evaluation metrics.

    Main entry point for multi-stage intermediate evaluation.

    Args:
        results: Method outputs with parsed.evidences
        gt_data: {business_id: GT with incidents}
        reviews_data: Optional {business_id: {review_id: text}} for snippet validation

    Returns:
        Dict with:
        - evidence_validity: Stage 1 metrics
        - classification_accuracy: Stage 2 metrics
        - verdict_support: Stage 3 metrics
        - summary: High-level summary metrics
    """
    # Filter to results that have structured output
    structured_results = [
        r for r in results
        if r.get("parsed") and isinstance(r.get("parsed"), dict)
        and "evidences" in r.get("parsed", {})
    ]

    if not structured_results:
        return {
            "error": "No results with structured output found",
            "n_total": len(results),
            "n_structured": 0,
        }

    # Compute each stage
    evidence_validity = compute_evidence_validity(
        structured_results, gt_data, reviews_data
    )
    classification_accuracy = compute_classification_accuracy(
        structured_results, gt_data
    )
    verdict_support = compute_verdict_support(
        structured_results, gt_data
    )

    # Build summary
    summary = {
        "n_total": len(results),
        "n_structured": len(structured_results),
        "evidence_precision": evidence_validity.get("evidence_precision"),
        "severity_accuracy": classification_accuracy.get("severity_accuracy"),
        "verdict_support_rate": verdict_support.get("verdict_support_rate"),
    }

    # Add snippet validity if available
    if evidence_validity.get("snippet_validity") is not None:
        summary["snippet_validity"] = evidence_validity["snippet_validity"]

    return {
        "evidence_validity": evidence_validity,
        "classification_accuracy": classification_accuracy,
        "verdict_support": verdict_support,
        "summary": summary,
    }


def load_gt_with_incidents(
    gt_task_id: str,
    domain: str,
    k: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Load ground truth with full incident data.

    Args:
        gt_task_id: Task/policy ID for GT lookup
        domain: Domain name (e.g., "yelp")
        k: K value for dataset

    Returns:
        Tuple of:
        - gt_data: {business_id: {verdict, incidents, ...}}
        - gt_verdicts: {business_id: verdict} (for backward compatibility)
    """
    from pathlib import Path

    gt_path = Path(f"data/answers/{domain}/{gt_task_id}_K{k}_groundtruth.json")
    if not gt_path.exists():
        # Fallback to old format
        gt_path = Path(f"data/answers/{domain}/{gt_task_id}_groundtruth.json")
        if not gt_path.exists():
            return {}, {}

    with open(gt_path) as f:
        gt_file = json.load(f)

    gt_data = {}
    gt_verdicts = {}

    for biz_id, data in gt_file.get("restaurants", {}).items():
        gt = data.get("ground_truth", {})
        gt_data[biz_id] = {
            "verdict": gt.get("verdict"),
            "score": gt.get("score"),
            "incidents": gt.get("incidents", []),
            "triggered_rule": gt.get("triggered_rule"),
        }
        gt_verdicts[biz_id] = gt.get("verdict", "Unknown")

    return gt_data, gt_verdicts


def build_reviews_data(
    restaurants: List[Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    """Build reviews_data mapping from restaurant list.

    Args:
        restaurants: List of restaurant dicts with reviews

    Returns:
        {business_id: {review_id: text}}
    """
    reviews_data = {}
    for restaurant in restaurants:
        biz_id = restaurant.get("business", {}).get("business_id")
        if not biz_id:
            continue

        reviews = restaurant.get("reviews", [])
        reviews_data[biz_id] = {
            r.get("review_id", ""): r.get("text", "")
            for r in reviews
            if r.get("review_id")
        }

    return reviews_data
