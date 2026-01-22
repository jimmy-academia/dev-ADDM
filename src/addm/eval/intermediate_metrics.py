"""
Intermediate process evaluation metrics for ADDM.

Evaluates the quality of intermediate reasoning steps, not just final verdicts:
- Stage 1: Evidence Validity (precision + snippet verification)
- Stage 2: L0 Classification Accuracy (for valid evidence)
- Stage 3: Verdict Support Validity (does claimed evidence support verdict?)

Uses "sufficiency mode" - methods can early-stop once threshold is met,
so we evaluate whether claimed evidence actually justifies the verdict.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from addm.eval.metrics import normalize_verdict


# Severity to base points mapping (from V2 policy)
SEVERITY_BASE_POINTS = {
    "mild": 1,
    "moderate": 3,
    "severe": 5,
}

# Modifier point adjustments
MODIFIER_POINTS = {
    "False assurance": 3,
    "Dismissive staff": 3,
}

# Verdict thresholds (from V2 policy)
VERDICT_THRESHOLDS = {
    "Critical Risk": 8,
    "High Risk": 4,
    "Low Risk": 0,
}


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
        - incident_precision: |claimed ∩ GT| / |claimed|
        - snippet_validity: |valid snippets| / |total snippets|
        - samples_with_no_claims: count of samples that claimed no evidence
        - per_sample: detailed breakdown per sample
    """
    total_claimed = 0
    total_matched = 0
    total_snippets = 0
    valid_snippets = 0
    samples_with_no_claims = 0  # Track samples with no claimed evidence
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

        # Track samples with no claims (excluded from precision calculation)
        if not claimed_review_ids:
            samples_with_no_claims += 1
            per_sample.append({
                "business_id": biz_id,
                "claimed_incidents": 0,
                "matched_incidents": 0,
                "incident_precision": None,  # Can't compute precision with no claims
                "snippet_valid": 0,
                "snippet_total": 0,
                "snippet_validity": None,
                "has_claims": False,
            })
            continue

        # Incident matching
        matched = claimed_review_ids & gt_review_ids
        sample_precision = len(matched) / len(claimed_review_ids)

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
            "matched_incidents": len(matched),
            "incident_precision": sample_precision,
            "snippet_valid": sample_snippet_valid,
            "snippet_total": sample_snippet_total,
            "snippet_validity": sample_snippet_validity,
            "has_claims": True,
        })

    return {
        "incident_precision": total_matched / total_claimed if total_claimed > 0 else None,
        "total_claimed": total_claimed,
        "total_matched": total_matched,
        "snippet_validity": valid_snippets / total_snippets if total_snippets > 0 else None,
        "total_snippets": total_snippets,
        "valid_snippets": valid_snippets,
        "samples_with_no_claims": samples_with_no_claims,
        "per_sample": per_sample,
    }


def compute_classification_accuracy(
    results: List[Dict[str, Any]],
    gt_data: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute L0 classification accuracy for valid evidence.

    Stage 2: For evidence items that match GT incidents,
    did the method classify them correctly?

    Args:
        results: Method outputs with parsed.evidences
        gt_data: GT data by business_id with incidents array

    Returns:
        Dict with:
        - severity_accuracy: correct severity classifications / total
        - modifier_accuracy: correct modifier detections / total
        - per_field: breakdown by field type
    """
    severity_correct = 0
    severity_total = 0
    modifier_correct = 0
    modifier_total = 0
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
            field = evidence.get("field", "").upper()
            judgement = evidence.get("judgement", "")

            # Track per-field stats
            if field not in per_field:
                per_field[field] = {"correct": 0, "total": 0}

            # Severity classification
            if field == "INCIDENT_SEVERITY":
                gt_severity = gt_incident.get("severity", "").lower()
                severity_values = list(SEVERITY_BASE_POINTS.keys())
                normalized_judgement = normalize_judgement(judgement, severity_values)
                severity_total += 1
                per_field[field]["total"] += 1
                if normalized_judgement == gt_severity:
                    severity_correct += 1
                    per_field[field]["correct"] += 1

            # Assurance claim (modifier)
            elif field in ("ASSURANCE_CLAIM", "ASSURANCE_OF_SAFETY"):
                gt_modifiers = gt_incident.get("modifiers", [])
                has_false_assurance = "False assurance" in gt_modifiers
                # Normalize: detect various ways of indicating false/no assurance
                assurance_positive = ["true", "yes", "detected", "false assurance", "no assurance"]
                method_detected = normalize_judgement(judgement, assurance_positive) is not None
                modifier_total += 1
                per_field[field]["total"] += 1
                if method_detected == has_false_assurance:
                    modifier_correct += 1
                    per_field[field]["correct"] += 1

            # Staff response (modifier)
            elif field == "STAFF_RESPONSE":
                gt_modifiers = gt_incident.get("modifiers", [])
                has_dismissive = "Dismissive staff" in gt_modifiers
                # Normalize: detect various ways of indicating negative staff response
                staff_negative = ["dismissive", "negative", "bad", "rude", "unhelpful"]
                method_detected = normalize_judgement(judgement, staff_negative) is not None
                modifier_total += 1
                per_field[field]["total"] += 1
                if method_detected == has_dismissive:
                    modifier_correct += 1
                    per_field[field]["correct"] += 1

    # Compute per-field accuracy
    for field_stats in per_field.values():
        field_stats["accuracy"] = (
            field_stats["correct"] / field_stats["total"]
            if field_stats["total"] > 0
            else None
        )

    return {
        "severity_accuracy": severity_correct / severity_total if severity_total > 0 else None,
        "severity_correct": severity_correct,
        "severity_total": severity_total,
        "modifier_accuracy": modifier_correct / modifier_total if modifier_total > 0 else None,
        "modifier_correct": modifier_correct,
        "modifier_total": modifier_total,
        "per_field": per_field,
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
        "incident_precision": evidence_validity.get("incident_precision"),
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
