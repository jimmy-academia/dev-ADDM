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
from sklearn.metrics import average_precision_score, f1_score
from typing import Dict, List, Optional, Any, Tuple

from addm.eval.constants import (
    VERDICT_TO_ORDINAL,
    CLASS_NAMES,
    SEVERITY_BASE_POINTS,
    MODIFIER_POINTS,
    VERDICT_THRESHOLDS,
    get_verdict_to_ordinal,
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
    - Underscore format: "HIGH_RISK" -> "High Risk"
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
        # Normalize for comparison: lowercase and replace underscores with spaces
        normalized_for_cmp = normalized.lower().replace("_", " ")
        for valid in valid_verdicts:
            if valid.lower() == normalized_for_cmp:
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
        return {"ordinal_auprc": None, "n_samples": n_samples}

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
    results["ordinal_auprc"] = np.mean(auprc_values) if auprc_values else None
    results["n_samples"] = n_samples

    return results


def _extract_evidences(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract evidences array from result's parsed field."""
    parsed = result.get("parsed", {})
    if isinstance(parsed, dict):
        return parsed.get("evidences", [])
    return []


def _compute_verdict_from_seed(
    evidences: List[Dict[str, Any]],
    formula_seed: Dict[str, Any],
) -> Tuple[str, float]:
    """Compute verdict from evidences using formula seed's rules.

    This is the policy-aware approach: extract verdict rules from the seed's
    compute section (the VERDICT case operation) and execute them against
    the evidences.

    Args:
        evidences: List of evidence dicts with field and judgement
        formula_seed: Formula Seed with compute section containing VERDICT rules

    Returns:
        Tuple of (computed_verdict, computed_score)
    """
    import re

    compute_ops = formula_seed.get("compute", [])
    extract_info = formula_seed.get("extract", {})

    # Find the outcome field from seed metadata
    outcome_field = extract_info.get("outcome_field", "INCIDENT_SEVERITY")
    none_values = extract_info.get("none_values", ["none", "n/a"])

    # Build namespace by executing compute operations in order
    namespace: Dict[str, Any] = {}

    # Helper to get field value from evidence (case-insensitive)
    def get_evidence_field(ev: Dict[str, Any], field_name: str) -> Optional[str]:
        for key in [field_name, field_name.lower(), field_name.upper()]:
            if key in ev:
                return ev[key]
        # Try field attribute
        if ev.get("field", "").upper() == field_name.upper():
            return ev.get("judgement")
        return None

    # Helper to check if value is "none" (case-insensitive)
    def is_none_value(value: Any) -> bool:
        if value is None:
            return True
        value_str = str(value).strip().lower()
        return value_str in [nv.lower() for nv in none_values]

    # Helper to fuzzy match enum values
    def fuzzy_match(actual: str, expected: str) -> bool:
        if actual is None or expected is None:
            return actual == expected
        a, e = str(actual).lower().strip(), str(expected).lower().strip()
        return a == e or e in a or a in e

    # Execute each compute operation
    for op_def in compute_ops:
        name = op_def.get("name", "")
        op = op_def.get("op") or op_def.get("operation", "")

        if op == "count":
            # Count evidences matching conditions
            where = op_def.get("where", {})
            count = 0
            for ev in evidences:
                matches = True
                for field_name, expected in where.items():
                    if field_name in ("field", "equals", "not_equals", "and", "or"):
                        continue  # Skip meta keys
                    actual = get_evidence_field(ev, field_name)
                    if actual is None:
                        matches = False
                        break
                    if isinstance(expected, list):
                        if not any(fuzzy_match(actual, e) for e in expected):
                            matches = False
                            break
                    elif not fuzzy_match(actual, expected):
                        matches = False
                        break
                if matches:
                    count += 1
            namespace[name] = count

        elif op == "sum":
            # Sum expression over evidences
            expr = op_def.get("expr", "1")
            where = op_def.get("where", {})
            total = 0.0

            # Check if SQL CASE expression
            is_sql_case = "CASE" in expr.upper()

            for ev in evidences:
                # Check where condition
                if where:
                    matches = True
                    for field_name, expected in where.items():
                        if field_name in ("field", "equals", "not_equals", "and", "or"):
                            continue
                        actual = get_evidence_field(ev, field_name)
                        if actual is None:
                            matches = False
                            break
                        if isinstance(expected, list):
                            if not any(fuzzy_match(actual, e) for e in expected):
                                matches = False
                                break
                        elif not fuzzy_match(actual, expected):
                            matches = False
                            break
                    if not matches:
                        continue

                if is_sql_case:
                    # Parse CASE WHEN ... THEN ... END
                    case_pattern = r"WHEN\s+(\w+)\s*=\s*['\"]([^'\"]+)['\"]\s+THEN\s+(\d+(?:\.\d+)?)"
                    matches_list = re.findall(case_pattern, expr, re.IGNORECASE)
                    value_added = False
                    for field_name, expected_val, then_val in matches_list:
                        actual = get_evidence_field(ev, field_name)
                        if actual and fuzzy_match(actual, expected_val):
                            total += float(then_val)
                            value_added = True
                            break
                    if not value_added:
                        # Check for ELSE clause
                        else_match = re.search(r"ELSE\s+(\d+(?:\.\d+)?)", expr, re.IGNORECASE)
                        if else_match:
                            total += float(else_match.group(1))

            namespace[name] = total

        elif op == "expr":
            # Evaluate mathematical expression
            expr = op_def.get("expr", "0")
            try:
                safe_builtins = {"min": min, "max": max, "abs": abs, "round": round}
                namespace[name] = eval(expr, {"__builtins__": safe_builtins}, namespace)
            except Exception:
                namespace[name] = 0

        elif op == "case":
            # Apply case rules
            source = op_def.get("source", "")
            rules = op_def.get("rules", [])
            source_value = namespace.get(source, 0)

            result = None
            for rule in rules:
                if "else" in rule:
                    result = rule["else"]
                    break

                when = rule.get("when", "")
                then = rule.get("then", "")

                # Try to evaluate as expression
                try:
                    context = {**namespace, source: source_value}
                    match = re.match(r"(\w+)\s*([<>=!]+)\s*(\d+(?:\.\d+)?)", when)
                    if match:
                        var, op_str, threshold = match.groups()
                        actual_val = context.get(var, 0)
                        threshold_val = float(threshold)
                        if op_str == ">=" and actual_val >= threshold_val:
                            result = then
                            break
                        elif op_str == "<=" and actual_val <= threshold_val:
                            result = then
                            break
                        elif op_str == ">" and actual_val > threshold_val:
                            result = then
                            break
                        elif op_str == "<" and actual_val < threshold_val:
                            result = then
                            break
                        elif op_str in ("==", "=") and actual_val == threshold_val:
                            result = then
                            break
                except Exception:
                    pass

            namespace[name] = result

    # Get final verdict and score
    verdict = namespace.get("VERDICT", "Low Risk")
    score = namespace.get("SCORE", namespace.get("N_INCIDENTS", 0))

    if verdict is None:
        verdict = "Low Risk"

    return str(verdict), float(score) if isinstance(score, (int, float)) else 0.0


def _legacy_g1_verdict(evidences: List[Dict[str, Any]]) -> Tuple[str, float]:
    """Legacy G1-specific verdict computation (fallback).

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


def _compute_verdict_from_evidences(
    evidences: List[Dict[str, Any]],
    formula_seed: Optional[Dict[str, Any]] = None,
) -> Tuple[str, float]:
    """Compute verdict from evidences using formula seed's rules.

    If formula_seed is provided, uses its compute section to calculate
    the verdict (policy-aware). Otherwise, falls back to legacy G1 scoring.

    Args:
        evidences: List of evidence dicts with field and judgement
        formula_seed: Optional Formula Seed with compute/verdict rules

    Returns:
        Tuple of (computed_verdict, computed_score)
    """
    if formula_seed and formula_seed.get("compute"):
        return _compute_verdict_from_seed(evidences, formula_seed)
    else:
        return _legacy_g1_verdict(evidences)


def extract_verdict_from_rule(triggered_rule: str) -> Optional[str]:
    """Extract verdict label from triggered_rule string.

    OUTPUT-BASED CONSISTENCY: Instead of re-computing verdict from evidences,
    we check if method's triggered_rule leads to its claimed verdict.

    Examples:
        "SCORE >= 5 → Critical Risk" → "Critical Risk"
        "N_NEGATIVE >= 1 → Not Recommended" → "Not Recommended"
        "else → Low Risk" → "Low Risk"
        "N_SEVERE >= 1 (actual: N_SEVERE=0) → Critical Risk" → "Critical Risk"

    Args:
        triggered_rule: Rule string from method output

    Returns:
        Verdict label extracted from rule, or None if not parseable
    """
    if not triggered_rule:
        return None

    # Try arrow separator (→ or ->)
    for sep in ["→", "->"]:
        if sep in triggered_rule:
            return triggered_rule.split(sep)[-1].strip()

    return None


def compute_score_accuracy(
    results: List[Dict[str, Any]],
    gt_data: Dict[str, Dict[str, Any]],
    policy_id: Optional[str] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Compute score accuracy - does method's computed score match GT?

    Only evaluated for policies with score systems (V2/V3).
    Returns None for V1/V2 (condition-based) policies.

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
            return None, {"skipped": "Policy does not use scoring system (V1/V2)"}

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
    formula_seed: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """OUTPUT-BASED verdict consistency - checks triggered_rule → verdict.

    This is an OUTPUT-BASED metric: we check if the method's claimed triggered_rule
    leads to its claimed verdict. This validates internal consistency without
    re-computing verdicts (which may use different logic than the method).

    For each result:
    1. Extract verdict from method's triggered_rule (e.g., "SCORE >= 5 → Critical Risk")
    2. Compare extracted verdict to method's claimed verdict
    3. Consistent if they match

    Note: We no longer re-compute verdict from evidences. The method's internal
    logic (as expressed in triggered_rule) is the source of truth for consistency.

    Args:
        results: Method outputs with parsed.justification.triggered_rule
        policy_id: Optional policy ID for verdict normalization
        formula_seed: Unused (kept for API compatibility)

    Returns:
        Tuple of (consistency_rate, details_dict)
    """
    consistent = 0
    total = 0
    skipped_no_rule = 0
    per_sample = []

    for result in results:
        if "error" in result:
            continue

        method_verdict = result.get("verdict")
        if not method_verdict:
            continue

        # Get triggered_rule from justification
        parsed = result.get("parsed", {})
        justification = parsed.get("justification", {}) if isinstance(parsed, dict) else {}
        triggered_rule = justification.get("triggered_rule", "") if isinstance(justification, dict) else ""

        if not triggered_rule:
            skipped_no_rule += 1
            continue

        # Extract verdict from triggered_rule string
        # e.g., "SCORE >= 5 → Critical Risk" → "Critical Risk"
        rule_verdict = extract_verdict_from_rule(triggered_rule)

        if not rule_verdict:
            skipped_no_rule += 1
            continue

        total += 1

        # Consistency: rule_verdict should match method_verdict (normalized)
        method_norm = normalize_verdict(method_verdict, policy_id)
        rule_norm = normalize_verdict(rule_verdict, policy_id)
        is_consistent = method_norm == rule_norm

        if is_consistent:
            consistent += 1

        per_sample.append({
            "business_id": result.get("business_id"),
            "method_verdict": method_verdict,
            "method_verdict_normalized": method_norm,
            "triggered_rule": triggered_rule,
            "rule_verdict": rule_verdict,
            "rule_verdict_normalized": rule_norm,
            "consistent": is_consistent,
        })

    if total == 0:
        return None, {"skipped_no_rule": skipped_no_rule, "total": 0}

    return consistent / total, {
        "consistent": consistent,
        "total": total,
        "skipped_no_rule": skipped_no_rule,
        "per_sample": per_sample,
    }


def compute_evaluation_metrics(
    results: List[Dict[str, Any]],
    gt_data: Dict[str, Dict[str, Any]],
    gt_verdicts: Dict[str, str],
    method: str = "direct",
    reviews_data: Optional[Dict[str, Dict[str, str]]] = None,
    policy_id: Optional[str] = None,
    formula_seed: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute all evaluation metrics (7 metrics total).

    Returns separate, interpretable metrics without weighted composites.

    Tier 1: Final Quality (ranking + classification)
    - AUPRC: Ordinal ranking quality (>=High, >=Critical). Measures if high-risk
      items are RANKED above low-risk items. Can be high even with wrong predictions
      if the relative ordering is correct.
    - Macro F1: Classification quality across all classes. Measures if predictions
      are CORRECT. Averages F1 across classes, resistant to class imbalance.

    Tier 2: Evidence Quality
    - Evidence Precision: % claimed incidents that exist in GT
    - Evidence Recall: % GT incidents found by method
    - Snippet Validity: % snippets that match actual review text

    Tier 3: Reasoning Quality
    - Judgement Accuracy: % correct field values for matched incidents
    - Verdict Consistency: Does (evidence + triggered_rule) → verdict?

    Args:
        results: List of method outputs with parsed.evidences
        gt_data: {business_id: GT with incidents}
        gt_verdicts: {business_id: verdict}
        method: Method name ("direct", "amos", etc.)
        reviews_data: Optional {business_id: {review_id: text}} for snippet validation
        policy_id: Optional policy ID for verdict normalization
        formula_seed: Optional Formula Seed for policy-aware verdict computation

    Returns:
        {
            "auprc": float,  # 0.0-1.0, ordinal ranking quality (are risky items ranked higher?)
            "macro_f1": float,  # 0.0-1.0, classification quality (are predictions correct?)
            "evidence_precision": float or None,  # 0.0-1.0, % claimed that exist in GT
            "evidence_recall": float or None,  # 0.0-1.0, % GT incidents found
            "snippet_validity": float or None,  # 0.0-1.0, % snippets that match source
            "judgement_accuracy": float or None,  # 0.0-1.0, field correctness
            "verdict_consistency": float or None,  # 0.0-1.0, evidence+rule→verdict
            "details": {
                "auprc_metrics": {...},
                "incident_details": {...},
                "judgement_details": {...},
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

    # 1. Compute AUPRC and Macro F1
    y_true = []
    y_pred = []
    y_scores = []

    # Get policy-aware verdict mapping (G1 uses Risk, G2 uses Recommended, etc.)
    verdict_mapping = get_verdict_to_ordinal(policy_id) if policy_id else VERDICT_TO_ORDINAL

    for result in results:
        if "error" in result:
            continue

        biz_id = result.get("business_id")
        gt_verdict = gt_verdicts.get(biz_id)

        if not gt_verdict:
            continue

        # Normalize GT verdict for lookup (defensive - should already be normalized)
        gt_verdict_norm = normalize_verdict(gt_verdict, policy_id)
        gt_ord = verdict_mapping.get(gt_verdict_norm or gt_verdict)
        if gt_ord is None:
            continue

        # Normalize predicted verdict for lookup (handles "HIGH_RISK" -> "High Risk")
        pred_verdict = normalize_verdict(result.get("verdict"), policy_id)
        pred_ord = verdict_mapping.get(pred_verdict)

        # Get risk score (continuous), fallback to verdict ordinal * 5
        risk_score = result.get("risk_score")
        if risk_score is None:
            risk_score = pred_ord * 5.0 if pred_ord is not None else None

        if risk_score is not None and pred_ord is not None:
            y_true.append(gt_ord)
            y_pred.append(pred_ord)
            y_scores.append(risk_score)

    auprc_metrics = {}
    auprc = None
    macro_f1 = 0.0
    if len(y_true) >= 2:
        auprc_metrics = compute_ordinal_auprc(np.array(y_true), np.array(y_scores))
        auprc = auprc_metrics.get("ordinal_auprc")  # None if no class variation
        # Macro F1: average F1 across all classes (handles skewed data)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

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

    # 4. Compute Enhanced Verdict Consistency (with triggered_rule check)
    verdict_consistency, consistency_details = compute_verdict_consistency_enhanced(
        results, policy_id, formula_seed
    )

    return {
        # Tier 1: Final Quality (ranking + classification)
        "auprc": auprc,
        "macro_f1": macro_f1,
        # Tier 2: Evidence Quality
        "evidence_precision": evidence_precision,
        "evidence_recall": evidence_recall,
        "snippet_validity": snippet_validity,
        # Tier 3: Reasoning Quality
        "judgement_accuracy": judgement_accuracy,
        "verdict_consistency": verdict_consistency,
        # Details
        "details": {
            "auprc_metrics": auprc_metrics,
            "incident_details": incident_details,
            "judgement_details": judgement_details,
            "consistency_details": consistency_details,
        },
        "n_samples": len(results),
        "n_with_gt": len(y_true),
        "n_structured": len(structured_results),
    }
