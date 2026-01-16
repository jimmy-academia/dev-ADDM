"""
Ground Truth computation for G4b (Server Quality - Simple + L1.5).

Implements the formula from data/tasks/yelp/G4b_prompt.txt.
Simple formula with L1.5 service aspect grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

# L1.5 Service Pattern Bonus values
SERVICE_PATTERN_BONUS = {
    "consistently_excellent": 2.0,
    "friendly_staff": 1.5,
    "attentive_service": 1.5,
    "knowledgeable_staff": 1.0,
    "mixed_strengths": 0.0,
    "needs_improvement": -1.0,
}


# =============================================================================
# Helpers
# =============================================================================


def get_service_bucket(judgment: Dict[str, Any]) -> str:
    """Determine which service aspect is primary for this review."""
    if judgment.get("attentiveness", "not_mentioned") != "not_mentioned":
        return "attentiveness"
    elif judgment.get("menu_knowledge", "not_mentioned") != "not_mentioned":
        return "knowledge"
    elif judgment.get("friendliness", "not_mentioned") != "not_mentioned" or judgment.get("professionalism", "not_mentioned") != "not_mentioned":
        return "attitude"
    return "other"


def compute_l1_positive_service(judgment: Dict[str, Any]) -> bool:
    attentiveness = judgment.get("attentiveness", "not_mentioned")
    friendliness = judgment.get("friendliness", "not_mentioned")
    menu_knowledge = judgment.get("menu_knowledge", "not_mentioned")

    if attentiveness in ("excellent", "good"):
        return True
    if friendliness in ("warm", "professional"):
        return True
    if menu_knowledge in ("expert", "good"):
        return True
    return False


def compute_l1_negative_service(judgment: Dict[str, Any]) -> bool:
    attentiveness = judgment.get("attentiveness", "not_mentioned")
    friendliness = judgment.get("friendliness", "not_mentioned")
    error_handling = judgment.get("error_handling", "not_mentioned")

    if attentiveness in ("poor", "neglectful"):
        return True
    if friendliness in ("cold", "rude"):
        return True
    if error_handling in ("poor", "denied"):
        return True
    return False


def compute_l15_buckets(service_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = {
        "attentiveness": {"n_reviews": 0, "n_positive": 0, "n_negative": 0},
        "knowledge": {"n_reviews": 0, "n_positive": 0, "n_negative": 0},
        "attitude": {"n_reviews": 0, "n_positive": 0, "n_negative": 0},
    }

    for j in service_judgments:
        bucket = get_service_bucket(j)
        if bucket not in buckets:
            continue

        buckets[bucket]["n_reviews"] += 1

        if compute_l1_positive_service(j):
            buckets[bucket]["n_positive"] += 1
        if compute_l1_negative_service(j):
            buckets[bucket]["n_negative"] += 1

    for key, bucket in buckets.items():
        n_positive = bucket["n_positive"]
        n_negative = bucket["n_negative"]
        bucket["satisfaction_rate"] = n_positive / max(n_positive + n_negative, 1)

    buckets_with_data = [(k, v) for k, v in buckets.items() if v["n_positive"] + v["n_negative"] > 0]

    if buckets_with_data:
        best_aspect, best_bucket = max(buckets_with_data, key=lambda x: x[1]["satisfaction_rate"])
        best_satisfaction_rate = best_bucket["satisfaction_rate"]
        worst_aspect, _ = min(buckets_with_data, key=lambda x: x[1]["satisfaction_rate"])
    else:
        best_aspect = None
        best_satisfaction_rate = 0.0
        worst_aspect = None

    n_aspects_strong = sum(1 for k, v in buckets.items() if v["satisfaction_rate"] >= 0.7 and v["n_reviews"] > 0)

    # Determine service pattern
    if n_aspects_strong >= 3:
        service_pattern = "consistently_excellent"
    elif buckets["attitude"]["satisfaction_rate"] >= 0.8 and buckets["attitude"]["n_reviews"] > 0:
        service_pattern = "friendly_staff"
    elif buckets["attentiveness"]["satisfaction_rate"] >= 0.8 and buckets["attentiveness"]["n_reviews"] > 0:
        service_pattern = "attentive_service"
    elif buckets["knowledge"]["satisfaction_rate"] >= 0.8 and buckets["knowledge"]["n_reviews"] > 0:
        service_pattern = "knowledgeable_staff"
    elif n_aspects_strong >= 1:
        service_pattern = "mixed_strengths"
    else:
        service_pattern = "needs_improvement"

    return {
        "attentiveness": buckets["attentiveness"],
        "knowledge": buckets["knowledge"],
        "attitude": buckets["attitude"],
        "best_aspect": best_aspect,
        "best_aspect_rate": round(best_satisfaction_rate, 3),
        "worst_aspect": worst_aspect,
        "n_aspects_strong": n_aspects_strong,
        "service_pattern": service_pattern,
    }


# =============================================================================
# Main computation
# =============================================================================


def compute_ground_truth(
    judgments: List[Dict[str, Any]],
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
    service_judgments = [j for j in judgments if j.get("is_service_related", False)]
    n_service_reviews = len(service_judgments)

    if n_service_reviews == 0:
        confidence_level = "none"
    elif n_service_reviews <= 2:
        confidence_level = "low"
    elif n_service_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    n_positive = 0
    n_negative = 0

    for j in service_judgments:
        if compute_l1_positive_service(j):
            n_positive += 1
        if compute_l1_negative_service(j):
            n_negative += 1

    # L1.5 Service Buckets
    l15_buckets = compute_l15_buckets(service_judgments)
    service_pattern = l15_buckets["service_pattern"]
    service_pattern_bonus = SERVICE_PATTERN_BONUS.get(service_pattern, 0.0)

    # Formulas
    satisfaction_rate = n_positive / max(n_positive + n_negative, 1)
    positive_score = n_positive * 1.5
    negative_score = n_negative * 1.5

    raw_score = BASE_SCORE + positive_score - negative_score + service_pattern_bonus
    final_score = max(0.0, min(10.0, raw_score))

    if final_score >= 7.5:
        base_verdict_by_score = "Excellent Service"
    elif final_score >= 5.5:
        base_verdict_by_score = "Good Service"
    elif final_score >= 3.5:
        base_verdict_by_score = "Mixed Service"
    else:
        base_verdict_by_score = "Poor Service"

    override_applied = "none"
    verdict = base_verdict_by_score
    strength_note = None
    weakness_note = None

    if service_pattern == "consistently_excellent" and verdict in ("Mixed Service", "Poor Service"):
        override_applied = "consistent_min_good"
        verdict = "Good Service"
    elif service_pattern == "needs_improvement" and n_negative >= 3:
        override_applied = "needs_improvement_with_negative"
        verdict = "Poor Service"

    # Strength/weakness notes
    if service_pattern == "friendly_staff":
        strength_note = "Staff are notably friendly"
    elif service_pattern == "attentive_service":
        strength_note = "Service is very attentive"
    elif service_pattern == "knowledgeable_staff":
        strength_note = "Staff have good menu knowledge"

    if l15_buckets["worst_aspect"] and buckets_rate_below_threshold(l15_buckets, 0.5):
        weakness_note = f"Could improve on {l15_buckets['worst_aspect']}"

    result = {
        "L1_5_service_buckets": l15_buckets,
        "N_SERVICE_REVIEWS": n_service_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_POSITIVE": n_positive,
        "N_NEGATIVE": n_negative,
        "SATISFACTION_RATE": round(satisfaction_rate, 3),
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "SERVICE_PATTERN_BONUS": service_pattern_bonus,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_service_reviews": n_service_reviews,
    }

    if strength_note:
        result["strength_note"] = strength_note
    if weakness_note:
        result["weakness_note"] = weakness_note

    return result


def buckets_rate_below_threshold(l15_buckets: Dict[str, Any], threshold: float) -> bool:
    worst = l15_buckets.get("worst_aspect")
    if worst and worst in l15_buckets:
        return l15_buckets[worst].get("satisfaction_rate", 1.0) < threshold
    return False
