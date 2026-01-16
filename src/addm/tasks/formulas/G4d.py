"""
Ground Truth computation for G4d (Server Quality - Complex + L1.5).

Implements the formula from data/tasks/yelp/G4d_prompt.txt.
Complex formula with weighted service factors + L1.5 service aspect grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

ATTENTIVENESS_SCORES = {
    "excellent": 3.0, "good": 1.5, "adequate": 0, "poor": -2.0,
    "neglectful": -4.0, "not_mentioned": 0,
}
KNOWLEDGE_SCORES = {"expert": 2.0, "good": 1.0, "limited": -0.5, "poor": -2.0, "not_mentioned": 0}
FRIENDLINESS_SCORES = {"warm": 2.5, "professional": 1.5, "neutral": 0, "cold": -2.0, "rude": -4.0, "not_mentioned": 0}
ERROR_HANDLING_SCORES = {"excellent": 2.5, "good": 1.0, "poor": -2.0, "denied": -3.5, "no_errors": 0.5, "not_mentioned": 0}
PROFESSIONALISM_SCORES = {"exemplary": 2.0, "professional": 1.0, "unprofessional": -3.0, "not_mentioned": 0}
PERSONALIZATION_MODIFIERS = {"personalized": 1.5, "standard": 0, "impersonal": -0.5, "not_mentioned": 0}

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
    if judgment.get("attentiveness", "not_mentioned") != "not_mentioned":
        return "attentiveness"
    elif judgment.get("menu_knowledge", "not_mentioned") != "not_mentioned":
        return "knowledge"
    elif judgment.get("friendliness", "not_mentioned") != "not_mentioned" or judgment.get("professionalism", "not_mentioned") != "not_mentioned":
        return "attitude"
    return "other"


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    attentiveness = judgment.get("attentiveness", "not_mentioned")
    menu_knowledge = judgment.get("menu_knowledge", "not_mentioned")
    friendliness = judgment.get("friendliness", "not_mentioned")
    error_handling = judgment.get("error_handling", "not_mentioned")
    professionalism = judgment.get("professionalism", "not_mentioned")
    personalization = judgment.get("personalization", "not_mentioned")
    named_server = judgment.get("specific_server_named", "not_named")

    attentiveness_score = ATTENTIVENESS_SCORES.get(attentiveness, 0)
    knowledge_score = KNOWLEDGE_SCORES.get(menu_knowledge, 0)
    friendliness_score = FRIENDLINESS_SCORES.get(friendliness, 0)
    error_handling_score = ERROR_HANDLING_SCORES.get(error_handling, 0)
    professionalism_score = PROFESSIONALISM_SCORES.get(professionalism, 0)
    personalization_modifier = PERSONALIZATION_MODIFIERS.get(personalization, 0)

    l1_service_score = (
        attentiveness_score + knowledge_score + friendliness_score +
        error_handling_score + professionalism_score + personalization_modifier
    )

    rude_and_neglectful = -3.0 if (friendliness == "rude" and attentiveness in ("poor", "neglectful")) else 0.0
    warm_and_knowledgeable = 2.0 if (friendliness == "warm" and menu_knowledge == "expert") else 0.0

    if named_server == "positive":
        named_server_weight = 1.1
    elif named_server == "negative":
        named_server_weight = 1.2
    else:
        named_server_weight = 1.0

    l1_total_score = (l1_service_score + rude_and_neglectful + warm_and_knowledgeable) * named_server_weight

    return {
        "attentiveness_score": attentiveness_score,
        "knowledge_score": knowledge_score,
        "friendliness_score": friendliness_score,
        "l1_service_score": round(l1_service_score, 2),
        "rude_and_neglectful": rude_and_neglectful,
        "warm_and_knowledgeable": warm_and_knowledgeable,
        "named_server_weight": named_server_weight,
        "l1_total_score": round(l1_total_score, 2),
    }


def compute_l1_positive_service(judgment: Dict[str, Any]) -> bool:
    attentiveness = judgment.get("attentiveness", "not_mentioned")
    friendliness = judgment.get("friendliness", "not_mentioned")
    menu_knowledge = judgment.get("menu_knowledge", "not_mentioned")
    return attentiveness in ("excellent", "good") or friendliness in ("warm", "professional") or menu_knowledge in ("expert", "good")


def compute_l1_negative_service(judgment: Dict[str, Any]) -> bool:
    attentiveness = judgment.get("attentiveness", "not_mentioned")
    friendliness = judgment.get("friendliness", "not_mentioned")
    error_handling = judgment.get("error_handling", "not_mentioned")
    return attentiveness in ("poor", "neglectful") or friendliness in ("cold", "rude") or error_handling in ("poor", "denied")


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

    sum_l1_score = 0.0
    n_rude = 0
    n_neglectful = 0
    n_warm = 0
    n_named_negative = 0

    for j in service_judgments:
        l1 = compute_l1_complex(j)
        sum_l1_score += l1["l1_total_score"]

        if j.get("friendliness") == "rude":
            n_rude += 1
        if j.get("attentiveness") == "neglectful":
            n_neglectful += 1
        if j.get("friendliness") == "warm":
            n_warm += 1
        if j.get("specific_server_named") == "negative":
            n_named_negative += 1

    # L1.5 Service Buckets
    l15_buckets = compute_l15_buckets(service_judgments)
    service_pattern = l15_buckets["service_pattern"]
    service_pattern_bonus = SERVICE_PATTERN_BONUS.get(service_pattern, 0.0)

    mean_l1_score = sum_l1_score / max(n_service_reviews, 1)

    adjusted_score = mean_l1_score
    raw_score = BASE_SCORE + adjusted_score + service_pattern_bonus
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

    if service_pattern == "consistently_excellent" and verdict in ("Mixed Service", "Poor Service"):
        override_applied = "consistent_min_good"
        verdict = "Good Service"
    elif service_pattern == "needs_improvement" and (n_rude >= 1 or n_neglectful >= 2):
        override_applied = "needs_improvement_with_issues"
        verdict = "Poor Service"
    elif n_named_negative >= 2 and verdict in ("Excellent Service", "Good Service"):
        override_applied = "named_negative_max_mixed"
        verdict = "Mixed Service"
    elif mean_l1_score < -2:
        override_applied = "low_mean_score"
        verdict = "Poor Service"

    result = {
        "L1_5_service_buckets": l15_buckets,
        "N_SERVICE_REVIEWS": n_service_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_RUDE": n_rude,
        "N_NEGLECTFUL": n_neglectful,
        "N_WARM": n_warm,
        "N_NAMED_NEGATIVE": n_named_negative,
        "SUM_L1_SCORE": round(sum_l1_score, 2),
        "MEAN_L1_SCORE": round(mean_l1_score, 3),
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
        "SERVICE_PATTERN_BONUS": service_pattern_bonus,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_service_reviews": n_service_reviews,
    }

    return result
