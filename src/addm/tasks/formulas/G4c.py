"""
Ground Truth computation for G4c (Server Quality - Complex).

Implements the formula from data/tasks/yelp/G4c_prompt.txt.
Complex formula with weighted service factors and interaction effects.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

ATTENTIVENESS_SCORES = {
    "excellent": 3.0,
    "good": 1.5,
    "adequate": 0,
    "poor": -2.0,
    "neglectful": -4.0,
    "not_mentioned": 0,
}

KNOWLEDGE_SCORES = {
    "expert": 2.0,
    "good": 1.0,
    "limited": -0.5,
    "poor": -2.0,
    "not_mentioned": 0,
}

FRIENDLINESS_SCORES = {
    "warm": 2.5,
    "professional": 1.5,
    "neutral": 0,
    "cold": -2.0,
    "rude": -4.0,
    "not_mentioned": 0,
}

ERROR_HANDLING_SCORES = {
    "excellent": 2.5,
    "good": 1.0,
    "poor": -2.0,
    "denied": -3.5,
    "no_errors": 0.5,
    "not_mentioned": 0,
}

PROFESSIONALISM_SCORES = {
    "exemplary": 2.0,
    "professional": 1.0,
    "unprofessional": -3.0,
    "not_mentioned": 0,
}

PERSONALIZATION_MODIFIERS = {
    "personalized": 1.5,
    "standard": 0,
    "impersonal": -0.5,
    "not_mentioned": 0,
}


# =============================================================================
# Helpers
# =============================================================================


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

    # Interaction effects
    rude_and_neglectful = -3.0 if (friendliness == "rude" and attentiveness in ("poor", "neglectful")) else 0.0
    warm_and_knowledgeable = 2.0 if (friendliness == "warm" and menu_knowledge == "expert") else 0.0

    # Named server weight
    if named_server == "positive":
        named_server_weight = 1.1
    elif named_server == "negative":
        named_server_weight = 1.2  # Amplify complaints
    else:
        named_server_weight = 1.0

    l1_total_score = (l1_service_score + rude_and_neglectful + warm_and_knowledgeable) * named_server_weight

    return {
        "attentiveness_score": attentiveness_score,
        "knowledge_score": knowledge_score,
        "friendliness_score": friendliness_score,
        "error_handling_score": error_handling_score,
        "professionalism_score": professionalism_score,
        "personalization_modifier": personalization_modifier,
        "l1_service_score": round(l1_service_score, 2),
        "rude_and_neglectful": rude_and_neglectful,
        "warm_and_knowledgeable": warm_and_knowledgeable,
        "named_server_weight": named_server_weight,
        "l1_total_score": round(l1_total_score, 2),
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
    n_excellent_attention = 0
    n_named_positive = 0
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
        if j.get("attentiveness") == "excellent":
            n_excellent_attention += 1
        if j.get("specific_server_named") == "positive":
            n_named_positive += 1
        if j.get("specific_server_named") == "negative":
            n_named_negative += 1

    mean_l1_score = sum_l1_score / max(n_service_reviews, 1)

    adjusted_score = mean_l1_score
    raw_score = BASE_SCORE + adjusted_score
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

    if n_rude >= 2:
        override_applied = "rude_pattern"
        verdict = "Poor Service"
    elif n_neglectful >= 2 and n_rude >= 1:
        override_applied = "neglectful_and_rude"
        verdict = "Poor Service"
    elif n_named_negative >= 2 and verdict in ("Excellent Service", "Good Service"):
        override_applied = "named_negative_max_mixed"
        verdict = "Mixed Service"
    elif mean_l1_score >= 3 and n_warm >= 2 and verdict in ("Mixed Service", "Poor Service"):
        override_applied = "warm_min_good"
        verdict = "Good Service"
    elif mean_l1_score < -2:
        override_applied = "low_mean_score"
        verdict = "Poor Service"

    return {
        "N_SERVICE_REVIEWS": n_service_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_RUDE": n_rude,
        "N_NEGLECTFUL": n_neglectful,
        "N_WARM": n_warm,
        "N_EXCELLENT_ATTENTION": n_excellent_attention,
        "N_NAMED_POSITIVE": n_named_positive,
        "N_NAMED_NEGATIVE": n_named_negative,
        "SUM_L1_SCORE": round(sum_l1_score, 2),
        "MEAN_L1_SCORE": round(mean_l1_score, 3),
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_service_reviews": n_service_reviews,
    }
