"""
Ground Truth computation for G4a (Server Quality - Simple).

Implements the formula from data/tasks/yelp/G4a_prompt.txt.
Simple formula without credibility weighting.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0


# =============================================================================
# Helpers
# =============================================================================


def compute_l1_positive_service(judgment: Dict[str, Any]) -> bool:
    """
    POSITIVE_SERVICE = true iff ANY:
      - ATTENTIVENESS in {excellent, good}
      - FRIENDLINESS in {warm, professional}
      - MENU_KNOWLEDGE in {expert, good}
      - ERROR_HANDLING = excellent
    """
    attentiveness = judgment.get("attentiveness", "not_mentioned")
    friendliness = judgment.get("friendliness", "not_mentioned")
    menu_knowledge = judgment.get("menu_knowledge", "not_mentioned")
    error_handling = judgment.get("error_handling", "not_mentioned")

    if attentiveness in ("excellent", "good"):
        return True
    if friendliness in ("warm", "professional"):
        return True
    if menu_knowledge in ("expert", "good"):
        return True
    if error_handling == "excellent":
        return True

    return False


def compute_l1_negative_service(judgment: Dict[str, Any]) -> bool:
    """
    NEGATIVE_SERVICE = true iff ANY:
      - ATTENTIVENESS in {poor, neglectful}
      - FRIENDLINESS in {cold, rude}
      - ERROR_HANDLING in {poor, denied}
      - PROFESSIONALISM = unprofessional
    """
    attentiveness = judgment.get("attentiveness", "not_mentioned")
    friendliness = judgment.get("friendliness", "not_mentioned")
    error_handling = judgment.get("error_handling", "not_mentioned")
    professionalism = judgment.get("professionalism", "not_mentioned")

    if attentiveness in ("poor", "neglectful"):
        return True
    if friendliness in ("cold", "rude"):
        return True
    if error_handling in ("poor", "denied"):
        return True
    if professionalism == "unprofessional":
        return True

    return False


# =============================================================================
# Main computation
# =============================================================================


def compute_ground_truth(
    judgments: List[Dict[str, Any]],
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute ground truth from extracted judgments."""
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
    n_excellent_attention = 0
    n_neglectful = 0
    n_rude = 0
    n_warm = 0
    n_named_server = 0

    for j in service_judgments:
        if compute_l1_positive_service(j):
            n_positive += 1
        if compute_l1_negative_service(j):
            n_negative += 1

        if j.get("attentiveness") == "excellent":
            n_excellent_attention += 1
        if j.get("attentiveness") == "neglectful":
            n_neglectful += 1
        if j.get("friendliness") == "rude":
            n_rude += 1
        if j.get("friendliness") == "warm":
            n_warm += 1
        if j.get("specific_server_named") == "yes":
            n_named_server += 1

    # Formulas
    positive_score = (n_positive * 1.5) + (n_excellent_attention * 0.5) + (n_warm * 0.5)
    negative_score = (n_negative * 1.5) + (n_neglectful * 1) + (n_rude * 2)

    raw_score = BASE_SCORE + positive_score - negative_score
    final_score = max(0.0, min(10.0, raw_score))

    # Decision Policy
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
    elif n_neglectful >= 2 and verdict in ("Excellent Service", "Good Service"):
        override_applied = "neglectful_max_mixed"
        verdict = "Mixed Service"
    elif n_excellent_attention >= 3 and n_negative == 0 and verdict in ("Mixed Service", "Poor Service"):
        override_applied = "excellent_attention_min_good"
        verdict = "Good Service"

    return {
        "N_SERVICE_REVIEWS": n_service_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_POSITIVE": n_positive,
        "N_NEGATIVE": n_negative,
        "N_EXCELLENT_ATTENTION": n_excellent_attention,
        "N_NEGLECTFUL": n_neglectful,
        "N_RUDE": n_rude,
        "N_WARM": n_warm,
        "N_NAMED_SERVER": n_named_server,
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_service_reviews": n_service_reviews,
    }
