"""
Ground Truth computation for G6c (Uniqueness - Complex).

Implements the formula from data/tasks/yelp/G6c_prompt.txt.
Complex formula with weighted uniqueness factors.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

DISH_UNIQUENESS_SCORES = {"signature_standout": 4.0, "distinctive": 2.5, "creative": 1.5, "standard": -0.5, "not_mentioned": 0}
ATMOSPHERE_SCORES = {"one_of_a_kind": 3.5, "memorable": 2.0, "themed": 1.5, "generic": -1.0, "not_mentioned": 0}
SERVICE_STYLE_SCORES = {"exceptional_approach": 3.0, "personalized": 2.0, "standard": 0, "impersonal": -1.5, "not_mentioned": 0}
CONCEPT_SCORES = {"groundbreaking": 3.5, "fresh": 2.0, "interesting": 1.0, "derivative": -1.5, "not_mentioned": 0}
STANDOUT_SCORES = {"multiple": 3.0, "yes_food": 2.0, "yes_atmosphere": 2.0, "yes_service": 2.0, "yes_value": 1.5, "yes_location": 1.5, "none": -2.0, "not_mentioned": 0}
MEMORABILITY_SCORES = {"unforgettable": 4.0, "memorable": 2.0, "forgettable": -2.5, "negative_memorable": -3.0, "not_mentioned": 0}
REPUTATION_SCORES = {"destination": 3.0, "local_favorite": 2.0, "hidden_gem": 2.5, "average": 0, "not_mentioned": 0}


# =============================================================================
# Helpers
# =============================================================================


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    unique_dishes = judgment.get("unique_dishes", "not_mentioned")
    atmosphere_distinctiveness = judgment.get("atmosphere_distinctiveness", "not_mentioned")
    service_style = judgment.get("service_style", "not_mentioned")
    concept_innovation = judgment.get("concept_innovation", "not_mentioned")
    standout_feature = judgment.get("standout_feature", "not_mentioned")
    memorability = judgment.get("memorability", "not_mentioned")
    local_reputation = judgment.get("local_reputation", "not_mentioned")

    dish_score = DISH_UNIQUENESS_SCORES.get(unique_dishes, 0)
    atmosphere_score = ATMOSPHERE_SCORES.get(atmosphere_distinctiveness, 0)
    service_score = SERVICE_STYLE_SCORES.get(service_style, 0)
    concept_score = CONCEPT_SCORES.get(concept_innovation, 0)
    standout_score = STANDOUT_SCORES.get(standout_feature, 0)
    memorability_score = MEMORABILITY_SCORES.get(memorability, 0)
    reputation_score = REPUTATION_SCORES.get(local_reputation, 0)

    l1_uniqueness_score = (
        dish_score + atmosphere_score + service_score +
        concept_score + standout_score + memorability_score + reputation_score
    )

    # Count generic ratings
    generic_count = 0
    if unique_dishes == "standard":
        generic_count += 1
    if atmosphere_distinctiveness == "generic":
        generic_count += 1
    if concept_innovation == "derivative":
        generic_count += 1
    if service_style == "impersonal":
        generic_count += 1

    if generic_count >= 3:
        total_generic_penalty = -2.0
    elif generic_count >= 2:
        total_generic_penalty = -1.0
    else:
        total_generic_penalty = 0.0

    l1_total_score = l1_uniqueness_score + total_generic_penalty

    return {
        "dish_uniqueness_score": dish_score,
        "atmosphere_score": atmosphere_score,
        "service_style_score": service_score,
        "concept_score": concept_score,
        "standout_score": standout_score,
        "memorability_score": memorability_score,
        "reputation_score": reputation_score,
        "l1_uniqueness_score": round(l1_uniqueness_score, 2),
        "total_generic_penalty": total_generic_penalty,
        "l1_total_score": round(l1_total_score, 2),
    }


# =============================================================================
# Main computation
# =============================================================================


def compute_ground_truth(
    judgments: List[Dict[str, Any]],
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
    uniqueness_judgments = [j for j in judgments if j.get("is_uniqueness_related", False)]
    n_uniqueness_reviews = len(uniqueness_judgments)

    if n_uniqueness_reviews == 0:
        confidence_level = "none"
    elif n_uniqueness_reviews <= 2:
        confidence_level = "low"
    elif n_uniqueness_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    sum_l1_score = 0.0
    n_signature = 0
    n_destination = 0
    n_forgettable = 0
    n_multiple_standouts = 0

    for j in uniqueness_judgments:
        l1 = compute_l1_complex(j)
        sum_l1_score += l1["l1_total_score"]

        if j.get("unique_dishes") == "signature_standout":
            n_signature += 1
        if j.get("local_reputation") == "destination":
            n_destination += 1
        if j.get("memorability") == "forgettable":
            n_forgettable += 1
        if j.get("standout_feature") == "multiple":
            n_multiple_standouts += 1

    mean_l1_score = sum_l1_score / max(n_uniqueness_reviews, 1)

    adjusted_score = mean_l1_score
    raw_score = BASE_SCORE + adjusted_score
    final_score = max(0.0, min(10.0, raw_score))

    if final_score >= 7.5:
        base_verdict_by_score = "Highly Unique"
    elif final_score >= 5.5:
        base_verdict_by_score = "Distinctive"
    elif final_score >= 3.5:
        base_verdict_by_score = "Average Uniqueness"
    else:
        base_verdict_by_score = "Generic"

    override_applied = "none"
    verdict = base_verdict_by_score
    destination_note = None

    if n_destination >= 2 and verdict in ("Average Uniqueness", "Generic"):
        override_applied = "destination_min_distinctive"
        verdict = "Distinctive"
        destination_note = "Known as a dining destination"
    elif n_multiple_standouts >= 2 and verdict in ("Average Uniqueness", "Generic"):
        override_applied = "multiple_standouts_min_distinctive"
        verdict = "Distinctive"
    elif n_forgettable >= 3 and verdict in ("Highly Unique", "Distinctive"):
        override_applied = "forgettable_max_average"
        verdict = "Average Uniqueness"
    elif mean_l1_score < -2:
        override_applied = "low_mean_score"
        verdict = "Generic"

    result = {
        "N_UNIQUENESS_REVIEWS": n_uniqueness_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_SIGNATURE": n_signature,
        "N_DESTINATION": n_destination,
        "N_FORGETTABLE": n_forgettable,
        "N_MULTIPLE_STANDOUTS": n_multiple_standouts,
        "SUM_L1_SCORE": round(sum_l1_score, 2),
        "MEAN_L1_SCORE": round(mean_l1_score, 3),
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_uniqueness_reviews": n_uniqueness_reviews,
    }

    if destination_note:
        result["destination_note"] = destination_note

    return result
