"""
Ground Truth computation for G6a (Uniqueness - Simple).

Implements the formula from data/tasks/yelp/G6a_prompt.txt.
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


def compute_l1_positive_uniqueness(judgment: Dict[str, Any]) -> bool:
    unique_dishes = judgment.get("unique_dishes", "not_mentioned")
    atmosphere_distinctiveness = judgment.get("atmosphere_distinctiveness", "not_mentioned")
    service_style = judgment.get("service_style", "not_mentioned")
    concept_innovation = judgment.get("concept_innovation", "not_mentioned")
    memorability = judgment.get("memorability", "not_mentioned")

    if unique_dishes in ("signature_standout", "distinctive", "creative"):
        return True
    if atmosphere_distinctiveness in ("one_of_a_kind", "memorable", "themed"):
        return True
    if service_style in ("exceptional_approach", "personalized"):
        return True
    if concept_innovation in ("groundbreaking", "fresh", "interesting"):
        return True
    if memorability in ("unforgettable", "memorable"):
        return True
    return False


def compute_l1_negative_uniqueness(judgment: Dict[str, Any]) -> bool:
    unique_dishes = judgment.get("unique_dishes", "not_mentioned")
    atmosphere_distinctiveness = judgment.get("atmosphere_distinctiveness", "not_mentioned")
    concept_innovation = judgment.get("concept_innovation", "not_mentioned")
    standout_feature = judgment.get("standout_feature", "not_mentioned")
    memorability = judgment.get("memorability", "not_mentioned")

    # standard AND generic AND derivative
    if (unique_dishes == "standard" and
        atmosphere_distinctiveness == "generic" and
        concept_innovation == "derivative"):
        return True
    if standout_feature == "none":
        return True
    if memorability == "forgettable":
        return True
    return False


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

    n_positive = 0
    n_negative = 0
    n_signature_dish = 0
    n_one_of_kind = 0
    n_forgettable = 0

    for j in uniqueness_judgments:
        if compute_l1_positive_uniqueness(j):
            n_positive += 1
        if compute_l1_negative_uniqueness(j):
            n_negative += 1

        if j.get("unique_dishes") == "signature_standout":
            n_signature_dish += 1
        if j.get("atmosphere_distinctiveness") == "one_of_a_kind":
            n_one_of_kind += 1
        if j.get("memorability") == "forgettable":
            n_forgettable += 1

    # Formulas
    positive_score = (n_positive * 1.5) + (n_signature_dish * 1.0) + (n_one_of_kind * 1.0)
    negative_score = (n_negative * 1.5) + (n_forgettable * 0.5)

    raw_score = BASE_SCORE + positive_score - negative_score
    final_score = max(0.0, min(10.0, raw_score))

    # Decision Policy
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
    atmosphere_highlight = None

    if n_signature_dish >= 2 and verdict in ("Average Uniqueness", "Generic"):
        override_applied = "signature_dish_min_distinctive"
        verdict = "Distinctive"
    elif n_forgettable >= 3 and verdict in ("Highly Unique", "Distinctive"):
        override_applied = "forgettable_max_average"
        verdict = "Average Uniqueness"

    if n_one_of_kind >= 1:
        atmosphere_highlight = "Unique atmosphere noted"

    result = {
        "N_UNIQUENESS_REVIEWS": n_uniqueness_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_POSITIVE": n_positive,
        "N_NEGATIVE": n_negative,
        "N_SIGNATURE_DISH": n_signature_dish,
        "N_ONE_OF_KIND": n_one_of_kind,
        "N_FORGETTABLE": n_forgettable,
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_uniqueness_reviews": n_uniqueness_reviews,
    }

    if atmosphere_highlight:
        result["atmosphere_highlight"] = atmosphere_highlight

    return result
