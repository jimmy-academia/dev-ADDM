"""
Ground Truth computation for G6d (Uniqueness - Complex + L1.5).

Implements the formula from data/tasks/yelp/G6d_prompt.txt.
Complex formula with weighted uniqueness factors + L1.5 uniqueness-type grouping.

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
MEMORABILITY_SCORES = {"unforgettable": 4.0, "memorable": 2.0, "forgettable": -2.5, "negative_memorable": -3.0, "not_mentioned": 0}


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

    dish_score = DISH_UNIQUENESS_SCORES.get(unique_dishes, 0)
    atmosphere_score = ATMOSPHERE_SCORES.get(atmosphere_distinctiveness, 0)
    service_score = SERVICE_STYLE_SCORES.get(service_style, 0)
    concept_score = CONCEPT_SCORES.get(concept_innovation, 0)
    memorability_score = MEMORABILITY_SCORES.get(memorability, 0)

    l1_uniqueness_score = (
        dish_score + atmosphere_score + service_score +
        concept_score + memorability_score
    )

    multi_dimensional_bonus = 2.0 if standout_feature == "multiple" else 0.0
    l1_total_score = l1_uniqueness_score + multi_dimensional_bonus

    return {
        "dish_uniqueness_score": dish_score,
        "atmosphere_score": atmosphere_score,
        "service_style_score": service_score,
        "concept_score": concept_score,
        "memorability_score": memorability_score,
        "l1_uniqueness_score": round(l1_uniqueness_score, 2),
        "multi_dimensional_bonus": multi_dimensional_bonus,
        "l1_total_score": round(l1_total_score, 2),
    }


def compute_l15_buckets(uniqueness_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = {
        "culinary": {"n_reviews": 0, "sum_score": 0.0, "n_exceptional": 0, "n_generic": 0},
        "atmosphere": {"n_reviews": 0, "sum_score": 0.0, "n_exceptional": 0, "n_generic": 0},
        "experience": {"n_reviews": 0, "sum_score": 0.0, "n_exceptional": 0, "n_generic": 0},
    }

    for j in uniqueness_judgments:
        unique_dishes = j.get("unique_dishes", "not_mentioned")
        atmosphere_distinctiveness = j.get("atmosphere_distinctiveness", "not_mentioned")
        service_style = j.get("service_style", "not_mentioned")
        concept_innovation = j.get("concept_innovation", "not_mentioned")

        dish_score = DISH_UNIQUENESS_SCORES.get(unique_dishes, 0)
        atmosphere_score = ATMOSPHERE_SCORES.get(atmosphere_distinctiveness, 0)
        service_score = SERVICE_STYLE_SCORES.get(service_style, 0)
        concept_score = CONCEPT_SCORES.get(concept_innovation, 0)

        if unique_dishes != "not_mentioned":
            buckets["culinary"]["n_reviews"] += 1
            buckets["culinary"]["sum_score"] += dish_score
            if unique_dishes == "signature_standout":
                buckets["culinary"]["n_exceptional"] += 1
            if unique_dishes == "standard":
                buckets["culinary"]["n_generic"] += 1

        if atmosphere_distinctiveness != "not_mentioned":
            buckets["atmosphere"]["n_reviews"] += 1
            buckets["atmosphere"]["sum_score"] += atmosphere_score
            if atmosphere_distinctiveness == "one_of_a_kind":
                buckets["atmosphere"]["n_exceptional"] += 1
            if atmosphere_distinctiveness == "generic":
                buckets["atmosphere"]["n_generic"] += 1

        if service_style != "not_mentioned" or concept_innovation != "not_mentioned":
            buckets["experience"]["n_reviews"] += 1
            buckets["experience"]["sum_score"] += service_score + concept_score
            if service_style == "exceptional_approach" or concept_innovation == "groundbreaking":
                buckets["experience"]["n_exceptional"] += 1
            if service_style == "standard" or concept_innovation == "derivative":
                buckets["experience"]["n_generic"] += 1

    for key, bucket in buckets.items():
        bucket["mean_score"] = bucket["sum_score"] / max(bucket["n_reviews"], 1)

    buckets_with_data = [(k, v) for k, v in buckets.items() if v["n_reviews"] > 0]

    if buckets_with_data:
        strongest_category, strongest_bucket = max(buckets_with_data, key=lambda x: x[1]["mean_score"])
        strongest_score = strongest_bucket["mean_score"]
        weakest_category, weakest_bucket = min(buckets_with_data, key=lambda x: x[1]["mean_score"])
        weakest_score = weakest_bucket["mean_score"]
    else:
        strongest_category = None
        strongest_score = 0.0
        weakest_category = None
        weakest_score = 0.0

    n_categories_distinctive = sum(1 for k, v in buckets.items() if v["mean_score"] >= 2.0 and v["n_reviews"] > 0)

    if n_categories_distinctive >= 3:
        uniqueness_pattern = "original_concept"
    elif buckets["culinary"]["mean_score"] >= 3.0 and buckets["culinary"]["n_reviews"] > 0:
        uniqueness_pattern = "culinary_destination"
    elif buckets["atmosphere"]["mean_score"] >= 3.0 and buckets["atmosphere"]["n_reviews"] > 0:
        uniqueness_pattern = "immersive_environment"
    elif buckets["experience"]["mean_score"] >= 2.5 and buckets["experience"]["n_reviews"] > 0:
        uniqueness_pattern = "experiential_dining"
    elif strongest_score - weakest_score > 3.0 and buckets_with_data:
        uniqueness_pattern = "one_dimensional"
    elif n_categories_distinctive >= 1:
        uniqueness_pattern = "emerging_identity"
    else:
        uniqueness_pattern = "identity_crisis"

    return {
        "culinary": buckets["culinary"],
        "atmosphere": buckets["atmosphere"],
        "experience": buckets["experience"],
        "strongest_category": strongest_category,
        "strongest_score": round(strongest_score, 3) if strongest_score else 0.0,
        "weakest_category": weakest_category,
        "weakest_score": round(weakest_score, 3) if weakest_score else 0.0,
        "n_categories_distinctive": n_categories_distinctive,
        "uniqueness_pattern": uniqueness_pattern,
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

    n_unforgettable = 0
    n_forgettable = 0

    for j in uniqueness_judgments:
        if j.get("memorability") == "unforgettable":
            n_unforgettable += 1
        if j.get("memorability") == "forgettable":
            n_forgettable += 1

    # L1.5 Category Buckets
    l15_buckets = compute_l15_buckets(uniqueness_judgments)
    uniqueness_pattern = l15_buckets["uniqueness_pattern"]

    # Pattern multiplier
    if uniqueness_pattern == "original_concept":
        pattern_mult = 1.25
    elif uniqueness_pattern in ("culinary_destination", "immersive_environment", "experiential_dining"):
        pattern_mult = 1.15
    elif uniqueness_pattern == "one_dimensional":
        pattern_mult = 0.95
    elif uniqueness_pattern == "identity_crisis":
        pattern_mult = 0.7
    else:
        pattern_mult = 1.0

    # Aggregate L1.5 scores
    total_category_score = sum(
        v["mean_score"] for v in [l15_buckets["culinary"], l15_buckets["atmosphere"], l15_buckets["experience"]]
        if v["n_reviews"] > 0
    )
    n_category_buckets_active = sum(
        1 for v in [l15_buckets["culinary"], l15_buckets["atmosphere"], l15_buckets["experience"]]
        if v["n_reviews"] > 0
    )
    mean_category_score = total_category_score / max(n_category_buckets_active, 1)

    adjusted_score = mean_category_score * pattern_mult
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
    identity_note = None
    development_note = None

    if n_unforgettable >= 2 and verdict in ("Average Uniqueness", "Generic"):
        override_applied = "unforgettable_min_distinctive"
        verdict = "Distinctive"
    elif uniqueness_pattern == "original_concept" and verdict in ("Average Uniqueness", "Generic"):
        override_applied = "original_concept_min_distinctive"
        verdict = "Distinctive"
    elif uniqueness_pattern == "identity_crisis" and n_forgettable >= 2 and verdict in ("Highly Unique", "Distinctive"):
        override_applied = "identity_crisis_max_average"
        verdict = "Average Uniqueness"

    if l15_buckets["strongest_category"]:
        identity_note = f"Identity strength: {l15_buckets['strongest_category']}"

    if l15_buckets["weakest_category"] and l15_buckets["weakest_score"] < 0:
        development_note = f"Could improve: {l15_buckets['weakest_category']}"

    result = {
        "L1_5_category_buckets": l15_buckets,
        "N_UNIQUENESS_REVIEWS": n_uniqueness_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_UNFORGETTABLE": n_unforgettable,
        "N_FORGETTABLE": n_forgettable,
        "TOTAL_CATEGORY_SCORE": round(total_category_score, 3),
        "MEAN_CATEGORY_SCORE": round(mean_category_score, 3),
        "PATTERN_MULT": pattern_mult,
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_uniqueness_reviews": n_uniqueness_reviews,
    }

    if identity_note:
        result["identity_note"] = identity_note
    if development_note:
        result["development_note"] = development_note

    return result
