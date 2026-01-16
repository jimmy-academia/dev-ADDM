"""
Ground Truth computation for G1a (Allergy Safety).

Implements the formula from data/tasks/yelp/G1a_prompt.txt.
Takes extracted L0 judgments and computes L1 → L2 → Final verdict.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List

# Cuisine risk modifiers from the prompt
CUISINE_MODIFIERS = {
    "Thai": 2.0,
    "Vietnamese": 1.8,
    "Chinese": 1.5,
    "Asian": 1.5,
    "Asian Fusion": 1.5,
    "Indian": 1.3,
    "Japanese": 1.2,
    "Korean": 1.2,
    "Mexican": 1.0,
    "Italian": 0.5,
    "American": 0.5,
    "American (Traditional)": 0.5,
    "American (New)": 0.5,
    "Pizza": 0.5,
}
DEFAULT_CUISINE_MODIFIER = 1.0

# Note: CURRENT_YEAR computed inside function to avoid stale value in long-running processes


@dataclass
class G1aGroundTruth:
    """Ground truth for G1a task."""

    # L2 Aggregates
    n_allergy_reviews: int
    confidence_level: str
    n_mild: int
    n_moderate: int
    n_severe: int
    n_total_incidents: int
    n_false_assurance: int

    # Derived values
    most_recent_incident_year: int
    incident_age: int
    recency_decay: float
    cuisine_modifier: float

    # Score components
    incident_score: float
    base_risk: float
    cuisine_impact: float
    raw_risk: float
    final_risk_score: float

    # Final
    base_verdict_by_score: str
    override_applied: str
    verdict: str


def get_cuisine_modifier(categories: str) -> float:
    """Get highest matching cuisine modifier from category string."""
    if not categories:
        return DEFAULT_CUISINE_MODIFIER

    cats = [c.strip() for c in categories.split(",")]
    max_modifier = DEFAULT_CUISINE_MODIFIER

    for cat in cats:
        # Direct match
        if cat in CUISINE_MODIFIERS:
            max_modifier = max(max_modifier, CUISINE_MODIFIERS[cat])
        # Partial match (e.g., "Thai Restaurant" contains "Thai")
        else:
            for cuisine, modifier in CUISINE_MODIFIERS.items():
                if cuisine.lower() in cat.lower():
                    max_modifier = max(max_modifier, modifier)

    return max_modifier


def derive_l1(judgment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derive L1 composites from L0 primitives.

    L1 composites:
    - FALSE_ASSURANCE: true iff (firsthand AND incident AND assurance_claim=true)
    - FIRSTHAND_INCIDENT: true iff (firsthand AND severity != none)
    """
    account_type = judgment.get("account_type", "hypothetical")
    incident_severity = judgment.get("incident_severity", "none")
    assurance_claim = judgment.get("assurance_claim", "false")

    # FALSE_ASSURANCE
    is_firsthand = account_type == "firsthand"
    has_incident = incident_severity in ("mild", "moderate", "severe")
    has_assurance = assurance_claim == "true"
    false_assurance = is_firsthand and has_incident and has_assurance

    # FIRSTHAND_INCIDENT
    firsthand_incident = is_firsthand and has_incident

    return {
        "false_assurance": false_assurance,
        "firsthand_incident": firsthand_incident,
    }


def compute_ground_truth(
    judgments: List[Dict[str, Any]],
    restaurant_meta: Dict[str, Any],
) -> G1aGroundTruth:
    """
    Compute ground truth from extracted judgments.

    Args:
        judgments: List of L0 judgments (one per allergy-related review)
        restaurant_meta: Restaurant metadata with 'categories' field

    Returns:
        G1aGroundTruth dataclass
    """
    categories = restaurant_meta.get("categories", "")

    current_year = datetime.now().year

    # Filter to allergy-related judgments only
    allergy_judgments = [
        j for j in judgments if j.get("is_allergy_related", False)
    ]
    n_allergy_reviews = len(allergy_judgments)

    # Confidence level
    if n_allergy_reviews == 0:
        confidence_level = "none"
    elif n_allergy_reviews <= 2:
        confidence_level = "low"
    elif n_allergy_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    # Derive L1 for each judgment and compute L2 aggregates
    n_mild = n_moderate = n_severe = 0
    n_false_assurance = 0
    incident_years: List[int] = []

    for j in allergy_judgments:
        l1 = derive_l1(j)

        # Count by severity (firsthand only)
        if l1["firsthand_incident"]:
            severity = j.get("incident_severity", "none")
            if severity == "mild":
                n_mild += 1
            elif severity == "moderate":
                n_moderate += 1
            elif severity == "severe":
                n_severe += 1

            # Track incident years for recency
            date_str = j.get("date", "")
            if date_str:
                try:
                    year = int(date_str[:4])
                    incident_years.append(year)
                except (ValueError, IndexError):
                    pass

        # Count false assurance
        if l1["false_assurance"]:
            n_false_assurance += 1

    n_total_incidents = n_mild + n_moderate + n_severe

    # Recency calculation
    if incident_years:
        most_recent_incident_year = max(incident_years)
    else:
        most_recent_incident_year = 2020

    incident_age = current_year - most_recent_incident_year
    recency_decay = max(0.3, 1.0 - (incident_age * 0.15))

    # Cuisine modifier
    cuisine_modifier = get_cuisine_modifier(categories)

    # Score computation
    incident_score = (n_mild * 2) + (n_moderate * 5) + (n_severe * 15)
    base_risk = 2.5
    cuisine_impact = cuisine_modifier * 0.5
    raw_risk = base_risk + (incident_score * recency_decay) + cuisine_impact
    final_risk_score = max(0.0, min(20.0, raw_risk))

    # Base verdict by score
    if final_risk_score < 4.0:
        base_verdict_by_score = "Low Risk"
    elif final_risk_score < 8.0:
        base_verdict_by_score = "High Risk"
    else:
        base_verdict_by_score = "Critical Risk"

    # Apply overrides
    override_applied = "none"
    verdict = base_verdict_by_score

    if n_severe >= 1:
        override_applied = "severe_incident"
        verdict = "Critical Risk"
    elif n_false_assurance >= 1 and verdict == "Low Risk":
        override_applied = "false_assurance"
        verdict = "High Risk"

    return G1aGroundTruth(
        n_allergy_reviews=n_allergy_reviews,
        confidence_level=confidence_level,
        n_mild=n_mild,
        n_moderate=n_moderate,
        n_severe=n_severe,
        n_total_incidents=n_total_incidents,
        n_false_assurance=n_false_assurance,
        most_recent_incident_year=most_recent_incident_year,
        incident_age=incident_age,
        recency_decay=round(recency_decay, 3),
        cuisine_modifier=cuisine_modifier,
        incident_score=float(incident_score),
        base_risk=base_risk,
        cuisine_impact=round(cuisine_impact, 3),
        raw_risk=round(raw_risk, 3),
        final_risk_score=round(final_risk_score, 2),
        base_verdict_by_score=base_verdict_by_score,
        override_applied=override_applied,
        verdict=verdict,
    )


def to_dict(gt: G1aGroundTruth) -> Dict[str, Any]:
    """Convert ground truth to dictionary."""
    return asdict(gt)
