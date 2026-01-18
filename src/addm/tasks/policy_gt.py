"""Policy-based ground truth computation.

This module provides functions to compute ground truth verdicts from L0 judgments
using PolicyIR scoring rules (V2+) or qualitative decision rules (V0/V1).

Two-step flow:
1. extract.py: Extract L0 judgments from reviews (multi-model, aggregated)
2. compute_gt.py: Apply policy scoring to produce verdicts (this module)
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from addm.query.libraries import TERMS_DIR
from addm.query.models.policy import PolicyIR, ScoringSystem
from addm.query.models.term import Term, TermLibrary

# =============================================================================
# Multi-model aggregation constants
# =============================================================================

MODEL_WEIGHTS = {
    "gpt-5.1": 3,
    "gpt-5-mini": 2,
    "gpt-5-nano": 1,
}
REQUIRED_RUNS = {
    "gpt-5.1": 1,
    "gpt-5-mini": 3,
    "gpt-5-nano": 5,
}
TOTAL_WEIGHT = sum(MODEL_WEIGHTS[m] * REQUIRED_RUNS[m] for m in MODEL_WEIGHTS)  # 14

# =============================================================================
# Cuisine risk detection
# =============================================================================

HIGH_RISK_CUISINES = [
    "Thai",
    "Vietnamese",
    "Chinese",
    "Asian",
    "Asian Fusion",
]


def is_high_risk_cuisine(categories: str) -> bool:
    """Check if restaurant categories include high-risk cuisines."""
    if not categories:
        return False
    categories_lower = categories.lower()
    for cuisine in HIGH_RISK_CUISINES:
        if cuisine.lower() in categories_lower:
            return True
    return False


# =============================================================================
# Term library loading
# =============================================================================


def load_term_library() -> TermLibrary:
    """Load all term libraries from the standard location."""
    library = TermLibrary()

    # Load shared terms
    shared_path = TERMS_DIR / "_shared.yaml"
    if shared_path.exists():
        library.load_domain("shared", shared_path)

    # Load topic-specific terms (allergy, dietary, etc.)
    for yaml_file in TERMS_DIR.glob("*.yaml"):
        if yaml_file.name.startswith("_"):
            continue
        domain = yaml_file.stem  # e.g., "allergy" from "allergy.yaml"
        library.load_domain(domain, yaml_file)

    return library


def build_l0_schema_from_policy(
    policy: PolicyIR, library: TermLibrary
) -> Dict[str, Dict[str, str]]:
    """
    Build L0 extraction schema from policy's term references.

    Args:
        policy: PolicyIR with term refs in normative.terms
        library: Loaded TermLibrary

    Returns:
        Dict mapping field name (lowercase) to {value_id: description}
    """
    schema = {}
    for ref in policy.get_term_refs():
        term = library.resolve(ref)
        # Field name is term ID in lowercase (e.g., "ACCOUNT_TYPE" -> "account_type")
        field_name = term.id.lower()
        schema[field_name] = {v.id: v.description for v in term.values}
    return schema


def build_l0_schema_from_topic(topic: str, library: TermLibrary) -> Dict[str, Dict[str, str]]:
    """
    Build L0 extraction schema from a topic (loads any V* policy to get term refs).

    Topic format: "G1_allergy" (group_topic)

    Args:
        topic: Topic identifier (e.g., "G1_allergy")
        library: Loaded TermLibrary

    Returns:
        Dict mapping field name to {value_id: description}
    """
    # Find policy directory for this topic
    parts = topic.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid topic format: {topic} (expected G1_allergy)")

    group, topic_name = parts

    # Load any policy file (V0 is base)
    policies_dir = Path("src/addm/query/policies")
    policy_path = policies_dir / group / topic_name / "V0.yaml"

    if not policy_path.exists():
        raise FileNotFoundError(f"Policy not found: {policy_path}")

    policy = PolicyIR.load(policy_path)
    return build_l0_schema_from_policy(policy, library)


# =============================================================================
# Term version tracking (hash-based)
# =============================================================================


def compute_term_hash(l0_schema: Dict[str, Dict[str, str]]) -> str:
    """
    Compute deterministic hash of term definitions for version tracking.

    Args:
        l0_schema: L0 schema dict

    Returns:
        16-char hex hash string
    """
    canonical = json.dumps(l0_schema, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# =============================================================================
# Multi-model aggregation
# =============================================================================


def aggregate_judgments(
    raw_judgments: List[Dict[str, Any]], l0_schema: Dict[str, Dict[str, str]]
) -> Dict[str, Any]:
    """
    Aggregate multiple model extractions via weighted majority voting.

    Args:
        raw_judgments: List of raw judgments with "_model" field
        l0_schema: L0 schema for field enumeration

    Returns:
        Aggregated judgment with "_confidence" per field
    """
    if not raw_judgments:
        return {"_aggregated": True, "_confidence": {}}

    aggregated: Dict[str, Any] = {"_aggregated": True, "_confidence": {}}

    # Copy non-schema fields from first judgment (review_id, date, etc.)
    first = raw_judgments[0]
    for key in ["review_id", "date", "stars", "useful"]:
        if key in first:
            aggregated[key] = first[key]

    # Aggregate each L0 field via weighted voting
    for field in l0_schema.keys():
        votes: Dict[str, int] = defaultdict(int)

        for j in raw_judgments:
            model = j.get("_model", "gpt-5-nano")
            value = j.get(field, "none")
            weight = MODEL_WEIGHTS.get(model, 1)
            votes[value] += weight

        if votes:
            winner, weight = max(votes.items(), key=lambda x: x[1])
            aggregated[field] = winner
            total_votes = sum(votes.values())
            aggregated["_confidence"][field] = round(weight / total_votes, 3)
        else:
            # Default to first valid value
            aggregated[field] = list(l0_schema[field].keys())[0]
            aggregated["_confidence"][field] = 0.0

    # Handle is_allergy_related / is_relevant specially (not in l0_schema)
    relevance_votes: Dict[str, int] = defaultdict(int)
    for j in raw_judgments:
        model = j.get("_model", "gpt-5-nano")
        # Support both field names
        is_relevant = j.get("is_allergy_related", j.get("is_relevant", False))
        relevance_votes[str(is_relevant).lower()] += MODEL_WEIGHTS.get(model, 1)

    if relevance_votes:
        winner, _ = max(relevance_votes.items(), key=lambda x: x[1])
        aggregated["is_allergy_related"] = winner == "true"

    return aggregated


# =============================================================================
# Scoring-based GT computation (V2+)
# =============================================================================


def get_severity_points(severity: str, scoring: ScoringSystem) -> int:
    """Get points for a severity level from scoring system."""
    # Map severity to label (case-insensitive partial match)
    for sp in scoring.severity_points:
        if severity.lower() in sp.label.lower():
            return sp.points
    return 0


def get_modifier_points(label: str, scoring: ScoringSystem) -> int:
    """Get points for a modifier by label (case-insensitive partial match)."""
    for mod in scoring.modifiers:
        if label.lower() in mod.label.lower():
            return mod.points
    return 0


def determine_verdict(score: int, scoring: ScoringSystem) -> str:
    """Determine verdict based on score and thresholds."""
    # Sort thresholds descending by min_score
    sorted_thresholds = sorted(
        scoring.thresholds, key=lambda t: t.min_score, reverse=True
    )

    for threshold in sorted_thresholds:
        if score >= threshold.min_score:
            return threshold.verdict

    # Default to first verdict (usually "Low Risk")
    if sorted_thresholds:
        # Find the lowest threshold and return the "below" verdict
        lowest = min(scoring.thresholds, key=lambda t: t.min_score)
        # The verdict for scores below the lowest threshold
        # Get available verdicts from policy
        return "Low Risk"  # Standard default

    return "Unknown"


def compute_gt_from_policy_scoring(
    judgments: List[Dict[str, Any]],
    policy: PolicyIR,
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute GT using policy scoring rules (V2+).

    Args:
        judgments: List of aggregated L0 judgments (one per review)
        policy: PolicyIR with scoring system
        restaurant_meta: Restaurant metadata with 'categories' field

    Returns:
        Dict with verdict, score, and incident details
    """
    scoring = policy.normative.scoring
    if not scoring:
        raise ValueError(f"Policy {policy.policy_id} has no scoring system")

    categories = restaurant_meta.get("categories", "")
    total_score = 0
    incidents: List[Dict[str, Any]] = []

    for j in judgments:
        # Filter non-relevant
        is_relevant = j.get("is_allergy_related", j.get("is_relevant", False))
        if not is_relevant:
            continue

        # Filter non-firsthand
        if j.get("account_type", "hypothetical") != "firsthand":
            continue

        # Get severity
        severity = j.get("incident_severity", "none")
        if severity == "none":
            continue

        # Base points for severity
        points = get_severity_points(severity, scoring)

        # Check modifiers
        applied_modifiers: List[str] = []

        # False assurance modifier
        assurance = j.get("assurance_claim", "false")
        if assurance == "true":
            mod_pts = get_modifier_points("False assurance", scoring)
            points += mod_pts
            if mod_pts > 0:
                applied_modifiers.append("False assurance")

        # Dismissive staff modifier
        staff_response = j.get("staff_response", "none")
        if staff_response == "dismissive":
            mod_pts = get_modifier_points("Dismissive staff", scoring)
            points += mod_pts
            if mod_pts > 0:
                applied_modifiers.append("Dismissive staff")

        total_score += points
        incidents.append({
            "review_id": j.get("review_id", ""),
            "severity": severity,
            "base_points": get_severity_points(severity, scoring),
            "modifiers": applied_modifiers,
            "total_points": points,
        })

    # Cuisine modifier (restaurant-level, not per-incident)
    cuisine_modifier_applied = False
    if is_high_risk_cuisine(categories):
        mod_pts = get_modifier_points("High-risk cuisine", scoring)
        total_score += mod_pts
        if mod_pts > 0:
            cuisine_modifier_applied = True

    # Determine verdict
    verdict = determine_verdict(total_score, scoring)

    return {
        "verdict": verdict,
        "score": total_score,
        "n_incidents": len(incidents),
        "incidents": incidents,
        "cuisine_modifier": cuisine_modifier_applied,
        "policy_id": policy.policy_id,
    }


# =============================================================================
# Qualitative GT computation (V0/V1)
# =============================================================================


def evaluate_structured_condition(
    condition: Dict[str, Any], judgments: List[Dict[str, Any]]
) -> bool:
    """
    Evaluate a structured condition against judgments.

    Condition types:
    - count_threshold: count matching >= min_count
    - exists: at least one matching
    """
    cond_type = condition.get("type", "count_threshold")
    filter_spec = condition.get("filter", {})
    min_count = condition.get("min_count", 1)

    # Count matching judgments
    count = 0
    for j in judgments:
        # Filter non-relevant
        is_relevant = j.get("is_allergy_related", j.get("is_relevant", False))
        if not is_relevant:
            continue

        # Check all filter conditions
        matches = True
        for field, expected in filter_spec.items():
            actual = j.get(field, "")
            if isinstance(expected, list):
                if actual not in expected:
                    matches = False
                    break
            elif actual != expected:
                matches = False
                break

        if matches:
            count += 1

    if cond_type == "exists":
        return count >= 1
    else:  # count_threshold
        return count >= min_count


def compute_gt_from_policy_qualitative(
    judgments: List[Dict[str, Any]],
    policy: PolicyIR,
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute GT using qualitative decision rules (V0/V1).

    Args:
        judgments: List of aggregated L0 judgments
        policy: PolicyIR with decision rules
        restaurant_meta: Restaurant metadata

    Returns:
        Dict with verdict and rule that matched
    """
    rules = policy.normative.decision.rules

    # Evaluate rules in order (precedence ladder)
    for rule in rules:
        if rule.default:
            continue  # Skip default, evaluate it last

        # Get structured conditions
        structured = rule.get_structured_conditions()

        if not structured:
            # No structured conditions, skip (NL-only rules)
            continue

        # Evaluate based on logic (ANY or ALL)
        if rule.logic.value == "ANY":
            matched = any(
                evaluate_structured_condition(c, judgments) for c in structured
            )
        else:  # ALL
            matched = all(
                evaluate_structured_condition(c, judgments) for c in structured
            )

        if matched:
            return {
                "verdict": rule.verdict,
                "matched_rule": rule.label,
                "policy_id": policy.policy_id,
            }

    # Find default rule
    for rule in rules:
        if rule.default:
            return {
                "verdict": rule.verdict,
                "matched_rule": rule.label,
                "policy_id": policy.policy_id,
            }

    return {
        "verdict": "Unknown",
        "matched_rule": None,
        "policy_id": policy.policy_id,
    }


# =============================================================================
# Main entry point
# =============================================================================


def compute_gt_from_policy(
    judgments: List[Dict[str, Any]],
    policy: PolicyIR,
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute ground truth using policy rules.

    Automatically selects scoring (V2+) or qualitative (V0/V1) based on policy.

    Args:
        judgments: List of aggregated L0 judgments (one per review)
        policy: PolicyIR loaded from YAML
        restaurant_meta: Restaurant metadata with 'categories' field

    Returns:
        Dict with verdict and supporting details
    """
    if policy.normative.scoring:
        return compute_gt_from_policy_scoring(judgments, policy, restaurant_meta)
    else:
        return compute_gt_from_policy_qualitative(judgments, policy, restaurant_meta)


def load_policy(policy_id: str) -> PolicyIR:
    """
    Load a policy by ID.

    Policy ID format: "G1_allergy_V2" -> policies/G1/allergy/V2.yaml

    Args:
        policy_id: Policy identifier

    Returns:
        Loaded PolicyIR
    """
    parts = policy_id.split("_")
    if len(parts) != 3:
        raise ValueError(f"Invalid policy_id format: {policy_id} (expected G1_allergy_V2)")

    group, topic, version = parts
    policies_dir = Path("src/addm/query/policies")
    policy_path = policies_dir / group / topic / f"{version}.yaml"

    if not policy_path.exists():
        raise FileNotFoundError(f"Policy not found: {policy_path}")

    return PolicyIR.load(policy_path)


def get_topic_from_policy_id(policy_id: str) -> str:
    """
    Extract topic from policy ID.

    "G1_allergy_V2" -> "G1_allergy"
    """
    parts = policy_id.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid policy_id: {policy_id}")
    return f"{parts[0]}_{parts[1]}"
