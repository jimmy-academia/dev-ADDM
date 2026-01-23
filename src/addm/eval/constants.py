"""Evaluation constants for ADDM benchmark.

Canonical scoring values for V2 policies. These are used for:
- Verdict consistency checks (recompute verdict from evidence)
- Judgment accuracy evaluation

Note: Future improvement is to load these from policy YAML files
to support different scoring schemes across policy groups (G1-G6).
"""

# Canonical severity base points (V2 policy scoring)
SEVERITY_BASE_POINTS = {
    "mild": 2,
    "moderate": 5,
    "severe": 15,
}

# Modifier point adjustments (V2 policy)
MODIFIER_POINTS = {
    "False assurance": 5,
    "Dismissive staff": 3,
}

# Verdict thresholds (V2 policy)
# Score >= threshold → that verdict
VERDICT_THRESHOLDS = {
    "Critical Risk": 8,
    "High Risk": 4,
    "Low Risk": 0,
}

# Ordinal mapping for AUPRC computation
VERDICT_TO_ORDINAL = {
    "Low Risk": 0,
    "High Risk": 1,
    "Critical Risk": 2,
}

# Class names in order (low to high)
CLASS_NAMES = ["Low Risk", "High Risk", "Critical Risk"]

# Evidence field definitions per policy group
# Each group has a primary outcome field and values that constitute "evidence"
# (non-neutral values that indicate something noteworthy happened)
EVIDENCE_FIELDS = {
    "G1": {
        "field": "incident_severity",
        "evidence_values": {"mild", "moderate", "severe"},  # Exclude "none"
    },
    "G2": {
        "field": "date_outcome",
        "evidence_values": {"positive", "negative"},  # Exclude "neutral", "none"
    },
    "G3": {
        "field": "quality_for_price",
        "evidence_values": {"excellent", "good", "poor", "terrible"},  # Exclude "fair", "none"
    },
    "G4": {
        "field": "attentiveness",
        "evidence_values": {"excellent", "good", "poor", "terrible"},  # Exclude "adequate", "none"
    },
    "G5": {
        "field": "service_degradation",
        "evidence_values": {"minor", "moderate", "severe"},  # Exclude "none"
    },
    "G6": {
        "field": "memorability",
        "evidence_values": {"memorable", "remarkable", "forgettable"},  # Exclude "average", "none"
    },
}


def get_evidence_config(policy_id: str) -> dict:
    """Get evidence field configuration for a policy.

    Args:
        policy_id: Policy ID like "G1_allergy_V0"

    Returns:
        Dict with 'field' and 'evidence_values' keys
    """
    # Extract group from policy_id (e.g., "G1" from "G1_allergy_V0")
    group = policy_id.split("_")[0] if "_" in policy_id else policy_id[:2]
    return EVIDENCE_FIELDS.get(group, EVIDENCE_FIELDS["G1"])  # Default to G1


# Policy group verdict mappings (ordinal: lowest=0 to highest=n)
# Verified from src/addm/query/policies/G{1-6}/*/V0.yaml
POLICY_VERDICTS = {
    "G1": ["Low Risk", "High Risk", "Critical Risk"],
    "G2": ["Not Recommended", "Acceptable", "Recommended"],
    "G3": ["Poor Value", "Fair Value", "Good Value"],
    "G4": ["Needs Improvement", "Satisfactory", "Excellent"],
    "G5": ["Needs Improvement", "Satisfactory", "Excellent"],
    "G6": ["Generic", "Differentiated", "Highly Unique"],
}


def get_verdict_to_ordinal(policy_id: str) -> dict:
    """Get verdict-to-ordinal mapping for a policy.

    Args:
        policy_id: Policy ID like "G1_allergy_V0" or "G2_romance_V1"

    Returns:
        Dict mapping verdict strings to ordinal values (0, 1, 2, ...)
    """
    group = policy_id.split("_")[0] if "_" in policy_id else policy_id[:2]
    verdicts = POLICY_VERDICTS.get(group, POLICY_VERDICTS["G1"])
    return {v: i for i, v in enumerate(verdicts)}
