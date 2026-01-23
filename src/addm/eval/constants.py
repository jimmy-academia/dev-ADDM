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
