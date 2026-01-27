"""Evaluation constants for ADDM benchmark.

T* System: 5 tiers × 7 variants = 35 policies (T1P1, T1P2, etc.)

Each tier has evidence fields and verdict mappings for evaluation.
"""

# Ordinal mapping for AUPRC computation (T1 default)
VERDICT_TO_ORDINAL = {
    "Low Risk": 0,
    "High Risk": 1,
    "Critical Risk": 2,
}

# Scoring constants (for backward compatibility with V3/V4 evaluation logic)
# These are used by compute_verdict_consistency_enhanced in metrics.py
SEVERITY_BASE_POINTS = {
    "mild": 2,
    "moderate": 5,
    "severe": 15,
}

MODIFIER_POINTS = {
    "False assurance": 5,
    "Dismissive staff": 3,
}

VERDICT_THRESHOLDS = {
    "Critical Risk": 8,
    "High Risk": 4,
    "Low Risk": 0,
}

# Class names in order (low to high) - T1 default
CLASS_NAMES = ["Low Risk", "High Risk", "Critical Risk"]

# Evidence field definitions per tier
# Each tier has a primary outcome field and values that constitute "evidence"
# (non-neutral values that indicate something noteworthy happened)
EVIDENCE_FIELDS = {
    "T1": {  # Allergy safety
        "field": "incident_severity",
        "evidence_values": {"mild", "moderate", "severe"},
    },
    "T2": {  # Price worth
        "field": "price_perception",
        "evidence_values": {"good_value", "overpriced", "ripoff", "steal"},
    },
    "T3": {  # Environment
        "field": "cleanliness",
        "evidence_values": {"dirty", "fair", "spotless"},
    },
    "T4": {  # Execution
        "field": "issue_resolution",
        "evidence_values": {"excellent", "none", "poor"},
    },
    "T5": {  # Server
        "field": "attentiveness",
        "evidence_values": {"absent", "excellent", "good", "poor"},
    },
}


def get_evidence_config(policy_id: str) -> dict:
    """Get evidence field configuration for a policy.

    Args:
        policy_id: Policy ID like "T1P1", "T3P5"

    Returns:
        Dict with 'field' and 'evidence_values' keys
    """
    # Extract tier from policy_id: "T1P1" -> "T1"
    if len(policy_id) >= 2 and policy_id[0] == "T" and policy_id[1].isdigit():
        tier = policy_id[:2]
        return EVIDENCE_FIELDS.get(tier, EVIDENCE_FIELDS["T1"])

    # Fallback
    return EVIDENCE_FIELDS["T1"]


# Policy tier verdict mappings (ordinal: lowest=0 to highest=n)
POLICY_VERDICTS = {
    "T1": ["Low Risk", "High Risk", "Critical Risk"],
    "T2": ["Poor Value", "Fair Value", "Good Value"],
    "T3": ["Needs Improvement", "Satisfactory", "Excellent"],
    "T4": ["Needs Improvement", "Satisfactory", "Excellent"],
    "T5": ["Needs Improvement", "Satisfactory", "Excellent"],
}


def get_verdict_to_ordinal(policy_id: str) -> dict:
    """Get verdict-to-ordinal mapping for a policy.

    Args:
        policy_id: Policy ID like "T1P1", "T2P3"

    Returns:
        Dict mapping verdict strings to ordinal values (0, 1, 2, ...)
    """
    # Extract tier: "T1P1" -> "T1"
    if len(policy_id) >= 2 and policy_id[0] == "T" and policy_id[1].isdigit():
        tier = policy_id[:2]
    else:
        tier = "T1"  # fallback

    verdicts = POLICY_VERDICTS.get(tier, POLICY_VERDICTS["T1"])
    return {v: i for i, v in enumerate(verdicts)}
