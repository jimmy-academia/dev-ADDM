"""Shared constants for tasks module.

T* System: 5 tiers × 7 variants = 35 policies (T1P1, T1P2, etc.)

Variants:
- P1-P3: Rule variations (base, extended, TBD)
- P4-P7: Format variations (reorder v1, reorder v2, XML, prose)
"""

from typing import List, Optional

# =============================================================================
# T* SYSTEM (35 policies: 5 tiers × 7 variants)
# =============================================================================

# T1-T5 tiers (selected by Macro F1 performance)
TIERS = ["T1", "T2", "T3", "T4", "T5"]

# Mapping from tier to legacy G* topic (for GT file lookup)
# These G* files exist in data/answers/yelp/ from previous benchmark
TIER_TO_GT_TOPIC = {
    "T1": "G1_allergy",      # F1: 0.64
    "T2": "G3_price_worth",  # F1: 0.56
    "T3": "G4_environment",  # F1: 0.56
    "T4": "G5_execution",    # F1: 0.56
    "T5": "G4_server",       # F1: 0.56
}

# P1-P7 variant descriptions
VARIANTS = {
    "P1": "base",           # Base rules (standard markdown)
    "P2": "extended",       # Extended rules (compound conditions)
    "P3": "tbd",            # TBD rule variation
    "P4": "reorder_v1",     # Verdicts before Definitions
    "P5": "reorder_v2",     # Interleaved structure
    "P6": "xml",            # XML format
    "P7": "prose",          # Prose format
}

# All 35 policies: 5 tiers × 7 variants (T1P1, T1P2, ..., T5P7)
ALL_POLICIES = [f"{tier}P{v}" for tier in TIERS for v in range(1, 8)]

# K values for ground truth / context sizes
K_VALUES = [25, 50, 100, 200]


def expand_policies(
    policy: Optional[str] = None,
    tier: Optional[str] = None,
) -> List[str]:
    """Expand policy specifier to list of policy IDs.

    Args:
        policy: Comma-separated policy IDs (e.g., "T1P1,T1P2")
        tier: Tier ID (e.g., "T1") - expands to all 7 variants

    Returns:
        List of policy IDs. Returns ALL_POLICIES if no argument specified.

    Raises:
        ValueError: If tier is invalid.
    """
    if policy:
        return [p.strip() for p in policy.split(",") if p.strip()]

    if tier:
        tier_upper = tier.upper()
        if tier_upper not in TIERS:
            raise ValueError(f"Unknown tier: {tier}. Valid: {TIERS}")
        return [f"{tier_upper}P{v}" for v in range(1, 8)]

    return ALL_POLICIES.copy()


def get_tier_from_policy_id(policy_id: str) -> str:
    """Extract tier from policy ID.

    "T1P1" -> "T1"
    "T3P5" -> "T3"
    """
    # Format: T{n}P{m}
    if len(policy_id) >= 2 and policy_id[0] == "T" and policy_id[1].isdigit():
        return policy_id[:2]
    raise ValueError(f"Invalid policy_id format: {policy_id}. Expected T{{n}}P{{m}}")


def get_variant_from_policy_id(policy_id: str) -> str:
    """Extract variant from policy ID.

    "T1P1" -> "P1"
    "T3P5" -> "P5"
    """
    # Format: T{n}P{m}
    if "P" in policy_id:
        idx = policy_id.index("P")
        return policy_id[idx:]
    raise ValueError(f"Invalid policy_id format: {policy_id}. Expected T{{n}}P{{m}}")


def get_gt_topic(tier: str) -> str:
    """Get the G* topic for ground truth lookup.

    "T1" -> "G1_allergy"
    "T2" -> "G3_price_worth"
    """
    return TIER_TO_GT_TOPIC.get(tier, tier)


def is_valid_policy_id(policy_id: str) -> bool:
    """Check if policy ID is valid.

    "T1P1" -> True
    "T5P7" -> True
    "T1_P1" -> False (old format)
    "G1_allergy_V1" -> False (legacy)
    """
    return policy_id in ALL_POLICIES
