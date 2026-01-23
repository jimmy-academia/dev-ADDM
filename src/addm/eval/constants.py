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
