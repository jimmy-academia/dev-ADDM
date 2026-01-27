"""Evaluation utilities."""

from addm.eval.constants import (
    VERDICT_TO_ORDINAL,
    CLASS_NAMES,
    EVIDENCE_FIELDS,
    get_evidence_config,
    POLICY_VERDICTS,
    get_verdict_to_ordinal,
    # Scoring constants (for backward compatibility)
    SEVERITY_BASE_POINTS,
    MODIFIER_POINTS,
    VERDICT_THRESHOLDS,
)
from addm.eval.metrics import (
    normalize_verdict,
    normalize_judgement,
    compute_ordinal_auprc,
    # New simplified metrics entry point
    compute_evaluation_metrics,
    # New metrics (v2)
    compute_verdict_consistency_enhanced,
    extract_verdict_from_rule,
)
from addm.eval.intermediate_metrics import (
    compute_intermediate_metrics,
    compute_evidence_validity,
    compute_judgement_accuracy,
    compute_verdict_support,
    load_gt_with_incidents,
    build_reviews_data,
)
from addm.eval.unified_metrics import (
    # Legacy 3-score system (deprecated, kept for backward compat)
    compute_unified_metrics,
    compute_process_score,
    compute_consistency_score,
    compute_false_positive_rate,
    normalize_amos_output,
    PROCESS_WEIGHTS,
)

__all__ = [
    # Constants
    "VERDICT_TO_ORDINAL",
    "CLASS_NAMES",
    "EVIDENCE_FIELDS",
    "get_evidence_config",
    "POLICY_VERDICTS",
    "get_verdict_to_ordinal",
    # Scoring constants (backward compat)
    "SEVERITY_BASE_POINTS",
    "MODIFIER_POINTS",
    "VERDICT_THRESHOLDS",
    # Helper functions
    "extract_verdict_from_rule",
    # Core metrics
    "normalize_verdict",
    "normalize_judgement",
    "compute_ordinal_auprc",
    # New simplified metrics (recommended - 7 metrics)
    "compute_evaluation_metrics",
    "compute_verdict_consistency_enhanced",
    # Intermediate metrics
    "compute_intermediate_metrics",
    "compute_evidence_validity",
    "compute_judgement_accuracy",
    "compute_verdict_support",
    "load_gt_with_incidents",
    "build_reviews_data",
    # Legacy unified metrics (deprecated)
    "compute_unified_metrics",
    "compute_process_score",
    "compute_consistency_score",
    "compute_false_positive_rate",
    "normalize_amos_output",
    "PROCESS_WEIGHTS",
]
