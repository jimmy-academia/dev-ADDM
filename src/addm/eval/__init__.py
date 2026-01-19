"""Evaluation utilities."""

from addm.eval.metrics import (
    VERDICT_TO_ORDINAL,
    CLASS_NAMES,
    compute_ordinal_auprc,
    compute_primitive_accuracy,
    compute_verdict_consistency,
    evaluate_results,
)
from addm.eval.intermediate_metrics import (
    compute_intermediate_metrics,
    compute_evidence_validity,
    compute_classification_accuracy,
    compute_verdict_support,
    load_gt_with_incidents,
    build_reviews_data,
)

__all__ = [
    "VERDICT_TO_ORDINAL",
    "CLASS_NAMES",
    "compute_ordinal_auprc",
    "compute_primitive_accuracy",
    "compute_verdict_consistency",
    "evaluate_results",
    # Intermediate metrics
    "compute_intermediate_metrics",
    "compute_evidence_validity",
    "compute_classification_accuracy",
    "compute_verdict_support",
    "load_gt_with_incidents",
    "build_reviews_data",
]
