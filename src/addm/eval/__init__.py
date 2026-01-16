"""Evaluation utilities."""

from addm.eval.metrics import (
    VERDICT_TO_ORDINAL,
    CLASS_NAMES,
    compute_ordinal_auprc,
    compute_primitive_accuracy,
    compute_verdict_consistency,
    evaluate_results,
)

__all__ = [
    "VERDICT_TO_ORDINAL",
    "CLASS_NAMES",
    "compute_ordinal_auprc",
    "compute_primitive_accuracy",
    "compute_verdict_consistency",
    "evaluate_results",
]
