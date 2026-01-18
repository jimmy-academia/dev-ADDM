"""Policy evaluation module for ground truth computation.

This module provides tools to evaluate judgments against PolicyIR
to compute ground truth verdicts.
"""

from .policy_evaluator import PolicyEvaluator
from .scoring import compute_score, apply_thresholds

__all__ = ["PolicyEvaluator", "compute_score", "apply_thresholds"]
