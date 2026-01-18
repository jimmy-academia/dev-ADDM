"""Tests for eval/metrics.py - Evaluation metrics."""

import numpy as np
import pytest
from addm.eval.metrics import (
    compute_ordinal_auprc,
    compute_primitive_accuracy,
    compute_verdict_consistency,
    evaluate_results,
    VERDICT_TO_ORDINAL,
)


class TestOrdinalAUPRC:
    """Test ordinal AUPRC computation."""

    def test_perfect_ranking(self):
        """Test AUPRC with perfect ranking."""
        y_true = np.array([0, 1, 2, 0, 1, 2])  # Low, High, Critical
        y_scores = np.array([0.1, 0.5, 0.9, 0.2, 0.6, 0.95])  # Perfect ordering

        metrics = compute_ordinal_auprc(y_true, y_scores)

        assert "ordinal_auprc" in metrics
        assert "auprc_ge_high" in metrics
        assert "auprc_ge_critical" in metrics
        assert metrics["ordinal_auprc"] > 0.9  # Should be very high
        assert metrics["n_samples"] == 6

    def test_random_ranking(self):
        """Test AUPRC with random ranking."""
        y_true = np.array([0, 0, 0, 1, 1, 2])
        y_scores = np.array([0.5, 0.3, 0.8, 0.2, 0.6, 0.1])  # Random ordering

        metrics = compute_ordinal_auprc(y_true, y_scores)

        assert "ordinal_auprc" in metrics
        # Random should be around 0.5 but can vary
        assert 0.0 <= metrics["ordinal_auprc"] <= 1.0

    def test_single_sample(self):
        """Test AUPRC with insufficient samples."""
        y_true = np.array([1])
        y_scores = np.array([0.5])

        metrics = compute_ordinal_auprc(y_true, y_scores)

        assert metrics["ordinal_auprc"] == 0.0
        assert metrics["n_samples"] == 1

    def test_all_same_class(self):
        """Test AUPRC when all samples have same label."""
        y_true = np.array([0, 0, 0, 0])  # All Low Risk
        y_scores = np.array([0.1, 0.3, 0.5, 0.7])

        metrics = compute_ordinal_auprc(y_true, y_scores)

        # Should not compute AUPRC for >=High (all negative class)
        assert "auprc_ge_high" not in metrics
        assert metrics["ordinal_auprc"] == 0.0

    def test_binary_split(self):
        """Test with clear binary split."""
        y_true = np.array([0, 0, 2, 2])  # Low and Critical only
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])

        metrics = compute_ordinal_auprc(y_true, y_scores)

        assert metrics["auprc_ge_critical"] > 0.9  # Should be high
        assert metrics["auprc_ge_high"] > 0.9  # Should also be high


class TestPrimitiveAccuracy:
    """Test primitive accuracy computation."""

    def test_exact_match(self):
        """Test exact matching of primitives."""
        predicted = {"count": 5, "score": 3.2, "flag": "true"}
        ground_truth = {"count": 5, "score": 3.2, "flag": "true"}

        accuracy = compute_primitive_accuracy(predicted, ground_truth)
        assert accuracy == 1.0

    def test_partial_match(self):
        """Test partial matching."""
        predicted = {"count": 5, "score": 3.2, "flag": "false"}
        ground_truth = {"count": 5, "score": 3.2, "flag": "true"}

        accuracy = compute_primitive_accuracy(predicted, ground_truth)
        assert accuracy == 2 / 3  # 2 out of 3 correct

    def test_with_tolerance(self):
        """Test numeric tolerance."""
        predicted = {"score": 3.21}
        ground_truth = {"score": 3.2}
        tolerances = {"score": 0.05}

        accuracy = compute_primitive_accuracy(predicted, ground_truth, tolerances)
        assert accuracy == 1.0

    def test_outside_tolerance(self):
        """Test value outside tolerance."""
        predicted = {"score": 3.3}
        ground_truth = {"score": 3.2}
        tolerances = {"score": 0.05}

        accuracy = compute_primitive_accuracy(predicted, ground_truth, tolerances)
        assert accuracy == 0.0

    def test_missing_predicted_field(self):
        """Test when predicted field is missing."""
        predicted = {"count": 5}
        ground_truth = {"count": 5, "score": 3.2}

        accuracy = compute_primitive_accuracy(predicted, ground_truth)
        # Only count field is checked (score missing)
        assert accuracy == 1.0

    def test_empty_ground_truth(self):
        """Test with empty ground truth."""
        predicted = {"count": 5}
        ground_truth = {}

        accuracy = compute_primitive_accuracy(predicted, ground_truth)
        assert accuracy == 0.0

    def test_type_mismatch(self):
        """Test when types don't match (int vs float should still work)."""
        predicted = {"count": 5}
        ground_truth = {"count": 5.0}

        accuracy = compute_primitive_accuracy(predicted, ground_truth)
        assert accuracy == 1.0


class TestVerdictConsistency:
    """Test verdict consistency checking."""

    def test_consistent_verdicts(self):
        """Test when verdicts match."""
        assert compute_verdict_consistency("High Risk", "High Risk") == 1

    def test_inconsistent_verdicts(self):
        """Test when verdicts don't match."""
        assert compute_verdict_consistency("High Risk", "Low Risk") == 0

    def test_case_sensitivity(self):
        """Test that comparison is case-sensitive."""
        assert compute_verdict_consistency("high risk", "High Risk") == 0


class TestEvaluateResults:
    """Test complete evaluation pipeline."""

    def test_basic_evaluation(self):
        """Test basic evaluation with simple results."""
        results = [
            {
                "business_id": "biz1",
                "verdict": "Low Risk",
                "risk_score": 1.5,
            },
            {
                "business_id": "biz2",
                "verdict": "High Risk",
                "risk_score": 5.0,
            },
            {
                "business_id": "biz3",
                "verdict": "Critical Risk",
                "risk_score": 8.0,
            },
        ]

        gt_verdicts = {
            "biz1": "Low Risk",
            "biz2": "High Risk",
            "biz3": "Critical Risk",
        }

        metrics = evaluate_results(results, gt_verdicts)

        assert "auprc_method" in metrics
        assert metrics["auprc_method"]["n_samples"] == 3

    def test_skip_errors(self):
        """Test that errors are skipped."""
        results = [
            {
                "business_id": "biz1",
                "verdict": "Low Risk",
                "risk_score": 1.5,
            },
            {
                "business_id": "biz2",
                "error": "API timeout",
            },
            {
                "business_id": "biz3",
                "verdict": "High Risk",
                "risk_score": 5.0,
            },
        ]

        gt_verdicts = {
            "biz1": "Low Risk",
            "biz2": "High Risk",
            "biz3": "High Risk",
        }

        metrics = evaluate_results(results, gt_verdicts)

        # Should only count 2 samples (biz2 has error)
        assert metrics["auprc_method"]["n_samples"] == 2

    def test_missing_ground_truth(self):
        """Test when GT is missing for a sample."""
        results = [
            {
                "business_id": "biz1",
                "verdict": "Low Risk",
                "risk_score": 1.5,
            },
            {
                "business_id": "biz2",
                "verdict": "High Risk",
                "risk_score": 5.0,
            },
        ]

        gt_verdicts = {
            "biz1": "Low Risk",
            # biz2 missing
        }

        metrics = evaluate_results(results, gt_verdicts)

        # Should only count biz1
        assert metrics["auprc_method"]["n_samples"] == 1

    def test_empty_results(self):
        """Test with empty results list."""
        results = []
        gt_verdicts = {"biz1": "Low Risk"}

        metrics = evaluate_results(results, gt_verdicts)

        # Should not crash, return empty metrics
        assert metrics == {}

    def test_verdict_ordinal_conversion(self):
        """Test that verdict strings convert to ordinals correctly."""
        assert VERDICT_TO_ORDINAL["Low Risk"] == 0
        assert VERDICT_TO_ORDINAL["High Risk"] == 1
        assert VERDICT_TO_ORDINAL["Critical Risk"] == 2
