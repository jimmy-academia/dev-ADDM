"""DEPRECATED: Tests for formula modules that no longer exist.

The ADDM project has transitioned from formula modules (src/addm/tasks/formulas/)
to a policy-based system (src/addm/query/policies/). These tests will fail with
ModuleNotFoundError and are kept only for historical reference.

Original purpose:
Comprehensive tests for all 72 formula modules.
Converted from scripts/verify_formulas.py to pytest format.
Tests basic functionality and schema compliance for all task formulas.

Current system:
- Policy definitions: src/addm/query/policies/ (72 YAML files, G1-G6)
- Term libraries: src/addm/query/libraries/terms/
- Ground truth: src/addm/tasks/policy_gt.py (policy-based GT computation)

All 382 tests in this file will fail. This is expected and intentional.
"""

import importlib
import inspect
import pytest
from typing import Any, Dict, List

# All 72 task IDs (6 groups × 12 variants)
ALL_TASKS = [f"G{g}{t}" for g in range(1, 7) for t in "abcdefghijkl"]

# Which variants have L1.5 (b/d for topic 1, f/h for topic 2, j/l for topic 3)
L15_VARIANTS = {"b", "d", "f", "h", "j", "l"}


class TestFormulaModules:
    """Test that all 72 formula modules can be imported and have correct structure."""

    @pytest.mark.parametrize("task_id", ALL_TASKS)
    def test_module_imports(self, task_id):
        """Test that formula module can be imported."""
        module = importlib.import_module(f"addm.tasks.formulas.{task_id}")
        assert module is not None

    @pytest.mark.parametrize("task_id", ALL_TASKS)
    def test_has_compute_ground_truth(self, task_id):
        """Test that module has compute_ground_truth function."""
        module = importlib.import_module(f"addm.tasks.formulas.{task_id}")
        assert hasattr(module, "compute_ground_truth")
        assert callable(module.compute_ground_truth)

    @pytest.mark.parametrize("task_id", ALL_TASKS)
    def test_function_signature(self, task_id):
        """Test that compute_ground_truth has correct signature."""
        module = importlib.import_module(f"addm.tasks.formulas.{task_id}")
        fn = module.compute_ground_truth
        sig = inspect.signature(fn)

        params = list(sig.parameters.keys())
        # Should have 2 parameters: judgments, restaurant_meta
        assert len(params) == 2, f"{task_id}: Expected 2 params, got {len(params)}"
        assert params[0] == "judgments", f"{task_id}: First param should be 'judgments'"
        assert params[1] == "restaurant_meta", f"{task_id}: Second param should be 'restaurant_meta'"


class TestFormulaExecution:
    """Test that formulas execute correctly with various inputs."""

    @pytest.mark.parametrize("task_id", ALL_TASKS)
    def test_empty_input(self, task_id, sample_restaurant_meta):
        """Test formula handles empty judgment list."""
        module = importlib.import_module(f"addm.tasks.formulas.{task_id}")

        result = module.compute_ground_truth([], sample_restaurant_meta)

        assert isinstance(result, dict), f"{task_id}: Should return dict"
        assert "verdict" in result, f"{task_id}: Missing verdict"
        assert isinstance(result["verdict"], str), f"{task_id}: Verdict should be string"
        assert len(result["verdict"]) > 0, f"{task_id}: Verdict should not be empty"

    @pytest.mark.parametrize("task_id", ALL_TASKS[:6])  # Test first 6 as sample
    def test_output_schema(self, task_id, sample_restaurant_meta):
        """Test output schema for sample tasks."""
        module = importlib.import_module(f"addm.tasks.formulas.{task_id}")

        result = module.compute_ground_truth([], sample_restaurant_meta)

        # All should have verdict
        assert "verdict" in result
        # Most have risk_score
        if "risk_score" not in result:
            pytest.skip(f"{task_id} doesn't use risk_score")


class TestG1Allergy:
    """Specific tests for G1 allergy tasks (a/b/c/d)."""

    def test_g1a_basic(self, allergy_judgments, thai_restaurant_meta):
        """Test G1a with sample allergy judgments."""
        from addm.tasks.formulas import G1a

        result = G1a.compute_ground_truth(allergy_judgments, thai_restaurant_meta)

        assert "verdict" in result
        # G1a uses risk_score field
        assert any(key.lower().find("score") >= 0 for key in result.keys()), \
            "Should have some score field"

    def test_g1a_empty(self, sample_restaurant_meta):
        """Test G1a with no allergies."""
        from addm.tasks.formulas import G1a

        result = G1a.compute_ground_truth([], sample_restaurant_meta)

        assert result["verdict"] == "Low Risk"

    def test_g1b_has_l15(self):
        """Test that G1b includes L1.5 grouping."""
        from addm.tasks.formulas import G1b

        # Check module has L1.5-related functionality
        # This is indicated by presence of allergen_type handling
        assert hasattr(G1b, "compute_ground_truth")
        # Full L1.5 testing would require more complex setup


class TestG2Social:
    """Tests for G2 social context tasks."""

    def test_g2a_romance(self, romance_judgments, sample_restaurant_meta):
        """Test G2a romance task."""
        from addm.tasks.formulas import G2a

        result = G2a.compute_ground_truth(romance_judgments, sample_restaurant_meta)

        assert "verdict" in result
        # Different formulas use different score field names
        assert isinstance(result, dict)


class TestG4TalentPerformance:
    """Tests for G4 talent and performance tasks."""

    def test_g4a_server(self, server_judgments, sample_restaurant_meta):
        """Test G4a server task."""
        from addm.tasks.formulas import G4a

        result = G4a.compute_ground_truth(server_judgments, sample_restaurant_meta)

        assert "verdict" in result
        assert isinstance(result, dict)

    def test_g4e_kitchen(self, kitchen_judgments, sample_restaurant_meta):
        """Test G4e kitchen task."""
        from addm.tasks.formulas import G4e

        result = G4e.compute_ground_truth(kitchen_judgments, sample_restaurant_meta)

        assert "verdict" in result
        assert isinstance(result, dict)


class TestFormulaConsistency:
    """Test consistency across formula variants."""

    def test_variants_within_topic(self):
        """Test that variants within same topic have compatible structure."""
        # G1 allergy tasks (a/b/c/d)
        from addm.tasks.formulas import G1a, G1b, G1c, G1d

        meta = {"categories": "Thai, Asian Fusion"}
        judgments = []

        results = {
            "G1a": G1a.compute_ground_truth(judgments, meta),
            "G1b": G1b.compute_ground_truth(judgments, meta),
            "G1c": G1c.compute_ground_truth(judgments, meta),
            "G1d": G1d.compute_ground_truth(judgments, meta),
        }

        # All should have verdict
        for task_id, result in results.items():
            assert "verdict" in result, f"{task_id} missing verdict"

    def test_l15_variants_marked_correctly(self):
        """Verify L1.5 variants are correctly identified."""
        # Variants b, d, f, h, j, l should have L1.5
        # This is a structural test - actual L1.5 testing requires judgment data

        l15_task_ids = [f"G1{v}" for v in ["b", "d"]]  # Sample from G1

        for task_id in l15_task_ids:
            # These should handle additional fields like allergen_type
            variant = task_id[-1]
            assert variant in L15_VARIANTS, f"{task_id} should have L1.5"


class TestFormulaRobustness:
    """Test formula robustness with edge cases."""

    @pytest.mark.parametrize("task_id", ["G1a", "G2a", "G3a", "G4a", "G5a", "G6a"])
    def test_missing_restaurant_fields(self, task_id):
        """Test formulas handle missing restaurant metadata gracefully."""
        module = importlib.import_module(f"addm.tasks.formulas.{task_id}")

        # Empty metadata
        result = module.compute_ground_truth([], {})

        assert "verdict" in result
        # Should have some default verdict (varies by group)
        assert isinstance(result["verdict"], str)
        assert len(result["verdict"]) > 0

    @pytest.mark.parametrize("task_id", ["G1a", "G1e", "G1i"])
    def test_partial_judgment_data(self, task_id):
        """Test formulas handle incomplete judgment data."""
        module = importlib.import_module(f"addm.tasks.formulas.{task_id}")

        # Judgment with minimal fields
        partial_judgment = {"is_relevant": True}

        result = module.compute_ground_truth([partial_judgment], {})

        assert "verdict" in result
        # Should not crash even with partial data


class TestVerdictValues:
    """Test that verdicts use correct string values."""

    @pytest.mark.parametrize("task_id", ALL_TASKS)
    def test_verdict_is_valid_string(self, task_id, sample_restaurant_meta):
        """Test that verdict is a non-empty string."""
        module = importlib.import_module(f"addm.tasks.formulas.{task_id}")

        result = module.compute_ground_truth([], sample_restaurant_meta)

        # Different groups use different verdict scales
        # Just verify it's a valid non-empty string
        assert isinstance(result["verdict"], str), f"{task_id}: Verdict should be string"
        assert len(result["verdict"]) > 0, f"{task_id}: Verdict should not be empty"
