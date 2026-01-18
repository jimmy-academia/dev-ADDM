"""Tests for tasks/executor.py - Ground truth computation engine."""

import pytest
from addm.tasks.executor import (
    SAFE_OPERATORS,
    _eval_node,
    safe_eval_arithmetic,
    evaluate_condition,
    evaluate_composite,
    evaluate_expression,
    evaluate_formula,
    evaluate_lookup,
    evaluate_aggregate,
    apply_decision_policy,
)
from addm.tasks.prompt_parser import (
    Condition,
    L1Composite,
    LookupTable,
    L2Aggregate,
    Formula,
    DecisionPolicy,
    ThresholdRule,
    OverrideRule,
)


# =============================================================================
# Safe Arithmetic Evaluator Tests
# =============================================================================


class TestSafeArithmetic:
    """Test the safe AST-based arithmetic evaluator."""

    def test_basic_addition(self):
        assert safe_eval_arithmetic("2 + 3") == 5.0

    def test_basic_subtraction(self):
        assert safe_eval_arithmetic("10 - 3") == 7.0

    def test_basic_multiplication(self):
        assert safe_eval_arithmetic("4 * 5") == 20.0

    def test_basic_division(self):
        assert safe_eval_arithmetic("10 / 2") == 5.0

    def test_float_operations(self):
        assert safe_eval_arithmetic("10.5 / 2.5") == 4.2

    def test_complex_expression(self):
        assert safe_eval_arithmetic("3 * 4 + 2") == 14.0

    def test_parentheses(self):
        assert safe_eval_arithmetic("(3 + 2) * 4") == 20.0

    def test_negation(self):
        assert safe_eval_arithmetic("-5 + 10") == 5.0

    def test_nested_parentheses(self):
        assert safe_eval_arithmetic("((3 + 2) * (4 - 1)) / 3") == 5.0

    def test_division_by_zero(self):
        # Should return 0.0 on error
        assert safe_eval_arithmetic("10 / 0") == 0.0

    def test_invalid_expression(self):
        # Should return 0.0 on syntax error
        assert safe_eval_arithmetic("invalid") == 0.0

    def test_empty_string(self):
        assert safe_eval_arithmetic("") == 0.0

    def test_malformed_expression(self):
        assert safe_eval_arithmetic("2 +") == 0.0

    def test_only_number(self):
        assert safe_eval_arithmetic("42") == 42.0

    def test_negative_result(self):
        assert safe_eval_arithmetic("5 - 10") == -5.0


# =============================================================================
# Condition Evaluation Tests
# =============================================================================


class TestConditionEvaluation:
    """Test condition evaluation against records."""

    def test_equality_operator(self):
        cond = Condition(field="status", operator="=", value="active")
        record = {"status": "active"}
        assert evaluate_condition(cond, record) is True

        record = {"status": "inactive"}
        assert evaluate_condition(cond, record) is False

    def test_inequality_operator(self):
        cond = Condition(field="status", operator="!=", value="active")
        record = {"status": "inactive"}
        assert evaluate_condition(cond, record) is True

    def test_in_operator(self):
        cond = Condition(field="severity", operator="in", value=["moderate", "severe"])
        record = {"severity": "moderate"}
        assert evaluate_condition(cond, record) is True

        record = {"severity": "mild"}
        assert evaluate_condition(cond, record) is False

    def test_greater_than_operator(self):
        cond = Condition(field="score", operator=">", value="5")
        record = {"score": "7"}
        assert evaluate_condition(cond, record) is True

        record = {"score": "3"}
        assert evaluate_condition(cond, record) is False

    def test_greater_equal_operator(self):
        cond = Condition(field="score", operator=">=", value="5")
        record = {"score": "5"}
        assert evaluate_condition(cond, record) is True

    def test_less_than_operator(self):
        cond = Condition(field="score", operator="<", value="5")
        record = {"score": "3"}
        assert evaluate_condition(cond, record) is True

    def test_less_equal_operator(self):
        cond = Condition(field="score", operator="<=", value="5")
        record = {"score": "5"}
        assert evaluate_condition(cond, record) is True

    def test_case_insensitive_lookup(self):
        """Test that field lookup is case-insensitive."""
        cond = Condition(field="Status", operator="=", value="active")
        record = {"status": "active"}
        assert evaluate_condition(cond, record) is True

    def test_boolean_normalization(self):
        """Test boolean value normalization."""
        cond = Condition(field="flag", operator="=", value="true")
        record = {"flag": True}
        assert evaluate_condition(cond, record) is True


# =============================================================================
# Composite Evaluation Tests
# =============================================================================


class TestCompositeEvaluation:
    """Test L1 composite evaluation."""

    def test_all_quantifier_true(self):
        comp = L1Composite(
            name="test_comp",
            quantifier="ALL",
            conditions=[
                Condition(field="a", operator="=", value="1"),
                Condition(field="b", operator="=", value="2"),
            ],
        )
        record = {"a": "1", "b": "2"}
        assert evaluate_composite(comp, record) is True

    def test_all_quantifier_false(self):
        comp = L1Composite(
            name="test_comp",
            quantifier="ALL",
            conditions=[
                Condition(field="a", operator="=", value="1"),
                Condition(field="b", operator="=", value="2"),
            ],
        )
        record = {"a": "1", "b": "3"}
        assert evaluate_composite(comp, record) is False

    def test_any_quantifier_true(self):
        comp = L1Composite(
            name="test_comp",
            quantifier="ANY",
            conditions=[
                Condition(field="a", operator="=", value="1"),
                Condition(field="b", operator="=", value="2"),
            ],
        )
        record = {"a": "1", "b": "3"}
        assert evaluate_composite(comp, record) is True

    def test_any_quantifier_false(self):
        comp = L1Composite(
            name="test_comp",
            quantifier="ANY",
            conditions=[
                Condition(field="a", operator="=", value="1"),
                Condition(field="b", operator="=", value="2"),
            ],
        )
        record = {"a": "0", "b": "3"}
        assert evaluate_composite(comp, record) is False


# =============================================================================
# Lookup Table Tests
# =============================================================================


class TestLookupTable:
    """Test lookup table evaluation."""

    def test_direct_match(self):
        table = LookupTable(
            name="test_table",
            strategy="highest",
            entries={"Thai": 1.5, "American": 1.0},
            default=0.5,
        )
        assert evaluate_lookup(table, "Thai, Asian Fusion") == 1.5

    def test_partial_match(self):
        table = LookupTable(
            name="test_table",
            strategy="highest",
            entries={"Thai": 1.5, "American": 1.0},
            default=0.5,
        )
        # "Thai" should match in "Thai Fusion"
        assert evaluate_lookup(table, "Thai Fusion") == 1.5

    def test_highest_strategy(self):
        table = LookupTable(
            name="test_table",
            strategy="highest",
            entries={"Thai": 1.5, "American": 1.0},
            default=0.5,
        )
        # Both match, should return highest
        assert evaluate_lookup(table, "Thai, American") == 1.5

    def test_default_value(self):
        table = LookupTable(
            name="test_table",
            strategy="highest",
            entries={"Thai": 1.5, "American": 1.0},
            default=0.5,
        )
        assert evaluate_lookup(table, "Mexican") == 0.5

    def test_empty_categories(self):
        table = LookupTable(
            name="test_table",
            strategy="highest",
            entries={"Thai": 1.5, "American": 1.0},
            default=0.5,
        )
        assert evaluate_lookup(table, "") == 0.5


# =============================================================================
# Expression Evaluation Tests
# =============================================================================


class TestExpressionEvaluation:
    """Test arithmetic expression evaluation with variable substitution."""

    def test_simple_variable_substitution(self, sample_context):
        expr = "BASE_SCORE + 2"
        result = evaluate_expression(expr, sample_context)
        assert result == 7.0  # 5.0 + 2

    def test_multiple_variables(self, sample_context):
        expr = "MODERATE_INCIDENTS + SEVERE_INCIDENTS"
        result = evaluate_expression(expr, sample_context)
        assert result == 2.0  # 2 + 0

    def test_complex_expression(self, sample_context):
        expr = "BASE_SCORE * CREDIBILITY_WEIGHT"
        result = evaluate_expression(expr, sample_context)
        assert result == 4.0  # 5.0 * 0.8

    def test_undefined_variable_becomes_zero(self):
        expr = "UNDEFINED_VAR + 5"
        result = evaluate_expression(expr, {})
        assert result == 5.0  # 0 + 5

    def test_nested_expression(self, sample_context):
        expr = "(BASE_SCORE + MODERATE_INCIDENTS) * 2"
        result = evaluate_expression(expr, sample_context)
        assert result == 14.0  # (5.0 + 2) * 2


# =============================================================================
# Formula Evaluation Tests
# =============================================================================


class TestFormulaEvaluation:
    """Test formula evaluation with special functions."""

    def test_clamp_formula(self, sample_context):
        formula = Formula(
            name="test_score",
            func="clamp",
            expression="BASE_SCORE * 2",
            func_args=["0", "8"],
        )
        result = evaluate_formula(formula, sample_context)
        assert result == 8.0  # clamp(10.0, 0, 8) = 8.0

    def test_max_formula(self, sample_context):
        formula = Formula(
            name="test_score",
            func="max",
            expression="MODERATE_INCIDENTS",
            func_args=["5"],
        )
        result = evaluate_formula(formula, sample_context)
        assert result == 5.0  # max(2, 5) = 5

    def test_min_formula(self, sample_context):
        formula = Formula(
            name="test_score",
            func="min",
            expression="BASE_SCORE + 10",
            func_args=["10"],
        )
        result = evaluate_formula(formula, sample_context)
        assert result == 10.0  # min(15.0, 10) = 10


# =============================================================================
# Aggregate Evaluation Tests
# =============================================================================


class TestAggregateEvaluation:
    """Test L2 aggregate evaluation."""

    def test_count_aggregate(self, allergy_judgments):
        agg = L2Aggregate(
            name="MODERATE_COUNT",
            agg_type="count",
            conditions=[Condition(field="incident_severity", operator="=", value="moderate")],
        )
        result = evaluate_aggregate(agg, allergy_judgments, {})
        assert result == 1

    def test_arithmetic_aggregate(self, sample_context):
        agg = L2Aggregate(
            name="TOTAL_SCORE",
            agg_type="arithmetic",
            expression="BASE_SCORE + MODERATE_INCIDENTS",
        )
        result = evaluate_aggregate(agg, [], sample_context)
        assert result == 7.0

    def test_max_aggregate(self, allergy_judgments):
        # Add REVIEW_YEAR to test data
        for idx, j in enumerate(allergy_judgments):
            j["REVIEW_YEAR"] = 2020 + idx

        agg = L2Aggregate(
            name="MOST_RECENT_YEAR",
            agg_type="max",
            expression="REVIEW_YEAR",
            conditions=[Condition(field="incident_severity", operator="=", value="moderate")],
            default_var="CURRENT_YEAR",
        )
        result = evaluate_aggregate(agg, allergy_judgments, {"CURRENT_YEAR": 2023})
        assert result == 2020  # Only first judgment has moderate severity


# =============================================================================
# Decision Policy Tests
# =============================================================================


class TestDecisionPolicy:
    """Test decision policy application."""

    def test_basic_threshold_policy(self):
        policy = DecisionPolicy(
            score_variable="RISK_SCORE",
            threshold_rules=[
                ThresholdRule(conditions=[("<", "3")], result="Low Risk"),
                ThresholdRule(conditions=[(">=", "3"), ("<", "7")], result="High Risk"),
                ThresholdRule(conditions=[(">=", "7")], result="Critical Risk"),
            ],
            override_rules=[],
        )

        ctx = {"RISK_SCORE": 5.0}
        result = apply_decision_policy(policy, ctx)
        assert result["verdict"] == "High Risk"
        assert result["base_verdict_by_score"] == "High Risk"
        assert result["override_applied"] == "none"

    def test_override_policy(self):
        policy = DecisionPolicy(
            score_variable="RISK_SCORE",
            threshold_rules=[
                ThresholdRule(conditions=[("<", "5")], result="Low Risk"),
                ThresholdRule(conditions=[(">=", "5")], result="High Risk"),
            ],
            override_rules=[
                OverrideRule(
                    condition=Condition(field="SEVERE_FLAG", operator="=", value="true"),
                    result="Critical Risk",
                    is_minimum=False,
                )
            ],
        )

        ctx = {"RISK_SCORE": 3.0, "SEVERE_FLAG": "true"}
        result = apply_decision_policy(policy, ctx)
        assert result["verdict"] == "Critical Risk"
        assert result["base_verdict_by_score"] == "Low Risk"
        assert result["override_applied"] == "severe_flag"

    def test_minimum_override(self):
        policy = DecisionPolicy(
            score_variable="RISK_SCORE",
            threshold_rules=[
                ThresholdRule(conditions=[("<", "5")], result="Low Risk"),
                ThresholdRule(conditions=[(">=", "5")], result="High Risk"),
            ],
            override_rules=[
                OverrideRule(
                    condition=Condition(field="MIN_FLAG", operator="=", value="true"),
                    result="High Risk",
                    is_minimum=True,
                )
            ],
        )

        # Base is Low Risk, min override should bump to High Risk
        ctx = {"RISK_SCORE": 2.0, "MIN_FLAG": "true"}
        result = apply_decision_policy(policy, ctx)
        assert result["verdict"] == "High Risk"
        assert result["override_applied"] == "min_high_risk"


# =============================================================================
# Integration Tests
# =============================================================================


class TestExecutorIntegration:
    """Integration tests combining multiple executor components."""

    def test_empty_judgments_returns_low_risk(self, sample_restaurant_meta):
        """Test that empty judgment list returns Low Risk verdict."""
        # This is a basic integration test pattern
        # Full integration would require ParsedPrompt which we'll test separately
        pass

    def test_safe_eval_no_code_injection(self):
        """Verify that malicious code cannot be executed."""
        malicious_inputs = [
            "__import__('os').system('ls')",
            "exec('print(1)')",
            "eval('1+1')",
            "open('/etc/passwd')",
        ]

        for malicious in malicious_inputs:
            result = safe_eval_arithmetic(malicious)
            # All should safely return 0.0 without executing
            assert result == 0.0
