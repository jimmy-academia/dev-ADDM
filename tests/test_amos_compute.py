"""Tests for AMOS Phase 2 compute logic.

Tests the FormulaSeedInterpreter's compute operations in isolation,
focusing on field matching, condition evaluation, and count/sum operations.

These tests are designed to expose bugs in the current implementation:
1. Case-insensitive field matching (extraction has lowercase, condition has uppercase)
2. Enum value normalization ("no reaction" should not match incident severity list)
3. Count with mixed case field names
4. Modifier points should not cause false positives on substring matches
"""

import pytest
from typing import Any, Dict, List
from unittest.mock import MagicMock, AsyncMock

from addm.methods.amos.phase2 import FormulaSeedInterpreter
from addm.methods.amos.config import AMOSConfig


# =============================================================================
# Mock LLM Service
# =============================================================================


class MockLLMService:
    """Mock LLM service for testing (no actual API calls)."""

    async def call_async_with_usage(self, messages, context=None):
        """Return empty response - not used in compute tests."""
        return '{"is_relevant": false}', {"prompt_tokens": 0, "completion_tokens": 0}


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_llm():
    """Create mock LLM service."""
    return MockLLMService()


@pytest.fixture
def minimal_seed():
    """Minimal Formula Seed for testing."""
    return {
        "filter": {"keywords": ["allergy", "allergic"]},
        "extract": {
            "fields": [
                {
                    "name": "account_type",
                    "type": "enum",
                    "values": {"firsthand": "Direct experience", "secondhand": "Heard from others"},
                },
                {
                    "name": "incident_severity",
                    "type": "enum",
                    "values": {
                        "none": "No incident",
                        "mild": "Minor reaction",
                        "moderate": "Significant reaction",
                        "severe": "Life-threatening",
                    },
                },
            ]
        },
        "compute": [],
        "output": ["VERDICT"],
    }


@pytest.fixture
def allergy_seed_v2():
    """Formula Seed for allergy V2 policy (scoring-based)."""
    return {
        "filter": {"keywords": ["allergy", "allergic", "reaction"]},
        "extract": {
            "fields": [
                {"name": "account_type", "type": "enum", "values": {"firsthand": "", "secondhand": ""}},
                {
                    "name": "incident_severity",
                    "type": "enum",
                    "values": {"none": "", "mild": "", "moderate": "", "severe": ""},
                },
                {"name": "specific_incident", "type": "str"},
            ]
        },
        "compute": [
            {
                "name": "INCIDENT_COUNT",
                "op": "count",
                "where": {
                    "ACCOUNT_TYPE": "firsthand",
                    "INCIDENT_SEVERITY": ["mild", "moderate", "severe"],
                },
            },
            {
                "name": "INCIDENT_POINTS",
                "op": "sum",
                "expr": """CASE
                    WHEN INCIDENT_SEVERITY = 'severe' THEN 15
                    WHEN INCIDENT_SEVERITY = 'moderate' THEN 8
                    WHEN INCIDENT_SEVERITY = 'mild' THEN 3
                    ELSE 0
                END""",
                "where": {"ACCOUNT_TYPE": "firsthand"},
            },
            {
                "name": "MODIFIER_POINTS",
                "op": "sum",
                "expr": """CASE
                    WHEN specific_incident IN ('False assurance', 'Dismissive staff', 'Cross-contamination') THEN 5
                    ELSE 0
                END""",
            },
            {
                "name": "SCORE",
                "op": "expr",
                "expr": "INCIDENT_POINTS + MODIFIER_POINTS",
            },
            {
                "name": "VERDICT",
                "op": "case",
                "source": "SCORE",
                "rules": [
                    {"when": ">= 15", "then": "Critical Risk"},
                    {"when": ">= 8", "then": "High Risk"},
                    {"when": ">= 3", "then": "Moderate Risk"},
                    {"else": "Low Risk"},
                ],
            },
        ],
        "output": ["VERDICT", "SCORE", "INCIDENT_COUNT"],
    }


# =============================================================================
# Test: Case-Insensitive Field Matching
# =============================================================================


class TestCaseInsensitiveFieldMatching:
    """Test that field matching works regardless of case."""

    def test_matches_condition_lowercase_extraction_uppercase_condition(self, mock_llm, minimal_seed):
        """Extraction has lowercase field names, condition has uppercase.

        This is a common scenario:
        - LLM extractions often return lowercase field names
        - Formula Seed conditions typically use UPPERCASE for clarity

        EXPECTED: Should match (case-insensitive)
        CURRENT BUG: Fails because extraction.get("ACCOUNT_TYPE") returns None
        """
        interpreter = FormulaSeedInterpreter(minimal_seed, mock_llm)

        # Extraction with lowercase field names (typical LLM output)
        extraction = {
            "account_type": "firsthand",
            "incident_severity": "mild",
        }

        # Condition with uppercase field names (from Formula Seed)
        condition = {
            "ACCOUNT_TYPE": "firsthand",
            "INCIDENT_SEVERITY": ["mild", "moderate", "severe"],
        }

        result = interpreter._matches_condition(extraction, condition)

        # This SHOULD match - both fields satisfy the condition
        assert result is True, (
            "Case-insensitive field matching failed. "
            "Extraction has lowercase 'account_type'='firsthand', "
            "condition has uppercase 'ACCOUNT_TYPE'='firsthand'. Should match."
        )

    def test_matches_condition_mixed_case_fields(self, mock_llm, minimal_seed):
        """Test with mixed case in both extraction and condition."""
        interpreter = FormulaSeedInterpreter(minimal_seed, mock_llm)

        extraction = {
            "Account_Type": "firsthand",
            "incident_severity": "moderate",
        }

        condition = {
            "account_type": "firsthand",
            "INCIDENT_SEVERITY": ["mild", "moderate"],
        }

        result = interpreter._matches_condition(extraction, condition)
        assert result is True, "Mixed case field matching should work"

    def test_field_equals_format_case_insensitive(self, mock_llm, minimal_seed):
        """Test field/equals condition format with case mismatch."""
        interpreter = FormulaSeedInterpreter(minimal_seed, mock_llm)

        extraction = {"account_type": "firsthand"}

        condition = {"field": "ACCOUNT_TYPE", "equals": "firsthand"}

        result = interpreter._matches_condition(extraction, condition)
        assert result is True, "field/equals format should be case-insensitive"


# =============================================================================
# Test: Enum Value Normalization
# =============================================================================


class TestEnumValueNormalization:
    """Test that enum values are normalized correctly."""

    def test_no_reaction_not_in_severity_list(self, mock_llm, minimal_seed):
        """'no reaction' should NOT match severity list ['mild', 'moderate', 'severe'].

        This tests that 'none' or 'no reaction' extractions are correctly
        excluded from incident counts.
        """
        interpreter = FormulaSeedInterpreter(minimal_seed, mock_llm)

        extraction = {
            "account_type": "firsthand",
            "incident_severity": "no reaction",
        }

        condition = {
            "account_type": "firsthand",
            "incident_severity": ["mild", "moderate", "severe"],
        }

        result = interpreter._matches_condition(extraction, condition)
        assert result is False, (
            "'no reaction' should NOT match ['mild', 'moderate', 'severe']. "
            "This indicates no actual incident occurred."
        )

    def test_none_severity_not_in_list(self, mock_llm, minimal_seed):
        """'none' severity should NOT match incident severity list."""
        interpreter = FormulaSeedInterpreter(minimal_seed, mock_llm)

        extraction = {
            "account_type": "firsthand",
            "incident_severity": "none",
        }

        condition = {
            "account_type": "firsthand",
            "incident_severity": ["mild", "moderate", "severe"],
        }

        result = interpreter._matches_condition(extraction, condition)
        assert result is False, "'none' should NOT match incident severity list"

    def test_mild_severity_matches_list(self, mock_llm, minimal_seed):
        """'mild' severity SHOULD match when in the list."""
        interpreter = FormulaSeedInterpreter(minimal_seed, mock_llm)

        extraction = {
            "account_type": "firsthand",
            "incident_severity": "mild",
        }

        condition = {
            "account_type": "firsthand",
            "incident_severity": ["mild", "moderate", "severe"],
        }

        result = interpreter._matches_condition(extraction, condition)
        assert result is True, "'mild' should match ['mild', 'moderate', 'severe']"


# =============================================================================
# Test: Count with Mixed Case
# =============================================================================


class TestCountWithMixedCase:
    """Test count operations with mixed case field names."""

    def test_count_with_uppercase_condition_lowercase_extraction(self, mock_llm, allergy_seed_v2):
        """Count should work when extractions have lowercase, conditions have uppercase."""
        interpreter = FormulaSeedInterpreter(allergy_seed_v2, mock_llm)

        # Simulate extractions with lowercase field names
        interpreter._extractions = [
            {
                "review_id": "r1",
                "is_relevant": True,
                "account_type": "firsthand",
                "incident_severity": "mild",
            },
            {
                "review_id": "r2",
                "is_relevant": True,
                "account_type": "firsthand",
                "incident_severity": "moderate",
            },
            {
                "review_id": "r3",
                "is_relevant": True,
                "account_type": "secondhand",  # Should NOT count
                "incident_severity": "severe",
            },
            {
                "review_id": "r4",
                "is_relevant": True,
                "account_type": "firsthand",
                "incident_severity": "none",  # Should NOT count
            },
        ]

        # Count operation with uppercase condition
        op_def = {
            "name": "INCIDENT_COUNT",
            "op": "count",
            "where": {
                "ACCOUNT_TYPE": "firsthand",
                "INCIDENT_SEVERITY": ["mild", "moderate", "severe"],
            },
        }

        count = interpreter._compute_count(op_def)

        # Should count r1 and r2 (firsthand + valid severity)
        # r3 is secondhand, r4 is none severity
        assert count == 2, (
            f"Expected 2 incidents (r1=mild, r2=moderate), got {count}. "
            "Case-insensitive matching may have failed."
        )

    def test_count_with_no_incidents(self, mock_llm, allergy_seed_v2):
        """Count should return 0 when no extractions match."""
        interpreter = FormulaSeedInterpreter(allergy_seed_v2, mock_llm)

        interpreter._extractions = [
            {
                "review_id": "r1",
                "account_type": "secondhand",
                "incident_severity": "severe",
            },
            {
                "review_id": "r2",
                "account_type": "firsthand",
                "incident_severity": "none",
            },
        ]

        op_def = {
            "name": "INCIDENT_COUNT",
            "op": "count",
            "where": {
                "ACCOUNT_TYPE": "firsthand",
                "INCIDENT_SEVERITY": ["mild", "moderate", "severe"],
            },
        }

        count = interpreter._compute_count(op_def)
        assert count == 0, f"Expected 0 incidents, got {count}"


# =============================================================================
# Test: Modifier Points (No False Positives)
# =============================================================================


class TestModifierPointsNoFalsePositives:
    """Test that modifier points don't match on substrings."""

    def test_modifier_points_exact_match_only(self, mock_llm, allergy_seed_v2):
        """Modifier points should only match exact values, not substrings.

        BUG: 'mango allergy accommodations blocked' might incorrectly match
        'False assurance' or 'Dismissive staff' due to substring matching.
        """
        interpreter = FormulaSeedInterpreter(allergy_seed_v2, mock_llm)

        # Extraction with descriptive text that should NOT match modifiers
        extraction = {
            "review_id": "r1",
            "account_type": "firsthand",
            "incident_severity": "mild",
            "specific_incident": "mango allergy accommodations blocked",
        }

        # The sum expression checks for specific modifier values
        expr = """CASE
            WHEN specific_incident IN ('False assurance', 'Dismissive staff', 'Cross-contamination') THEN 5
            ELSE 0
        END"""

        points = interpreter._eval_sql_case_expr(expr, extraction)

        assert points == 0, (
            f"Expected 0 modifier points for 'mango allergy accommodations blocked', "
            f"got {points}. This should NOT match 'False assurance' or other modifiers."
        )

    def test_modifier_points_exact_match_works(self, mock_llm, allergy_seed_v2):
        """Modifier points should match when exact value is present."""
        interpreter = FormulaSeedInterpreter(allergy_seed_v2, mock_llm)

        extraction = {
            "review_id": "r1",
            "account_type": "firsthand",
            "incident_severity": "mild",
            "specific_incident": "False assurance",
        }

        expr = """CASE
            WHEN specific_incident IN ('False assurance', 'Dismissive staff', 'Cross-contamination') THEN 5
            ELSE 0
        END"""

        points = interpreter._eval_sql_case_expr(expr, extraction)

        assert points == 5, f"Expected 5 modifier points for 'False assurance', got {points}"

    def test_modifier_points_case_insensitive_value(self, mock_llm, allergy_seed_v2):
        """Modifier value matching should be case-insensitive."""
        interpreter = FormulaSeedInterpreter(allergy_seed_v2, mock_llm)

        extraction = {
            "review_id": "r1",
            "account_type": "firsthand",
            "incident_severity": "mild",
            "specific_incident": "false assurance",  # lowercase
        }

        expr = """CASE
            WHEN specific_incident IN ('False assurance', 'Dismissive staff') THEN 5
            ELSE 0
        END"""

        points = interpreter._eval_sql_case_expr(expr, extraction)

        assert points == 5, (
            f"Expected 5 modifier points for 'false assurance' (case-insensitive), got {points}"
        )


# =============================================================================
# Test: SQL CASE Expression Evaluation
# =============================================================================


class TestSQLCaseExpressionEval:
    """Test SQL CASE WHEN ... THEN ... END expression evaluation."""

    def test_case_expression_basic(self, mock_llm, minimal_seed):
        """Test basic CASE expression evaluation."""
        interpreter = FormulaSeedInterpreter(minimal_seed, mock_llm)

        extraction = {"incident_severity": "moderate"}

        expr = """CASE
            WHEN INCIDENT_SEVERITY = 'severe' THEN 15
            WHEN INCIDENT_SEVERITY = 'moderate' THEN 8
            WHEN INCIDENT_SEVERITY = 'mild' THEN 3
            ELSE 0
        END"""

        result = interpreter._eval_sql_case_expr(expr, extraction)
        assert result == 8, f"Expected 8 for 'moderate', got {result}"

    def test_case_expression_lowercase_field(self, mock_llm, minimal_seed):
        """Test CASE expression with lowercase extraction field name."""
        interpreter = FormulaSeedInterpreter(minimal_seed, mock_llm)

        extraction = {"incident_severity": "severe"}  # lowercase field name

        # Expression uses uppercase field name
        expr = """CASE
            WHEN INCIDENT_SEVERITY = 'severe' THEN 15
            WHEN INCIDENT_SEVERITY = 'moderate' THEN 8
            ELSE 0
        END"""

        result = interpreter._eval_sql_case_expr(expr, extraction)
        assert result == 15, (
            f"Expected 15 for 'severe' with case-insensitive field lookup, got {result}"
        )

    def test_case_expression_else_fallback(self, mock_llm, minimal_seed):
        """Test CASE expression falls back to ELSE clause."""
        interpreter = FormulaSeedInterpreter(minimal_seed, mock_llm)

        extraction = {"incident_severity": "none"}

        expr = """CASE
            WHEN INCIDENT_SEVERITY = 'severe' THEN 15
            WHEN INCIDENT_SEVERITY = 'moderate' THEN 8
            ELSE 0
        END"""

        result = interpreter._eval_sql_case_expr(expr, extraction)
        assert result == 0, f"Expected 0 for 'none' (ELSE clause), got {result}"


# =============================================================================
# Test: Full Compute Pipeline
# =============================================================================


class TestFullComputePipeline:
    """Integration tests for the full compute pipeline."""

    def test_compute_verdict_with_lowercase_extractions(self, mock_llm, allergy_seed_v2):
        """Test full compute pipeline with lowercase extraction field names."""
        interpreter = FormulaSeedInterpreter(allergy_seed_v2, mock_llm)

        # Simulate extractions with lowercase field names
        interpreter._extractions = [
            {
                "review_id": "r1",
                "is_relevant": True,
                "account_type": "firsthand",
                "incident_severity": "moderate",
                "specific_incident": "allergic reaction to peanuts",
            },
        ]

        business = {"name": "Test Restaurant", "categories": "Thai, Asian"}

        interpreter._execute_compute(business)

        # Check computed values
        assert interpreter._namespace.get("INCIDENT_COUNT") == 1, (
            f"Expected INCIDENT_COUNT=1, got {interpreter._namespace.get('INCIDENT_COUNT')}"
        )
        assert interpreter._namespace.get("INCIDENT_POINTS") == 8, (
            f"Expected INCIDENT_POINTS=8 (moderate=8), got {interpreter._namespace.get('INCIDENT_POINTS')}"
        )
        assert interpreter._namespace.get("MODIFIER_POINTS") == 0, (
            f"Expected MODIFIER_POINTS=0, got {interpreter._namespace.get('MODIFIER_POINTS')}"
        )
        assert interpreter._namespace.get("SCORE") == 8, (
            f"Expected SCORE=8, got {interpreter._namespace.get('SCORE')}"
        )
        assert interpreter._namespace.get("VERDICT") == "High Risk", (
            f"Expected VERDICT='High Risk' (score >= 8), got {interpreter._namespace.get('VERDICT')}"
        )

    def test_compute_no_incidents_low_risk(self, mock_llm, allergy_seed_v2):
        """Test that no incidents results in Low Risk verdict."""
        interpreter = FormulaSeedInterpreter(allergy_seed_v2, mock_llm)

        interpreter._extractions = []  # No relevant extractions

        business = {"name": "Test Restaurant", "categories": "American"}

        interpreter._execute_compute(business)

        assert interpreter._namespace.get("INCIDENT_COUNT") == 0
        assert interpreter._namespace.get("SCORE") == 0
        assert interpreter._namespace.get("VERDICT") == "Low Risk"

    def test_compute_critical_risk_with_modifier(self, mock_llm, allergy_seed_v2):
        """Test Critical Risk when score >= 15 with modifier points."""
        interpreter = FormulaSeedInterpreter(allergy_seed_v2, mock_llm)

        interpreter._extractions = [
            {
                "review_id": "r1",
                "is_relevant": True,
                "account_type": "firsthand",
                "incident_severity": "moderate",  # 8 points
                "specific_incident": "False assurance",  # +5 modifier
            },
            {
                "review_id": "r2",
                "is_relevant": True,
                "account_type": "firsthand",
                "incident_severity": "mild",  # 3 points
                "specific_incident": "normal incident",
            },
        ]

        business = {"name": "Test Restaurant"}

        interpreter._execute_compute(business)

        # 8 (moderate) + 3 (mild) + 5 (False assurance) = 16
        expected_score = 16
        assert interpreter._namespace.get("SCORE") == expected_score, (
            f"Expected SCORE={expected_score}, got {interpreter._namespace.get('SCORE')}"
        )
        assert interpreter._namespace.get("VERDICT") == "Critical Risk"


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_extractions_list(self, mock_llm, allergy_seed_v2):
        """Test compute with empty extractions list."""
        interpreter = FormulaSeedInterpreter(allergy_seed_v2, mock_llm)
        interpreter._extractions = []

        business = {}
        interpreter._execute_compute(business)

        assert interpreter._namespace.get("INCIDENT_COUNT") == 0
        assert interpreter._namespace.get("VERDICT") == "Low Risk"

    def test_extraction_with_missing_fields(self, mock_llm, allergy_seed_v2):
        """Test extraction with some fields missing."""
        interpreter = FormulaSeedInterpreter(allergy_seed_v2, mock_llm)

        interpreter._extractions = [
            {
                "review_id": "r1",
                "is_relevant": True,
                "account_type": "firsthand",
                # incident_severity missing
            },
        ]

        op_def = {
            "name": "INCIDENT_COUNT",
            "op": "count",
            "where": {
                "account_type": "firsthand",
                "incident_severity": ["mild", "moderate", "severe"],
            },
        }

        count = interpreter._compute_count(op_def)

        # Should not count extraction with missing severity field
        assert count == 0, "Extraction with missing field should not match condition"

    def test_none_value_vs_missing_field(self, mock_llm, allergy_seed_v2):
        """Test distinction between None value and missing field."""
        interpreter = FormulaSeedInterpreter(allergy_seed_v2, mock_llm)

        # Extraction with explicit None value
        extraction_none = {
            "account_type": "firsthand",
            "incident_severity": None,
        }

        # Extraction with missing field
        extraction_missing = {
            "account_type": "firsthand",
        }

        condition = {
            "account_type": "firsthand",
            "incident_severity": ["mild", "moderate", "severe"],
        }

        # Both should NOT match
        assert interpreter._matches_condition(extraction_none, condition) is False
        assert interpreter._matches_condition(extraction_missing, condition) is False
