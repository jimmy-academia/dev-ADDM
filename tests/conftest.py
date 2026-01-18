"""Shared pytest fixtures for ADDM testing."""

import pytest
from typing import Any, Dict, List, Tuple


# =============================================================================
# Test Data Constants
# =============================================================================

ALL_TASKS = [f"G{g}{t}" for g in range(1, 7) for t in "abcdefghijkl"]

TOPICS = {
    # G1: Health & Safety
    "G1_allergy": ["G1a", "G1b", "G1c", "G1d"],
    "G1_dietary": ["G1e", "G1f", "G1g", "G1h"],
    "G1_hygiene": ["G1i", "G1j", "G1k", "G1l"],
    # G2: Social Context
    "G2_romance": ["G2a", "G2b", "G2c", "G2d"],
    "G2_business": ["G2e", "G2f", "G2g", "G2h"],
    "G2_group": ["G2i", "G2j", "G2k", "G2l"],
    # G3: Economic Value
    "G3_price_worth": ["G3a", "G3b", "G3c", "G3d"],
    "G3_hidden_costs": ["G3e", "G3f", "G3g", "G3h"],
    "G3_time_value": ["G3i", "G3j", "G3k", "G3l"],
    # G4: Talent & Performance
    "G4_server": ["G4a", "G4b", "G4c", "G4d"],
    "G4_kitchen": ["G4e", "G4f", "G4g", "G4h"],
    "G4_environment": ["G4i", "G4j", "G4k", "G4l"],
    # G5: Operational Efficiency
    "G5_capacity": ["G5a", "G5b", "G5c", "G5d"],
    "G5_execution": ["G5e", "G5f", "G5g", "G5h"],
    "G5_consistency": ["G5i", "G5j", "G5k", "G5l"],
    # G6: Competitive Strategy
    "G6_uniqueness": ["G6a", "G6b", "G6c", "G6d"],
    "G6_comparison": ["G6e", "G6f", "G6g", "G6h"],
    "G6_loyalty": ["G6i", "G6j", "G6k", "G6l"],
}

L15_VARIANTS = {"b", "d", "f", "h", "j", "l"}


# =============================================================================
# Restaurant Metadata Fixtures
# =============================================================================


@pytest.fixture
def sample_restaurant_meta() -> Dict[str, Any]:
    """Sample restaurant metadata."""
    return {
        "business_id": "rest_001",
        "name": "Test Restaurant",
        "categories": "American, Casual Dining",
        "review_count": 150,
    }


@pytest.fixture
def thai_restaurant_meta() -> Dict[str, Any]:
    """Thai restaurant metadata for allergy testing."""
    return {
        "business_id": "thai_001",
        "name": "Thai Fusion Bistro",
        "categories": "Thai, Asian Fusion",
        "review_count": 200,
    }


# =============================================================================
# Judgment Fixtures by Topic
# =============================================================================


@pytest.fixture
def allergy_judgments() -> List[Dict[str, Any]]:
    """Sample allergy-related judgments."""
    return [
        {
            "is_allergy_related": True,
            "incident_severity": "moderate",
            "account_type": "firsthand",
            "assurance_claim": "false",
            "staff_response": "accommodated",
            "allergen_type": "peanut",
            "date": "2021-05-15",
        },
        {
            "is_allergy_related": True,
            "incident_severity": "mild",
            "account_type": "firsthand",
            "assurance_claim": "false",
            "staff_response": "none",
            "allergen_type": "dairy",
            "date": "2020-03-10",
        },
    ]


@pytest.fixture
def dietary_judgments() -> List[Dict[str, Any]]:
    """Sample dietary-related judgments."""
    return [
        {
            "is_dietary_related": True,
            "diet_type": "vegetarian",
            "accommodation_quality": "excellent",
            "menu_clarity": "clear",
            "staff_knowledge": "knowledgeable",
            "diet_subtype": "vegan",
        },
    ]


@pytest.fixture
def hygiene_judgments() -> List[Dict[str, Any]]:
    """Sample hygiene-related judgments."""
    return [
        {
            "is_hygiene_related": True,
            "account_type": "firsthand",
            "issue_type": "cleanliness",
            "issue_severity": "moderate",
            "staff_response": "none",
        },
        {
            "is_hygiene_related": True,
            "account_type": "firsthand",
            "issue_type": "pest",
            "issue_severity": "severe",
            "staff_response": "dismissed",
        },
    ]


@pytest.fixture
def romance_judgments() -> List[Dict[str, Any]]:
    """Sample romance-related judgments."""
    return [
        {
            "is_romance_related": True,
            "occasion_type": "date_night",
            "ambiance_rating": "excellent",
            "privacy_level": "intimate",
            "noise_level": "quiet",
            "romantic_elements": "present",
            "experience_subtype": "anniversary",
        },
    ]


@pytest.fixture
def business_judgments() -> List[Dict[str, Any]]:
    """Sample business-related judgments."""
    return [
        {
            "is_business_related": True,
            "meeting_type": "client_meeting",
            "professionalism": "high",
            "noise_suitability": "appropriate",
            "service_timing": "efficient",
            "business_context": "formal",
        },
    ]


@pytest.fixture
def server_judgments() -> List[Dict[str, Any]]:
    """Sample server-related judgments."""
    return [
        {
            "is_server_related": True,
            "attentiveness": "excellent",
            "knowledge": "high",
            "friendliness": "warm",
            "professionalism": "high",
            "server_strength": "knowledge",
        },
    ]


@pytest.fixture
def kitchen_judgments() -> List[Dict[str, Any]]:
    """Sample kitchen-related judgments."""
    return [
        {
            "is_kitchen_related": True,
            "food_quality": "excellent",
            "consistency": "high",
            "creativity": "innovative",
            "execution": "flawless",
            "culinary_focus": "technique",
        },
    ]


# =============================================================================
# Executor and Context Fixtures
# =============================================================================


@pytest.fixture
def sample_context() -> Dict[str, Any]:
    """Sample context for expression evaluation."""
    return {
        "CREDIBILITY_WEIGHT": 0.8,
        "BASE_SCORE": 5.0,
        "MODERATE_INCIDENTS": 2,
        "SEVERE_INCIDENTS": 0,
        "TOTAL_REVIEWS": 10,
    }


@pytest.fixture
def empty_context() -> Dict[str, Any]:
    """Empty context for testing defaults."""
    return {}


# =============================================================================
# Usage Tracking Fixtures
# =============================================================================


@pytest.fixture
def sample_usage() -> Dict[str, Any]:
    """Sample usage data for a single LLM call."""
    return {
        "prompt_tokens": 150,
        "completion_tokens": 50,
        "total_tokens": 200,
        "model": "gpt-5-nano",
    }


@pytest.fixture
def multiple_usages() -> List[Dict[str, Any]]:
    """Multiple usage entries for accumulation testing."""
    return [
        {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "model": "gpt-5-nano",
        },
        {
            "prompt_tokens": 200,
            "completion_tokens": 75,
            "total_tokens": 275,
            "model": "gpt-5-nano",
        },
        {
            "prompt_tokens": 150,
            "completion_tokens": 100,
            "total_tokens": 250,
            "model": "gpt-5-nano",
        },
    ]


# =============================================================================
# Metrics Fixtures
# =============================================================================


@pytest.fixture
def binary_predictions() -> List[int]:
    """Sample binary predictions (0/1)."""
    return [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]


@pytest.fixture
def binary_labels() -> List[int]:
    """Sample binary ground truth labels."""
    return [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]


@pytest.fixture
def probability_scores() -> List[float]:
    """Sample probability scores for AUPRC calculation."""
    return [0.9, 0.1, 0.85, 0.7, 0.3, 0.95, 0.4, 0.2, 0.88, 0.6]


# =============================================================================
# Helper Functions
# =============================================================================


def get_test_data_for_topic(topic_name: str) -> Tuple[List[Dict], Dict]:
    """
    Generate synthetic test data appropriate for a topic.

    Returns:
        Tuple of (judgments, restaurant_meta)
    """
    restaurant_meta = {"categories": "American, Casual Dining", "name": "Test Restaurant"}

    # Topic-specific judgments (extracted from verify_formulas.py)
    topic_to_judgments = {
        "allergy": [
            {
                "is_allergy_related": True,
                "incident_severity": "moderate",
                "account_type": "firsthand",
                "assurance_claim": "false",
                "staff_response": "accommodated",
                "allergen_type": "peanut",
                "date": "2021-05-15",
            },
            {
                "is_allergy_related": True,
                "incident_severity": "mild",
                "account_type": "firsthand",
                "assurance_claim": "false",
                "staff_response": "none",
                "allergen_type": "dairy",
                "date": "2020-03-10",
            },
        ],
        "dietary": [
            {
                "is_dietary_related": True,
                "diet_type": "vegetarian",
                "accommodation_quality": "excellent",
                "menu_clarity": "clear",
                "staff_knowledge": "knowledgeable",
                "diet_subtype": "vegan",
            },
        ],
        "hygiene": [
            {
                "is_hygiene_related": True,
                "account_type": "firsthand",
                "issue_type": "cleanliness",
                "issue_severity": "moderate",
                "staff_response": "none",
            },
            {
                "is_hygiene_related": True,
                "account_type": "firsthand",
                "issue_type": "pest",
                "issue_severity": "severe",
                "staff_response": "dismissed",
            },
        ],
    }

    # Extract base topic name
    for key in topic_to_judgments:
        if key in topic_name.lower():
            judgments = topic_to_judgments[key]
            if key == "allergy":
                restaurant_meta["categories"] = "Thai, Asian Fusion"
            return judgments, restaurant_meta

    # Generic fallback
    return [{"is_relevant": True, "rating": "positive"}], restaurant_meta
