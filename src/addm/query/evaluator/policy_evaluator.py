"""Policy evaluator for ground truth computation.

Evaluates judgments against a PolicyIR to compute verdicts.
Supports both V0/V1 rule-based and V2/V3 scoring-based evaluation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..models.policy import PolicyIR, DecisionRule
from .scoring import compute_score, apply_thresholds, ScoreBreakdown


@dataclass
class EvaluationResult:
    """Result of evaluating a policy against judgments."""

    verdict: str
    score: Optional[float] = None  # Only for V2+ scoring
    breakdown: Optional[ScoreBreakdown] = None  # Detailed scoring breakdown
    matched_rule: Optional[str] = None  # Which rule triggered (for V0/V1)
    matched_conditions: List[str] = None  # Which conditions matched

    def __post_init__(self):
        if self.matched_conditions is None:
            self.matched_conditions = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {"verdict": self.verdict}
        if self.score is not None:
            result["score"] = self.score
        if self.breakdown is not None:
            result["breakdown"] = {
                "severity_counts": self.breakdown.severity_counts,
                "severity_points": self.breakdown.severity_points,
                "modifier_points": self.breakdown.modifier_points,
                "recency_adjustments": self.breakdown.recency_adjustments,
                "total_score": self.breakdown.total_score,
            }
        if self.matched_rule:
            result["matched_rule"] = self.matched_rule
        if self.matched_conditions:
            result["matched_conditions"] = self.matched_conditions
        return result


class PolicyEvaluator:
    """Evaluates judgments against a PolicyIR to compute ground truth."""

    def __init__(self, policy: PolicyIR):
        """Initialize evaluator with a policy.

        Args:
            policy: PolicyIR loaded from YAML
        """
        self.policy = policy
        self._topic_filter_field = self._infer_topic_filter(policy.policy_id)

    def _infer_topic_filter(self, policy_id: str) -> str:
        """Infer the topic filter field from policy ID.

        Args:
            policy_id: e.g., "G1_allergy_V2" or "G3_price_worth_V0"

        Returns:
            Filter field name, e.g., "is_allergy_related"
        """
        # Extract topic from policy_id
        # G1_allergy_V2 -> allergy, G3_price_worth_V0 -> price_worth
        parts = policy_id.split("_")
        if len(parts) >= 3:
            # Topic is everything between group and version
            topic = "_".join(parts[1:-1]).lower()
            # Map to filter field
            topic_filters = {
                "allergy": "is_allergy_related",
                "dietary": "is_dietary_related",
                "hygiene": "is_hygiene_related",
                "romance": "is_romance_related",
                "business": "is_business_related",
                "group": "is_group_related",
                "price_worth": "is_price_worth_related",
                "hidden_costs": "is_hidden_costs_related",
                "time_value": "is_time_value_related",
                "server": "is_server_related",
                "kitchen": "is_kitchen_related",
                "environment": "is_environment_related",
                "capacity": "is_capacity_related",
                "execution": "is_execution_related",
                "consistency": "is_consistency_related",
                "uniqueness": "is_uniqueness_related",
                "comparison": "is_comparison_related",
                "loyalty": "is_loyalty_related",
            }
            return topic_filters.get(topic, "is_relevant")
        return "is_relevant"

    def evaluate(
        self, judgments: List[Dict[str, Any]], restaurant_meta: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate judgments against the policy.

        Args:
            judgments: List of L0 judgments for the restaurant
            restaurant_meta: Restaurant metadata (name, categories)

        Returns:
            EvaluationResult with verdict and supporting information
        """
        if self.policy.normative.scoring:
            # V2/V3: Point-based scoring
            return self._evaluate_scoring(judgments, restaurant_meta)
        else:
            # V0/V1: Rule-based conditions
            return self._evaluate_rules(judgments, restaurant_meta)

    def _evaluate_scoring(
        self, judgments: List[Dict[str, Any]], restaurant_meta: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate using V2/V3 point-based scoring."""
        scoring = self.policy.normative.scoring
        score, breakdown = compute_score(
            judgments, scoring, restaurant_meta, self._topic_filter_field
        )
        verdict = apply_thresholds(score, scoring.thresholds)

        return EvaluationResult(verdict=verdict, score=score, breakdown=breakdown)

    def _evaluate_rules(
        self, judgments: List[Dict[str, Any]], restaurant_meta: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate using V0/V1 structured rules."""
        rules = self.policy.normative.decision.rules

        # Evaluate rules in order (they form a precedence ladder)
        for rule in rules:
            if rule.default:
                continue  # Skip default rule, check it last

            matched, matched_conditions = self._check_rule(rule, judgments)
            if matched:
                return EvaluationResult(
                    verdict=rule.verdict,
                    matched_rule=rule.label,
                    matched_conditions=matched_conditions,
                )

        # No rule matched, return default
        default_rule = next((r for r in rules if r.default), None)
        if default_rule:
            return EvaluationResult(
                verdict=default_rule.verdict, matched_rule="default"
            )

        # Fallback if no default rule defined
        verdicts = self.policy.normative.decision.verdicts
        return EvaluationResult(verdict=verdicts[-1] if verdicts else "Unknown")

    def _check_rule(
        self, rule: DecisionRule, judgments: List[Dict[str, Any]]
    ) -> tuple[bool, List[str]]:
        """Check if a rule's conditions are satisfied.

        Args:
            rule: DecisionRule to check
            judgments: List of judgments

        Returns:
            Tuple of (matched, list of matched condition descriptions)
        """
        matched_conditions = []

        for condition in rule.conditions:
            if isinstance(condition, str):
                # NL condition - skip for now (needs structured format)
                continue
            elif isinstance(condition, dict):
                # Structured condition
                if self._check_structured_condition(condition, judgments):
                    matched_conditions.append(self._describe_condition(condition))

        # Check logic (ANY vs ALL)
        if rule.logic.value == "ANY":
            matched = len(matched_conditions) > 0
        elif rule.logic.value == "ALL":
            # For ALL, we need all non-NL conditions to match
            structured_count = sum(
                1 for c in rule.conditions if isinstance(c, dict)
            )
            matched = len(matched_conditions) == structured_count
        else:
            matched = len(matched_conditions) > 0

        return matched, matched_conditions

    def _check_structured_condition(
        self, condition: Dict[str, Any], judgments: List[Dict[str, Any]]
    ) -> bool:
        """Check if a structured condition is satisfied.

        Condition types:
        - count_threshold: count matching judgments >= min_count
        - exists: at least one matching judgment exists
        - compound: judgment matches filter AND has additional field

        Args:
            condition: Structured condition dict
            judgments: List of judgments

        Returns:
            True if condition is satisfied
        """
        cond_type = condition.get("type", "count_threshold")
        filter_spec = condition.get("filter", {})
        min_count = condition.get("min_count", 1)

        # Count matching judgments
        matching = self._filter_judgments(judgments, filter_spec)
        count = len(matching)

        if cond_type == "count_threshold":
            return count >= min_count
        elif cond_type == "exists":
            return count >= 1
        elif cond_type == "compound":
            # Compound conditions have additional requirements
            requires = condition.get("requires", {})
            for j in matching:
                if self._judgment_matches_filter(j, requires):
                    return True
            return False

        return False

    def _filter_judgments(
        self, judgments: List[Dict[str, Any]], filter_spec: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter judgments by a filter specification.

        Args:
            judgments: List of judgments
            filter_spec: Dict of field -> value requirements

        Returns:
            List of matching judgments
        """
        # First filter by topic
        topic_filtered = [
            j for j in judgments if j.get(self._topic_filter_field, False)
        ]

        # Then apply filter spec
        return [j for j in topic_filtered if self._judgment_matches_filter(j, filter_spec)]

    def _judgment_matches_filter(
        self, judgment: Dict[str, Any], filter_spec: Dict[str, Any]
    ) -> bool:
        """Check if a judgment matches a filter specification.

        Args:
            judgment: Single judgment dict
            filter_spec: Dict of field -> value requirements

        Returns:
            True if all filter conditions are met
        """
        for field, required_value in filter_spec.items():
            actual_value = judgment.get(field)

            if isinstance(required_value, list):
                # Value must be one of the list items
                if actual_value not in required_value:
                    return False
            elif isinstance(required_value, str):
                # Exact match
                if actual_value != required_value:
                    return False
            elif isinstance(required_value, bool):
                if actual_value != required_value:
                    return False

        return True

    def _describe_condition(self, condition: Dict[str, Any]) -> str:
        """Generate a human-readable description of a condition.

        Args:
            condition: Structured condition dict

        Returns:
            Description string
        """
        cond_type = condition.get("type", "count_threshold")
        filter_spec = condition.get("filter", {})
        min_count = condition.get("min_count", 1)

        filter_desc = ", ".join(f"{k}={v}" for k, v in filter_spec.items())

        if cond_type == "count_threshold":
            return f"count({filter_desc}) >= {min_count}"
        elif cond_type == "exists":
            return f"exists({filter_desc})"
        elif cond_type == "compound":
            requires = condition.get("requires", {})
            req_desc = ", ".join(f"{k}={v}" for k, v in requires.items())
            return f"exists({filter_desc}) with {req_desc}"

        return str(condition)
