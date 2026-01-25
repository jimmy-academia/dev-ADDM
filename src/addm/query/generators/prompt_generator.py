"""Prompt Generator: PolicyIR -> NL Agenda Prompt.

This is the key module that transforms a PolicyIR into a natural language
agenda prompt with three sections:
1. Overview - Purpose and evidence scope
2. Definitions - Auto-generated from term library
3. Verdict Rules - Precedence ladder with conditions
"""

import re
from pathlib import Path
from typing import List, Optional

from jinja2 import Environment, FileSystemLoader

from ..models.policy import PolicyIR, DecisionRule, ScoringSystem
from ..models.term import Term, TermLibrary


def _get_effective_min_count(condition: dict, k: int) -> int:
    """Get the effective min_count for a condition based on K value.

    Args:
        condition: Condition dict with min_count and optional min_count_by_k
        k: The K value (context size)

    Returns:
        The effective min_count threshold for this K
    """
    min_count_by_k = condition.get("min_count_by_k", {})
    # Check for K-specific threshold (try both int and str keys)
    if k in min_count_by_k:
        return min_count_by_k[k]
    if str(k) in min_count_by_k:
        return min_count_by_k[str(k)]
    # Fall back to base min_count
    return condition.get("min_count", 1)


def _get_effective_score(threshold, k: int) -> int:
    """Get the effective min_score or max_score for a threshold based on K value.

    Args:
        threshold: ScoringThreshold object or dict with min_score/max_score
        k: The K value (context size)

    Returns:
        The effective score threshold for this K
    """
    # Handle ScoringThreshold objects (dataclass)
    if hasattr(threshold, 'min_score_by_k'):
        by_k = threshold.min_score_by_k or {}
        if k in by_k:
            return by_k[k]
        # Also check max_score_by_k
        max_by_k = threshold.max_score_by_k or {}
        if k in max_by_k:
            return max_by_k[k]
        # Fall back to base score
        if threshold.min_score is not None:
            return threshold.min_score
        if threshold.max_score is not None:
            return threshold.max_score
        return 0

    # Handle dicts (for backward compatibility)
    for field in ["min_score_by_k", "max_score_by_k"]:
        by_k = threshold.get(field, {})
        if k in by_k:
            return by_k[k]
        if str(k) in by_k:
            return by_k[str(k)]

    return threshold.get("min_score") or threshold.get("max_score", 0)


def _build_score_threshold_map(scoring_thresholds: list, k: int) -> dict:
    """Build a map of verdict -> K-specific score threshold.

    Args:
        scoring_thresholds: List of ScoringThreshold objects from scoring.thresholds
        k: The K value (context size)

    Returns:
        Dict mapping verdict names to their K-specific thresholds
        e.g., {"Recommended": 150, "Not Recommended": -150}
    """
    result = {}
    for threshold in scoring_thresholds:
        # Handle both ScoringThreshold objects and dicts
        verdict = getattr(threshold, 'verdict', None) or threshold.get('verdict')
        if verdict:
            score = _get_effective_score(threshold, k)
            result[verdict] = score
    return result


def _substitute_score_threshold_for_verdict(text: str, verdict: str, score_map: dict) -> str:
    """Substitute score threshold numbers in condition text with K-specific values.

    Looks for patterns like "score is X or higher" or "score is X or lower"
    and replaces X with the K-specific threshold for the given verdict.

    Args:
        text: Condition text like "Total suitability score is 1200 or higher"
        verdict: The verdict name this condition belongs to (e.g., "Recommended")
        score_map: Dict mapping verdict names to K-specific thresholds

    Returns:
        Text with score thresholds replaced
    """
    if not score_map or verdict not in score_map:
        return text

    k_score = score_map[verdict]

    # Pattern: match "score is NUMBER" or "score >= NUMBER" etc.
    # Handle both positive and negative numbers, including ranges
    pattern = r'(score\s+(?:is|=|>=|<=|>|<)\s*)(-?\d+)(\s+or\s+(?:higher|lower|more|above|below))?'

    def replace_with_k_score(match):
        prefix = match.group(1)
        suffix = match.group(3) or ""
        return f"{prefix}{k_score}{suffix}"

    text = re.sub(pattern, replace_with_k_score, text, flags=re.IGNORECASE, count=1)

    # Also handle "between X and Y" patterns for ranges
    between_pattern = r'(score\s+is\s+between\s+)(\d+)(\s+and\s+)(\d+)'
    match = re.search(between_pattern, text, re.IGNORECASE)
    if match:
        # For "between", the range changes based on thresholds
        # This is complex - for now just leave it as-is
        # or use a simple heuristic based on verdict
        pass

    return text


def _substitute_min_count(text: str, min_count: int) -> str:
    """Substitute {min_count} placeholder in text with actual value.

    Args:
        text: Condition text that may contain {min_count} placeholder
        min_count: The threshold value to substitute

    Returns:
        Text with {min_count} replaced by the actual number
    """
    return text.replace("{min_count}", str(min_count))


def _make_threshold_explicit(text: str, min_count: int) -> str:
    """Replace vague quantifiers in condition text with explicit threshold.

    Examples:
        "Multiple reviews praise..." + min_count=2 -> "2 or more reviews praise..."
        "Several complaints about..." + min_count=3 -> "3 or more complaints about..."
        "Consistent reports of..." + min_count=2 -> "2 or more reports of..."
        "{min_count} or more reviews..." + min_count=2 -> "2 or more reviews..."

    Args:
        text: The condition text that may contain vague quantifiers or {min_count}
        min_count: The explicit threshold to use

    Returns:
        Text with vague quantifiers replaced by explicit threshold
    """
    # First, substitute {min_count} placeholder if present
    text = _substitute_min_count(text, min_count)

    # Pattern to match vague quantifiers at the start of condition text
    vague_patterns = [
        (r"^Multiple\s+reviews?\s+", f"{min_count} or more reviews "),
        (r"^Several\s+reviews?\s+", f"{min_count} or more reviews "),
        (r"^Many\s+reviews?\s+", f"{min_count} or more reviews "),
        (r"^Consistent\s+(\w+)\s+", f"{min_count} or more \\1 "),
        (r"^Consistently\s+", f"{min_count} or more reviews "),
        (r"^Frequent\s+(\w+)\s+", f"{min_count} or more \\1 "),
        (r"^Frequently\s+", f"{min_count} or more reviews "),
        # General pattern for "Multiple X" where X is not "reviews"
        (r"^Multiple\s+", f"{min_count} or more "),
        (r"^Several\s+", f"{min_count} or more "),
    ]

    for pattern, replacement in vague_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)

    # If text starts with an explicit number, replace it with K-specific threshold
    # e.g., "5 or more reviews report..." + min_count=2 -> "2 or more reviews report..."
    match = re.match(r"^(\d+)\s+(or\s+more\s+)?(.*)$", text, re.IGNORECASE)
    if match:
        return f"{min_count} or more {match.group(3)}"

    # No vague quantifier and no explicit number - don't modify
    return text


class PromptGenerator:
    """Generate NL agenda prompts from PolicyIR."""

    def __init__(self, term_library: TermLibrary, templates_dir: Optional[Path] = None, k: int = 200):
        """Initialize the prompt generator.

        Args:
            term_library: Loaded term library for resolving term references
            templates_dir: Optional custom templates directory
            k: Context size (25, 50, 100, 200) - affects threshold values
        """
        self.term_library = term_library
        self.k = k

        # Set up Jinja2 environment
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"

        self.env = Environment(
            loader=FileSystemLoader(templates_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Register custom filters
        self.env.filters["make_threshold_explicit"] = _make_threshold_explicit
        # Register K-aware functions
        self.env.globals["get_effective_min_count"] = lambda c: _get_effective_min_count(c, self.k)

    def generate(self, policy: PolicyIR) -> str:
        """Generate NL agenda prompt from PolicyIR.

        Args:
            policy: The policy IR to generate from

        Returns:
            Complete NL agenda prompt as string
        """
        sections = []

        # 1. Overview section
        sections.append(self._render_overview(policy))

        # 2. Definitions section
        terms = self._resolve_terms(policy.normative.terms)
        sections.append(self._render_definitions(policy, terms))

        # 3. Scoring section (optional, for V2+)
        if policy.normative.scoring:
            sections.append(self._render_scoring(policy.normative.scoring))

        # 4. Verdict Rules section
        sections.append(self._render_verdict_rules(policy))

        return "\n\n".join(sections)

    def _resolve_terms(self, term_refs: List[str]) -> List[Term]:
        """Resolve term references to Term objects."""
        return [self.term_library.resolve(ref) for ref in term_refs]

    def _render_overview(self, policy: PolicyIR) -> str:
        """Render the Overview section."""
        template = self.env.get_template("overview.jinja2")
        return template.render(
            title=policy.overview.title,
            purpose=policy.overview.purpose.strip(),
        )

    def _render_definitions(self, policy: PolicyIR, terms: List[Term]) -> str:
        """Render the Definitions section from resolved terms."""
        template = self.env.get_template("definitions.jinja2")
        return template.render(
            terms=terms,
            incident_definition=policy.overview.incident_definition.strip() if policy.overview.incident_definition else None,
        )

    def _render_scoring(self, scoring: ScoringSystem) -> str:
        """Render the Scoring System section (for V2+ policies)."""
        template = self.env.get_template("scoring.jinja2")
        return template.render(
            description=scoring.description,
            severity_points=scoring.severity_points,
            modifiers=scoring.modifiers,
            thresholds=scoring.thresholds,
            recency_rules=scoring.recency_rules,
            reference_date=scoring.reference_date,
        )

    def _render_verdict_rules(self, policy: PolicyIR) -> str:
        """Render the Verdict Rules section.

        For V2+ policies with scoring, applies K-specific score thresholds
        to the verdict rules before rendering.
        """
        template = self.env.get_template("verdict_rules.jinja2")

        decision = policy.normative.decision
        output = policy.normative.output

        # Build score threshold map if scoring is defined
        score_map = {}
        if policy.normative.scoring and policy.normative.scoring.thresholds:
            score_map = _build_score_threshold_map(
                policy.normative.scoring.thresholds, self.k
            )

        # Pre-process rules to apply K-specific score thresholds
        # This ensures the rendered text has the correct threshold for K
        processed_rules = []
        for rule in decision.rules:
            # Get the verdict name for this rule
            verdict_name = rule.verdict if hasattr(rule, 'verdict') else rule.get('verdict', '')

            # Create a mutable copy of the rule
            if hasattr(rule, '_asdict'):
                processed_rule = dict(rule._asdict())
            elif hasattr(rule, '__dict__'):
                processed_rule = dict(rule.__dict__)
            else:
                processed_rule = dict(rule)

            # Process conditions list
            conditions = getattr(rule, 'conditions', None) or processed_rule.get('conditions', [])
            if conditions and score_map:
                new_conditions = []
                for cond in conditions:
                    if isinstance(cond, str):
                        # Apply K-specific score substitution using this rule's verdict
                        cond = _substitute_score_threshold_for_verdict(cond, verdict_name, score_map)
                    new_conditions.append(cond)
                processed_rule['conditions'] = new_conditions

            # Process especially_when list
            especially_when = getattr(rule, 'especially_when', None) or processed_rule.get('especially_when', [])
            if especially_when and score_map:
                new_especially = []
                for cond in especially_when:
                    if isinstance(cond, str):
                        cond = _substitute_score_threshold_for_verdict(cond, verdict_name, score_map)
                    new_especially.append(cond)
                processed_rule['especially_when'] = new_especially

            processed_rules.append(processed_rule)

        return template.render(
            verdicts=decision.verdicts,
            rules=processed_rules,
            output_fields=output.fields,  # Pass the fields dict directly
        )


def generate_prompt(
    policy_path: Path,
    term_library: TermLibrary,
    templates_dir: Optional[Path] = None,
) -> str:
    """Convenience function to generate prompt from a policy file.

    Args:
        policy_path: Path to the policy YAML file
        term_library: Loaded term library
        templates_dir: Optional custom templates directory

    Returns:
        Generated prompt text
    """
    policy = PolicyIR.load(policy_path)
    generator = PromptGenerator(term_library, templates_dir)
    return generator.generate(policy)
