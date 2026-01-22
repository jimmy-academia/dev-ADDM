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

    # If no vague quantifier found but min_count is provided,
    # check if text already has an explicit number at the start
    if re.match(r"^\d+\s+", text):
        # Already has explicit number, leave as-is
        return text

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
        )

    def _render_verdict_rules(self, policy: PolicyIR) -> str:
        """Render the Verdict Rules section."""
        template = self.env.get_template("verdict_rules.jinja2")

        decision = policy.normative.decision
        output = policy.normative.output

        return template.render(
            verdicts=decision.verdicts,
            rules=decision.rules,
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
