"""Prompt Generator: PolicyIR -> NL Agenda Prompt.

This is the key module that transforms a PolicyIR into a natural language
agenda prompt with three sections:
1. Overview - Purpose and evidence scope
2. Definitions - Auto-generated from term library
3. Verdict Rules - Precedence ladder with conditions
"""

from pathlib import Path
from typing import List, Optional

from jinja2 import Environment, FileSystemLoader

from ..models.policy import PolicyIR, DecisionRule
from ..models.term import Term, TermLibrary


class PromptGenerator:
    """Generate NL agenda prompts from PolicyIR."""

    def __init__(self, term_library: TermLibrary, templates_dir: Optional[Path] = None):
        """Initialize the prompt generator.

        Args:
            term_library: Loaded term library for resolving term references
            templates_dir: Optional custom templates directory
        """
        self.term_library = term_library

        # Set up Jinja2 environment
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"

        self.env = Environment(
            loader=FileSystemLoader(templates_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )

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
        sections.append(self._render_definitions(terms))

        # 3. Verdict Rules section
        sections.append(self._render_verdict_rules(policy))

        return "\n\n".join(sections)

    def _resolve_terms(self, term_refs: List[str]) -> List[Term]:
        """Resolve term references to Term objects."""
        return [self.term_library.resolve(ref) for ref in term_refs]

    def _render_overview(self, policy: PolicyIR) -> str:
        """Render the Overview section."""
        template = self.env.get_template("overview.jinja2")
        return template.render(
            purpose=policy.overview.purpose.strip(),
            evidence_scope=policy.overview.evidence_scope.strip(),
        )

    def _render_definitions(self, terms: List[Term]) -> str:
        """Render the Definitions section from resolved terms."""
        template = self.env.get_template("definitions.jinja2")
        return template.render(terms=terms)

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
