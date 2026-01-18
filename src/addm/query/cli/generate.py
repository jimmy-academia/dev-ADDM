"""CLI for generating prompts from policy IRs.

Usage:
    .venv/bin/python -m addm.query.cli.generate --policy G1/allergy/V1
    .venv/bin/python -m addm.query.cli.generate --policy G1/allergy/V1 --output data/query/generated/
"""

import argparse
from pathlib import Path

from ..generators.prompt_generator import PromptGenerator
from ..models.policy import PolicyIR
from ..models.term import TermLibrary
from ..libraries import TERMS_DIR


def load_term_library() -> TermLibrary:
    """Load all term libraries."""
    library = TermLibrary()

    # Load shared terms
    shared_path = TERMS_DIR / "_shared.yaml"
    if shared_path.exists():
        library.load_domain("shared", shared_path)

    # Load topic-specific terms
    for term_file in TERMS_DIR.glob("*.yaml"):
        if term_file.name.startswith("_"):
            continue
        domain = term_file.stem
        library.load_domain(domain, term_file)

    return library


def get_policies_dir() -> Path:
    """Get the policies directory."""
    return Path(__file__).parent.parent / "policies"


def generate_prompt(policy_path: str, output_dir: Path | None = None) -> str:
    """Generate a prompt from a policy path.

    Args:
        policy_path: Relative path like "G1/allergy/V1" or full path
        output_dir: Optional output directory to save the prompt

    Returns:
        Generated prompt text
    """
    # Resolve policy path
    policies_dir = get_policies_dir()

    if "/" in policy_path and not policy_path.endswith(".yaml"):
        # Relative path like "G1/allergy/V1"
        full_path = policies_dir / f"{policy_path}.yaml"
    elif policy_path.endswith(".yaml"):
        full_path = Path(policy_path)
    else:
        full_path = policies_dir / f"{policy_path}.yaml"

    if not full_path.exists():
        raise FileNotFoundError(f"Policy not found: {full_path}")

    # Load term library
    term_library = load_term_library()

    # Load policy and generate
    policy = PolicyIR.load(full_path)
    generator = PromptGenerator(term_library)
    prompt = generator.generate(policy)

    # Save if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{policy.policy_id}_prompt.txt"
        output_file.write_text(prompt)
        print(f"Saved to: {output_file}")

    return prompt


def main():
    parser = argparse.ArgumentParser(description="Generate prompt from policy IR")
    parser.add_argument(
        "--policy",
        required=True,
        help="Policy path (e.g., G1/allergy/V1 or full path)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory to save generated prompt",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        default=True,
        help="Print the generated prompt to stdout",
    )

    args = parser.parse_args()

    prompt = generate_prompt(args.policy, args.output)

    if args.print:
        print(prompt)


if __name__ == "__main__":
    main()
