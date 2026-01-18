"""CLI for generating prompts from policy IRs.

Usage:
    # Generate single policy
    .venv/bin/python -m addm.query.cli.generate --policy G1/allergy/V1

    # Generate all policies to data/query/yelp/
    .venv/bin/python -m addm.query.cli.generate --all

    # Generate to custom output directory
    .venv/bin/python -m addm.query.cli.generate --policy G1/allergy/V1 --output data/query/custom/
"""

import argparse
from pathlib import Path

from ..generators.prompt_generator import PromptGenerator
from ..models.policy import PolicyIR
from ..models.term import TermLibrary
from ..libraries import TERMS_DIR


# Default output directory for generated prompts
# Path: src/addm/query/cli/generate.py -> ../../.. -> src/addm -> ../.. -> project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "query" / "yelp"


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


def generate_prompt(
    policy_path: str,
    output_dir: Path | None = None,
    save: bool = True,
) -> tuple[str, str]:
    """Generate a prompt from a policy path.

    Args:
        policy_path: Relative path like "G1/allergy/V1" or full path
        output_dir: Output directory to save the prompt (default: data/query/yelp/)
        save: Whether to save to file

    Returns:
        Tuple of (prompt text, output file path or "")
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

    # Save to file
    output_file = ""
    if save:
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{policy.policy_id}_prompt.txt"
        output_file.write_text(prompt)

    return prompt, str(output_file) if output_file else ""


def generate_all(output_dir: Path | None = None) -> list[str]:
    """Generate prompts for all policies.

    Args:
        output_dir: Output directory (default: data/query/yelp/)

    Returns:
        List of generated file paths
    """
    policies_dir = get_policies_dir()
    generated = []

    # Find all policy YAML files
    for policy_file in policies_dir.rglob("*.yaml"):
        # Get relative path for policy_path argument
        rel_path = policy_file.relative_to(policies_dir)
        policy_path = str(rel_path.with_suffix(""))

        try:
            _, output_file = generate_prompt(policy_path, output_dir, save=True)
            if output_file:
                generated.append(output_file)
                print(f"Generated: {output_file}")
        except Exception as e:
            print(f"Error generating {policy_path}: {e}")

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate prompt from policy IR")
    parser.add_argument(
        "--policy",
        help="Policy path (e.g., G1/allergy/V1 or full path)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all policies",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save to file, just print",
    )

    args = parser.parse_args()

    if args.all:
        generated = generate_all(args.output)
        print(f"\nGenerated {len(generated)} prompts")
    elif args.policy:
        prompt, output_file = generate_prompt(
            args.policy,
            args.output,
            save=not args.no_save,
        )
        if args.no_save or not output_file:
            print(prompt)
        else:
            print(f"Saved to: {output_file}")
    else:
        parser.error("Either --policy or --all is required")


if __name__ == "__main__":
    main()
