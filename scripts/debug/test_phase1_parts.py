#!/usr/bin/env python
"""Test Phase 1 extraction steps separately.

Usage:
    .venv/bin/python scripts/debug/test_phase1_parts.py --policy G1_allergy_V2

This script runs each Phase 1 extraction step separately and prints the output,
so you can understand what each LLM call produces.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from addm.llm import LLMService
from addm.methods.amos.phase1 import (
    _parse_query_sections,
    _extract_terms_from_section,
    _extract_scoring_from_section,
    _extract_verdicts_from_section,
    _combine_parts_to_yaml,
)
from addm.methods.amos.seed_compiler import compile_yaml_to_seed


def print_section(title: str, content: str):
    """Print a section with a header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(content)


def print_json(title: str, data: dict):
    """Print JSON data with a header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(json.dumps(data, indent=2, default=str))


async def test_phase1_parts(policy_id: str):
    """Test each Phase 1 extraction step separately."""

    # Load the policy prompt (agenda)
    prompt_path = Path(f"data/query/yelp/{policy_id}_prompt.txt")
    if not prompt_path.exists():
        print(f"Error: Prompt file not found: {prompt_path}")
        print("Run: .venv/bin/python -m addm.query.cli.generate --policy <policy>")
        return

    agenda = prompt_path.read_text()
    print_section("AGENDA (Policy Prompt)", agenda[:2000] + "..." if len(agenda) > 2000 else agenda)

    # Initialize LLM
    llm = LLMService()

    # =========================================================================
    # STEP 0: Parse query into sections
    # =========================================================================
    print_section("STEP 0: Parse Query Sections", "")
    sections = _parse_query_sections(agenda)
    print(f"Found sections: {list(sections.keys())}")
    for name, content in sections.items():
        print(f"\n--- {name} ({len(content)} chars) ---")
        print(content[:500] + "..." if len(content) > 500 else content)

    input("\nPress Enter to continue to Step 1 (Extract Terms)...")

    # =========================================================================
    # STEP 1: Extract terms
    # =========================================================================
    print_section("STEP 1: Extract Terms", "")

    terms_section = sections.get("terms", sections.get("header", ""))
    print(f"Input section ({len(terms_section)} chars):")
    print(terms_section[:1000] + "..." if len(terms_section) > 1000 else terms_section)

    print("\n--- Calling LLM... ---")
    terms, usage1 = await _extract_terms_from_section(terms_section, llm, policy_id)

    print(f"\nExtracted {len(terms)} terms:")
    print_json("Terms Output", terms)
    print(f"\nUsage: {usage1.get('prompt_tokens', 0)} prompt + {usage1.get('completion_tokens', 0)} completion tokens")

    input("\nPress Enter to continue to Step 2 (Extract Scoring)...")

    # =========================================================================
    # STEP 2: Extract scoring
    # =========================================================================
    print_section("STEP 2: Extract Scoring", "")

    scoring_section = sections.get("scoring", "")
    if not scoring_section:
        print("No scoring section found - this is a count-based policy (V0/V1)")
        scoring = {"policy_type": "count_rule_based"}
    else:
        print(f"Input section ({len(scoring_section)} chars):")
        print(scoring_section[:1000] + "..." if len(scoring_section) > 1000 else scoring_section)

        print("\n--- Calling LLM... ---")
        scoring, usage2 = await _extract_scoring_from_section(scoring_section, terms, llm, policy_id)
        print(f"\nUsage: {usage2.get('prompt_tokens', 0)} prompt + {usage2.get('completion_tokens', 0)} completion tokens")

    print_json("Scoring Output", scoring)

    input("\nPress Enter to continue to Step 3 (Extract Verdicts)...")

    # =========================================================================
    # STEP 3: Extract verdicts
    # =========================================================================
    print_section("STEP 3: Extract Verdicts", "")

    verdicts_section = sections.get("verdicts", sections.get("header", ""))
    print(f"Input section ({len(verdicts_section)} chars):")
    print(verdicts_section[:1000] + "..." if len(verdicts_section) > 1000 else verdicts_section)

    print("\n--- Calling LLM... ---")
    verdicts_data, usage3 = await _extract_verdicts_from_section(
        verdicts_section, terms, scoring, llm, policy_id
    )

    print_json("Verdicts Output", verdicts_data)
    print(f"\nUsage: {usage3.get('prompt_tokens', 0)} prompt + {usage3.get('completion_tokens', 0)} completion tokens")

    input("\nPress Enter to continue to Final Step (Combine & Compile)...")

    # =========================================================================
    # FINAL: Combine and compile
    # =========================================================================
    print_section("FINAL: Combine Parts & Compile to Seed", "")

    # Combine
    yaml_data = _combine_parts_to_yaml(terms, scoring, verdicts_data, policy_id)
    print_json("Combined PolicyYAML", yaml_data)

    # Compile
    print("\n--- Compiling to Formula Seed... ---")
    seed = compile_yaml_to_seed(yaml_data, validate=False)

    print_json("Formula Seed (extract)", {"fields": seed.get("extract", {}).get("fields", [])})
    print_json("Formula Seed (compute)", seed.get("compute", []))
    print_json("Formula Seed (output)", seed.get("output", []))

    print("\n" + "="*60)
    print(" DONE - Phase 1 Complete")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Test Phase 1 extraction steps")
    parser.add_argument("--policy", required=True, help="Policy ID (e.g., G1_allergy_V2)")
    args = parser.parse_args()

    asyncio.run(test_phase1_parts(args.policy))


if __name__ == "__main__":
    main()
