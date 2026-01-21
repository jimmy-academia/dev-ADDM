#!/usr/bin/env python
"""Test all three Phase 1 approaches and compare results.

Usage:
    .venv/bin/python scripts/test_phase1_approaches.py --policy G1_allergy_V2
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from addm.llm import LLMService
from addm.methods.amos import Phase1Approach, generate_formula_seed
from addm.utils.output import output


def load_policy_prompt(policy_id: str, domain: str = "yelp") -> str:
    """Load prompt from policy-generated file."""
    # Normalize policy ID: G1_allergy_V2 -> G1_allergy_V2
    normalized = policy_id.replace("/", "_")
    prompt_path = Path(f"data/query/{domain}/{normalized}_prompt.txt")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Policy prompt not found: {prompt_path}")
    return prompt_path.read_text()


async def test_approach(
    approach: Phase1Approach,
    agenda: str,
    policy_id: str,
    llm: LLMService,
    output_dir: Path,
) -> dict:
    """Test a single approach and save results."""
    output.status(f"Testing {approach.value} approach...")

    try:
        # Use a separate cache dir for each approach to avoid collisions
        cache_dir = output_dir / "cache" / approach.value
        cache_dir.mkdir(parents=True, exist_ok=True)

        seed, usage = await generate_formula_seed(
            agenda=agenda,
            policy_id=policy_id,
            llm=llm,
            cache_dir=cache_dir,
            force_regenerate=True,  # Always regenerate for testing
            approach=approach,
        )

        # Save the result
        result_path = output_dir / f"{approach.value}_seed.json"
        with open(result_path, "w") as f:
            json.dump(seed, f, indent=2)

        output.success(f"{approach.value}: Generated {len(seed.get('filter', {}).get('keywords', []))} keywords")

        return {
            "approach": approach.value,
            "success": True,
            "seed": seed,
            "usage": usage,
            "path": str(result_path),
        }

    except Exception as e:
        output.error(f"{approach.value}: Failed - {e}")
        return {
            "approach": approach.value,
            "success": False,
            "error": str(e),
        }


def print_seed_summary(result: dict):
    """Print a summary of a formula seed."""
    if not result.get("success"):
        print(f"  ERROR: {result.get('error', 'Unknown error')}")
        return

    seed = result["seed"]

    # Keywords
    keywords = seed.get("filter", {}).get("keywords", [])
    print(f"  Keywords ({len(keywords)}):")
    # Show first 15, then count remaining
    for kw in keywords[:15]:
        print(f"    - {kw}")
    if len(keywords) > 15:
        print(f"    ... and {len(keywords) - 15} more")

    # Extraction fields
    fields = seed.get("extract", {}).get("fields", [])
    print(f"\n  Extraction Fields ({len(fields)}):")
    for field in fields:
        name = field.get("name", "?")
        ftype = field.get("type", "?")
        print(f"    - {name} ({ftype})")

    # Compute operations
    compute = seed.get("compute", [])
    print(f"\n  Compute Operations ({len(compute)}):")
    for op in compute:
        name = op.get("name", "?")
        op_type = op.get("op", "?")
        print(f"    - {name} ({op_type})")

    # Search strategy
    strategy = seed.get("search_strategy", {})
    if strategy:
        print(f"\n  Search Strategy:")
        priority_kw = strategy.get("priority_keywords", [])
        print(f"    Priority keywords: {priority_kw[:5]}...")
        if "stopping_condition" in strategy:
            print(f"    Stopping: {strategy['stopping_condition'][:50]}...")

    # Usage
    usage = result.get("usage", {})
    if usage:
        tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        cost = usage.get("cost_usd", 0)
        print(f"\n  Usage: {tokens:,} tokens, ${cost:.4f}")


def compare_keywords(results: list[dict]):
    """Compare keywords across approaches."""
    print("\n" + "=" * 60)
    print("KEYWORD COMPARISON")
    print("=" * 60)

    # Collect all keywords per approach
    approach_keywords = {}
    for result in results:
        if result.get("success"):
            keywords = set(result["seed"].get("filter", {}).get("keywords", []))
            approach_keywords[result["approach"]] = keywords

    if len(approach_keywords) < 2:
        print("Not enough successful results to compare")
        return

    # Find common keywords across all
    all_approaches = list(approach_keywords.keys())
    common = approach_keywords[all_approaches[0]]
    for approach in all_approaches[1:]:
        common = common & approach_keywords[approach]

    print(f"\nCommon to all ({len(common)}):")
    for kw in sorted(common)[:10]:
        print(f"  - {kw}")
    if len(common) > 10:
        print(f"  ... and {len(common) - 10} more")

    # Find unique keywords per approach
    for approach, keywords in approach_keywords.items():
        unique = keywords - common
        # Also subtract keywords from other approaches
        for other_approach, other_keywords in approach_keywords.items():
            if other_approach != approach:
                unique = unique - other_keywords

        if unique:
            print(f"\nUnique to {approach} ({len(unique)}):")
            for kw in sorted(unique)[:8]:
                print(f"  - {kw}")
            if len(unique) > 8:
                print(f"  ... and {len(unique) - 8} more")


async def main():
    parser = argparse.ArgumentParser(description="Test Phase 1 approaches")
    parser.add_argument("--policy", default="G1_allergy_V2", help="Policy ID to test")
    parser.add_argument("--output-dir", default="results/dev/phase1_test", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load agenda
    output.status(f"Loading agenda for {args.policy}...")
    agenda = load_policy_prompt(args.policy)
    if not agenda:
        output.error(f"Could not load agenda for {args.policy}")
        return 1

    output.success(f"Loaded agenda ({len(agenda)} chars)")

    # Initialize LLM
    llm = LLMService()

    # Test all three approaches
    approaches = [
        Phase1Approach.PLAN_AND_ACT,
        Phase1Approach.REACT,
        Phase1Approach.REFLEXION,
    ]

    results = []
    for approach in approaches:
        result = await test_approach(approach, agenda, args.policy, llm, output_dir)
        results.append(result)
        print()  # Spacing between approaches

    # Print summaries
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for result in results:
        print(f"\n--- {result['approach'].upper()} ---")
        print_seed_summary(result)

    # Compare keywords
    compare_keywords(results)

    # Save comparison
    comparison_path = output_dir / "comparison.json"
    comparison = {
        "policy": args.policy,
        "results": [
            {
                "approach": r["approach"],
                "success": r.get("success", False),
                "keywords_count": len(r.get("seed", {}).get("filter", {}).get("keywords", [])) if r.get("success") else 0,
                "fields_count": len(r.get("seed", {}).get("extract", {}).get("fields", [])) if r.get("success") else 0,
                "compute_count": len(r.get("seed", {}).get("compute", [])) if r.get("success") else 0,
                "usage": r.get("usage", {}),
            }
            for r in results
        ]
    }
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    output.success(f"\nResults saved to {output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
