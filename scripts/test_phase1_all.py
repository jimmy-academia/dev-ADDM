#!/usr/bin/env python3
"""Test Phase 1 seed generation for all 72 policies."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

# All 72 policies
POLICIES = [
    # G1 - Customer Safety
    "G1_allergy_V0", "G1_allergy_V1", "G1_allergy_V2", "G1_allergy_V3",
    "G1_dietary_V0", "G1_dietary_V1", "G1_dietary_V2", "G1_dietary_V3",
    "G1_hygiene_V0", "G1_hygiene_V1", "G1_hygiene_V2", "G1_hygiene_V3",
    # G2 - Customer Experience
    "G2_romance_V0", "G2_romance_V1", "G2_romance_V2", "G2_romance_V3",
    "G2_business_V0", "G2_business_V1", "G2_business_V2", "G2_business_V3",
    "G2_group_V0", "G2_group_V1", "G2_group_V2", "G2_group_V3",
    # G3 - Customer Value
    "G3_price_worth_V0", "G3_price_worth_V1", "G3_price_worth_V2", "G3_price_worth_V3",
    "G3_hidden_costs_V0", "G3_hidden_costs_V1", "G3_hidden_costs_V2", "G3_hidden_costs_V3",
    "G3_time_value_V0", "G3_time_value_V1", "G3_time_value_V2", "G3_time_value_V3",
    # G4 - Owner Operations
    "G4_server_V0", "G4_server_V1", "G4_server_V2", "G4_server_V3",
    "G4_kitchen_V0", "G4_kitchen_V1", "G4_kitchen_V2", "G4_kitchen_V3",
    "G4_environment_V0", "G4_environment_V1", "G4_environment_V2", "G4_environment_V3",
    # G5 - Owner Performance
    "G5_capacity_V0", "G5_capacity_V1", "G5_capacity_V2", "G5_capacity_V3",
    "G5_execution_V0", "G5_execution_V1", "G5_execution_V2", "G5_execution_V3",
    "G5_consistency_V0", "G5_consistency_V1", "G5_consistency_V2", "G5_consistency_V3",
    # G6 - Owner Strategy
    "G6_uniqueness_V0", "G6_uniqueness_V1", "G6_uniqueness_V2", "G6_uniqueness_V3",
    "G6_comparison_V0", "G6_comparison_V1", "G6_comparison_V2", "G6_comparison_V3",
    "G6_loyalty_V0", "G6_loyalty_V1", "G6_loyalty_V2", "G6_loyalty_V3",
]


def run_phase1(policy: str) -> tuple[bool, str]:
    """Run phase 1 for a single policy.

    Returns:
        (success, message)
    """
    cmd = [
        ".venv/bin/python", "-m", "addm.tasks.cli.run_experiment",
        "--phase", "1",
        "--dev",
        "--policy", policy,
        "--method", "amos",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,  # 3 minute timeout per policy
        )

        if result.returncode == 0:
            # Check for success message
            if "Phase 1 complete" in result.stdout:
                return True, "OK"
            else:
                return False, f"Unexpected output: {result.stdout[-200:]}"
        else:
            # Extract error message
            error_lines = result.stderr.split("\n")
            for line in reversed(error_lines):
                if "Error" in line or "error" in line or "ValueError" in line:
                    return False, line[:150]
            return False, result.stderr[-200:]

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT (180s)"
    except Exception as e:
        return False, f"Exception: {e}"


def main():
    print("=" * 70)
    print("Testing Phase 1 Formula Seed Generation for All 72 Policies")
    print("=" * 70)
    print()

    results = {"passed": [], "failed": []}

    for i, policy in enumerate(POLICIES, 1):
        print(f"[{i:2d}/72] {policy}...", end=" ", flush=True)
        success, message = run_phase1(policy)

        if success:
            print(f"✓ PASS")
            results["passed"].append(policy)
        else:
            print(f"✗ FAIL: {message}")
            results["failed"].append((policy, message))

    print()
    print("=" * 70)
    print(f"SUMMARY: {len(results['passed'])}/72 passed, {len(results['failed'])}/72 failed")
    print("=" * 70)

    if results["failed"]:
        print("\nFailed policies:")
        for policy, error in results["failed"]:
            print(f"  - {policy}: {error}")

    # Save results
    output_path = Path("results/dev/phase1_test_all.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return 0 if not results["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
