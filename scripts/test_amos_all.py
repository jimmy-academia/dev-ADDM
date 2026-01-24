#!/usr/bin/env python3
"""Test AMOS (Phase 1+2) on all 72 policies with K=25."""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

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


def run_amos(policy: str, n: int = 10, k: int = 25) -> dict:
    """Run AMOS (Phase 1+2) for a single policy.

    Returns:
        dict with policy, accuracy, correct, total, error
    """
    cmd = [
        ".venv/bin/python", "-m", "addm.tasks.cli.run_experiment",
        "--dev",
        "--policy", policy,
        "--method", "amos",
        "-n", str(n),
        "--k", str(k),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode == 0:
            # Find results.json path from output
            for line in result.stdout.split("\n"):
                if "Results saved to:" in line:
                    results_path = line.split("Results saved to:")[-1].strip()
                    with open(results_path) as f:
                        data = json.load(f)
                    return {
                        "policy": policy,
                        "accuracy": data.get("accuracy", 0) * 100,  # Convert to percentage
                        "correct": data.get("correct", 0),
                        "total": data.get("total", 0),
                        "error": None,
                    }
            return {"policy": policy, "accuracy": 0, "correct": 0, "total": 0, "error": "No results found"}
        else:
            # Extract error
            error_lines = result.stderr.split("\n")
            for line in reversed(error_lines):
                if "Error" in line or "ValueError" in line:
                    return {"policy": policy, "accuracy": 0, "correct": 0, "total": 0, "error": line[:150]}
            return {"policy": policy, "accuracy": 0, "correct": 0, "total": 0, "error": result.stderr[-200:]}

    except subprocess.TimeoutExpired:
        return {"policy": policy, "accuracy": 0, "correct": 0, "total": 0, "error": "TIMEOUT"}
    except Exception as e:
        return {"policy": policy, "accuracy": 0, "correct": 0, "total": 0, "error": str(e)}


def main():
    n_samples = 10
    k = 25

    # Allow specifying which policies to test
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            # Quick test: one V2 from each group
            policies = [
                "G1_allergy_V2", "G2_romance_V2", "G3_price_worth_V2",
                "G4_server_V2", "G5_capacity_V2", "G6_uniqueness_V2"
            ]
        elif sys.argv[1] == "--v2":
            # All V2 variants (18 policies)
            policies = [p for p in POLICIES if p.endswith("_V2")]
        else:
            # Specific policies
            policies = sys.argv[1:]
    else:
        policies = POLICIES

    print("=" * 70)
    print(f"Testing AMOS on {len(policies)} policies (n={n_samples}, K={k})")
    print("=" * 70)
    print()

    results = []
    total_correct = 0
    total_samples = 0

    for i, policy in enumerate(policies, 1):
        print(f"[{i:2d}/{len(policies)}] {policy}...", end=" ", flush=True)
        result = run_amos(policy, n=n_samples, k=k)
        results.append(result)

        if result["error"]:
            print(f"✗ ERROR: {result['error'][:60]}")
        else:
            total_correct += result["correct"]
            total_samples += result["total"]
            status = "✓" if result["accuracy"] >= 75 else "○"
            print(f"{status} {result['correct']}/{result['total']} = {result['accuracy']:.1f}%")

    print()
    print("=" * 70)
    print("SUMMARY BY GROUP")
    print("=" * 70)

    # Group results
    groups = {}
    for r in results:
        group = r["policy"][:2]  # G1, G2, etc.
        if group not in groups:
            groups[group] = {"correct": 0, "total": 0, "policies": 0, "errors": 0}
        if r["error"]:
            groups[group]["errors"] += 1
        else:
            groups[group]["correct"] += r["correct"]
            groups[group]["total"] += r["total"]
        groups[group]["policies"] += 1

    for group in sorted(groups.keys()):
        g = groups[group]
        if g["total"] > 0:
            acc = 100 * g["correct"] / g["total"]
            print(f"  {group}: {g['correct']}/{g['total']} = {acc:.1f}% ({g['policies']} policies, {g['errors']} errors)")
        else:
            print(f"  {group}: No results ({g['errors']} errors)")

    print()
    print("=" * 70)
    if total_samples > 0:
        overall_acc = 100 * total_correct / total_samples
        print(f"OVERALL: {total_correct}/{total_samples} = {overall_acc:.1f}%")
        target = 75
        if overall_acc >= target:
            print(f"✓ TARGET {target}% MET!")
        else:
            print(f"✗ Below target {target}% (need {int(target * total_samples / 100) - total_correct} more correct)")
    else:
        print("No valid results")
    print("=" * 70)

    # List failures
    failures = [r for r in results if r["error"]]
    low_acc = [r for r in results if not r["error"] and r["accuracy"] < 50]

    if failures:
        print(f"\nFailed policies ({len(failures)}):")
        for r in failures:
            print(f"  - {r['policy']}: {r['error'][:80]}")

    if low_acc:
        print(f"\nLow accuracy (<50%) policies ({len(low_acc)}):")
        for r in sorted(low_acc, key=lambda x: x["accuracy"]):
            print(f"  - {r['policy']}: {r['accuracy']:.1f}%")

    # Save results
    output_path = Path(f"results/dev/amos_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "n_samples": n_samples,
            "k": k,
            "total_correct": total_correct,
            "total_samples": total_samples,
            "overall_accuracy": overall_acc if total_samples > 0 else 0,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return 0 if total_samples > 0 and overall_acc >= 75 else 1


if __name__ == "__main__":
    sys.exit(main())
