#!/usr/bin/env python3
"""
Verify all 72 formula modules (GT formulas).

4-Tier Verification:
- Tier 1: Basic checks (import, signature, empty input, output schema)
- Tier 2: Cross-variant verification (a/b/c/d differentiation, L1.5 presence)
- Tier 3: Prompt-constant sync (BASE_SCORE, thresholds)
- Tier 4: Manual review file generation (non-blocking)

Usage:
    .venv/bin/python scripts/verify_formulas.py
"""

import importlib
import inspect
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# =============================================================================
# Constants
# =============================================================================

# All 72 task IDs
ALL_TASKS = [f"G{g}{t}" for g in range(1, 7) for t in "abcdefghijkl"]

# 18 topics (6 groups × 3 topics per group)
# Each topic has 4 variants (a/b/c/d for first topic, e/f/g/h for second, i/j/k/l for third)
TOPICS = {
    # G1: Health & Safety
    "G1_allergy": ["G1a", "G1b", "G1c", "G1d"],
    "G1_dietary": ["G1e", "G1f", "G1g", "G1h"],
    "G1_hygiene": ["G1i", "G1j", "G1k", "G1l"],
    # G2: Social Context
    "G2_romance": ["G2a", "G2b", "G2c", "G2d"],
    "G2_business": ["G2e", "G2f", "G2g", "G2h"],
    "G2_group": ["G2i", "G2j", "G2k", "G2l"],
    # G3: Economic Value
    "G3_price_worth": ["G3a", "G3b", "G3c", "G3d"],
    "G3_hidden_costs": ["G3e", "G3f", "G3g", "G3h"],
    "G3_time_value": ["G3i", "G3j", "G3k", "G3l"],
    # G4: Talent & Performance
    "G4_server": ["G4a", "G4b", "G4c", "G4d"],
    "G4_kitchen": ["G4e", "G4f", "G4g", "G4h"],
    "G4_environment": ["G4i", "G4j", "G4k", "G4l"],
    # G5: Operational Efficiency
    "G5_capacity": ["G5a", "G5b", "G5c", "G5d"],
    "G5_execution": ["G5e", "G5f", "G5g", "G5h"],
    "G5_consistency": ["G5i", "G5j", "G5k", "G5l"],
    # G6: Competitive Strategy
    "G6_uniqueness": ["G6a", "G6b", "G6c", "G6d"],
    "G6_comparison": ["G6e", "G6f", "G6g", "G6h"],
    "G6_loyalty": ["G6i", "G6j", "G6k", "G6l"],
}

# Which variants have L1.5 (b/d for topic 1, f/h for topic 2, j/l for topic 3)
L15_VARIANTS = {"b", "d", "f", "h", "j", "l"}

# Sample formulas for manual review (one per group)
SAMPLE_TASKS = ["G1a", "G2a", "G3a", "G4a", "G5a", "G6a"]

PROMPTS_DIR = PROJECT_ROOT / "data" / "tasks" / "yelp"
FORMULAS_DIR = PROJECT_ROOT / "src" / "addm" / "tasks" / "formulas"
MANUAL_REVIEW_FILE = PROJECT_ROOT / "scripts" / "manual_review.txt"


# =============================================================================
# Test Data Generators
# =============================================================================

def get_test_data_for_topic(topic_name: str) -> Tuple[List[Dict], Dict]:
    """Generate synthetic test data appropriate for a topic."""

    # Common restaurant metadata
    restaurant_meta = {"categories": "American, Casual Dining", "name": "Test Restaurant"}

    # Topic-specific judgments
    if "allergy" in topic_name:
        judgments = [
            {
                "is_allergy_related": True,
                "incident_severity": "moderate",
                "account_type": "firsthand",
                "assurance_claim": "false",
                "staff_response": "accommodated",
                "allergen_type": "peanut",  # for L1.5 variants
                "date": "2021-05-15",
            },
            {
                "is_allergy_related": True,
                "incident_severity": "mild",
                "account_type": "firsthand",
                "assurance_claim": "false",
                "staff_response": "none",
                "allergen_type": "dairy",
                "date": "2020-03-10",
            },
        ]
        restaurant_meta["categories"] = "Thai, Asian Fusion"

    elif "dietary" in topic_name:
        judgments = [
            {
                "is_dietary_related": True,
                "diet_type": "vegetarian",
                "accommodation_quality": "excellent",
                "menu_clarity": "clear",
                "staff_knowledge": "knowledgeable",
                "diet_subtype": "vegan",
            },
        ]

    elif "hygiene" in topic_name:
        judgments = [
            {
                "is_hygiene_related": True,
                "account_type": "firsthand",
                "issue_type": "cleanliness",
                "issue_severity": "moderate",
                "staff_response": "none",
            },
            {
                "is_hygiene_related": True,
                "account_type": "firsthand",
                "issue_type": "pest",
                "issue_severity": "severe",
                "staff_response": "dismissed",
            },
        ]

    elif "romance" in topic_name:
        judgments = [
            {
                "is_romance_related": True,
                "occasion_type": "date_night",
                "ambiance_rating": "excellent",
                "privacy_level": "intimate",
                "noise_level": "quiet",
                "romantic_elements": "present",
                "experience_subtype": "anniversary",
            },
        ]

    elif "business" in topic_name:
        judgments = [
            {
                "is_business_related": True,
                "meeting_type": "client_meeting",
                "professionalism": "high",
                "noise_suitability": "appropriate",
                "service_timing": "efficient",
                "business_context": "formal",
            },
        ]

    elif "group" in topic_name:
        judgments = [
            {
                "is_group_related": True,
                "group_size": "large",
                "accommodation_quality": "good",
                "seating_arrangement": "flexible",
                "noise_tolerance": "acceptable",
                "group_type": "celebration",
            },
        ]

    elif "price_worth" in topic_name:
        judgments = [
            {
                "is_value_related": True,
                "price_perception": "fair",
                "portion_size": "generous",
                "quality_for_price": "excellent",
                "value_category": "special_occasion",
            },
        ]

    elif "hidden_costs" in topic_name:
        judgments = [
            {
                "is_cost_related": True,
                "hidden_fee_type": "service_charge",
                "fee_disclosure": "unclear",
                "impact_on_value": "moderate",
                "cost_category": "mandatory",
            },
        ]

    elif "time_value" in topic_name:
        judgments = [
            {
                "is_time_related": True,
                "wait_time": "moderate",
                "time_worth_it": "yes",
                "service_speed": "efficient",
                "time_context": "peak_hours",
            },
        ]

    elif "server" in topic_name:
        judgments = [
            {
                "is_server_related": True,
                "attentiveness": "excellent",
                "knowledge": "high",
                "friendliness": "warm",
                "professionalism": "high",
                "server_strength": "knowledge",
            },
        ]

    elif "kitchen" in topic_name:
        judgments = [
            {
                "is_kitchen_related": True,
                "food_quality": "excellent",
                "consistency": "high",
                "creativity": "innovative",
                "execution": "flawless",
                "culinary_focus": "technique",
            },
        ]

    elif "environment" in topic_name:
        judgments = [
            {
                "is_environment_related": True,
                "atmosphere": "inviting",
                "cleanliness": "spotless",
                "comfort": "comfortable",
                "decor": "appealing",
                "ambiance_type": "cozy",
            },
        ]

    elif "capacity" in topic_name:
        judgments = [
            {
                "is_capacity_related": True,
                "crowd_level": "busy",
                "wait_management": "efficient",
                "seating_availability": "adequate",
                "reservation_ease": "easy",
                "capacity_pattern": "peak",
            },
        ]

    elif "execution" in topic_name:
        judgments = [
            {
                "is_execution_related": True,
                "order_accuracy": "perfect",
                "timing": "excellent",
                "coordination": "smooth",
                "error_handling": "professional",
                "execution_strength": "timing",
            },
        ]

    elif "consistency" in topic_name:
        judgments = [
            {
                "is_consistency_related": True,
                "visit_comparison": "consistent",
                "quality_variance": "none",
                "experience_reliability": "high",
                "consistency_area": "food",
            },
        ]

    elif "uniqueness" in topic_name:
        judgments = [
            {
                "is_uniqueness_related": True,
                "dish_uniqueness": "signature_standout",
                "atmosphere_uniqueness": "one_of_a_kind",
                "concept_innovation": "pioneering",
                "memorability": "unforgettable",
                "uniqueness_category": "culinary",
            },
        ]

    elif "comparison" in topic_name:
        judgments = [
            {
                "is_comparison_related": True,
                "competitor_mention": "favorable",
                "category_ranking": "top_tier",
                "price_comparison": "fair_value",
                "quality_comparison": "superior",
                "comparison_dimension": "quality",
            },
        ]

    elif "loyalty" in topic_name:
        judgments = [
            {
                "is_loyalty_related": True,
                "return_intention": "definitely_returning",
                "recommendation_likelihood": "highly_recommend",
                "visit_frequency": "regular",
                "relationship_depth": "personal_connection",
                "advocacy_behavior": "brings_others",
                "loyalty_type": "behavioral",
            },
        ]

    else:
        # Generic fallback
        judgments = [{"is_relevant": True, "rating": "positive"}]

    return judgments, restaurant_meta


# =============================================================================
# Tier 1: Basic Checks
# =============================================================================

def tier1_basic_checks() -> Dict[str, Any]:
    """Run basic checks: import, signature, empty input, output schema."""
    print("TIER 1: Basic Checks")
    print("-" * 40)

    results = {
        "import_pass": [],
        "import_fail": [],
        "signature_pass": [],
        "signature_fail": [],
        "empty_input_pass": [],
        "empty_input_fail": [],
        "output_schema_pass": [],
        "output_schema_warn": [],
    }

    for task_id in ALL_TASKS:
        # 1. Import check
        try:
            module = importlib.import_module(f"addm.tasks.formulas.{task_id}")
            results["import_pass"].append(task_id)
        except Exception as e:
            results["import_fail"].append((task_id, str(e)))
            continue

        # 2. Function signature check
        if not hasattr(module, "compute_ground_truth"):
            results["signature_fail"].append((task_id, "missing compute_ground_truth"))
            continue

        func = module.compute_ground_truth
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        if len(params) < 2:
            results["signature_fail"].append((task_id, f"expected 2 params, got {len(params)}"))
            continue
        results["signature_pass"].append(task_id)

        # 3. Empty input test
        try:
            output = func([], {})
            results["empty_input_pass"].append(task_id)
        except Exception as e:
            results["empty_input_fail"].append((task_id, str(e)))
            continue

        # 4. Output schema check
        required_fields = {"verdict"}  # At minimum, verdict should exist
        score_fields = {"FINAL_SCORE", "FINAL_RISK_SCORE", "final_score"}  # Common score field names

        has_verdict = "verdict" in output
        has_score = any(f in output for f in score_fields)

        if has_verdict or has_score:
            results["output_schema_pass"].append(task_id)
        else:
            results["output_schema_warn"].append((task_id, f"missing verdict/score, got keys: {list(output.keys())[:5]}"))

    # Print summary
    total = len(ALL_TASKS)
    print(f"[{'PASS' if not results['import_fail'] else 'FAIL'}] {len(results['import_pass'])}/{total} modules import successfully")
    if results["import_fail"]:
        for task_id, err in results["import_fail"][:3]:
            print(f"       ✗ {task_id}: {err[:50]}")
        if len(results["import_fail"]) > 3:
            print(f"       ... and {len(results['import_fail']) - 3} more")

    print(f"[{'PASS' if not results['signature_fail'] else 'FAIL'}] {len(results['signature_pass'])}/{total} have compute_ground_truth function")
    if results["signature_fail"]:
        for task_id, err in results["signature_fail"][:3]:
            print(f"       ✗ {task_id}: {err}")

    print(f"[{'PASS' if not results['empty_input_fail'] else 'FAIL'}] {len(results['empty_input_pass'])}/{total} handle empty input")
    if results["empty_input_fail"]:
        for task_id, err in results["empty_input_fail"][:3]:
            print(f"       ✗ {task_id}: {err[:50]}")

    warn_count = len(results["output_schema_warn"])
    print(f"[{'PASS' if warn_count == 0 else 'WARN'}] {len(results['output_schema_pass'])}/{total} have expected output fields")
    if results["output_schema_warn"]:
        for task_id, err in results["output_schema_warn"][:3]:
            print(f"       ⚠ {task_id}: {err}")

    print()
    return results


# =============================================================================
# Tier 2: Cross-Variant Verification
# =============================================================================

def tier2_cross_variant() -> Dict[str, Any]:
    """Verify variants produce different outputs and L1.5 presence."""
    print("TIER 2: Cross-Variant Checks")
    print("-" * 40)

    results = {
        "differentiation_pass": [],
        "differentiation_fail": [],
        "l15_presence_pass": [],
        "l15_presence_fail": [],
        "l15_absence_pass": [],
        "l15_absence_fail": [],
    }

    for topic_name, variants in TOPICS.items():
        # Get test data for this topic
        judgments, restaurant_meta = get_test_data_for_topic(topic_name)

        outputs = {}
        for task_id in variants:
            try:
                module = importlib.import_module(f"addm.tasks.formulas.{task_id}")
                outputs[task_id] = module.compute_ground_truth(judgments, restaurant_meta)
            except Exception as e:
                outputs[task_id] = {"error": str(e)}

        # Check differentiation: at least 2 different scores/verdicts among 4 variants
        scores = []
        for task_id, output in outputs.items():
            if "error" in output:
                continue
            # Try various score field names
            score = output.get("FINAL_SCORE") or output.get("FINAL_RISK_SCORE") or output.get("final_score")
            verdict = output.get("verdict", "")
            scores.append((score, verdict))

        unique_results = set(scores)
        if len(unique_results) >= 2:
            results["differentiation_pass"].append(topic_name)
        else:
            results["differentiation_fail"].append((topic_name, f"all variants same: {unique_results}"))

        # Check L1.5 presence in b/d/f/h/j/l variants
        for task_id, output in outputs.items():
            if "error" in output:
                continue

            variant_letter = task_id[-1]
            is_l15_variant = variant_letter in L15_VARIANTS

            # L1.5 indicators: keys containing "L1_5", "L15", "pattern", or "PATTERN_BONUS"
            has_l15_fields = any(
                "L1_5" in k or "L15" in k or "_pattern" in k.lower() or "PATTERN_BONUS" in k
                for k in output.keys()
            )

            if is_l15_variant:
                if has_l15_fields:
                    results["l15_presence_pass"].append(task_id)
                else:
                    results["l15_presence_fail"].append((task_id, "missing L1.5 fields"))
            else:
                # a/c/e/g/i/k should NOT have L1.5 pattern bonus
                has_pattern_bonus = any("PATTERN_BONUS" in k for k in output.keys())
                if not has_pattern_bonus:
                    results["l15_absence_pass"].append(task_id)
                else:
                    results["l15_absence_fail"].append((task_id, "unexpected PATTERN_BONUS field"))

    # Print summary
    total_topics = len(TOPICS)
    diff_pass = len(results["differentiation_pass"])
    print(f"[{'PASS' if diff_pass == total_topics else 'WARN'}] {diff_pass}/{total_topics} topics have variant differentiation")
    if results["differentiation_fail"]:
        for topic, msg in results["differentiation_fail"][:3]:
            print(f"       ⚠ {topic}: {msg[:50]}")

    l15_variants_count = len([t for t in ALL_TASKS if t[-1] in L15_VARIANTS])  # 36
    l15_pass = len(results["l15_presence_pass"])
    print(f"[{'PASS' if l15_pass >= l15_variants_count * 0.8 else 'WARN'}] {l15_pass}/{l15_variants_count} L1.5 variants (b/d/f/h/j/l) output pattern fields")
    if results["l15_presence_fail"]:
        for task_id, msg in results["l15_presence_fail"][:3]:
            print(f"       ⚠ {task_id}: {msg}")

    non_l15_count = len(ALL_TASKS) - l15_variants_count  # 36
    l15_abs_pass = len(results["l15_absence_pass"])
    print(f"[{'PASS' if not results['l15_absence_fail'] else 'WARN'}] {l15_abs_pass}/{non_l15_count} non-L1.5 variants (a/c/e/g/i/k) omit pattern bonus")
    if results["l15_absence_fail"]:
        for task_id, msg in results["l15_absence_fail"][:3]:
            print(f"       ⚠ {task_id}: {msg}")

    print()
    return results


# =============================================================================
# Tier 3: Prompt-Constant Sync
# =============================================================================

def extract_base_score_from_prompt(prompt_path: Path) -> Optional[float]:
    """Extract BASE_SCORE or BASE_RISK from prompt file."""
    try:
        content = prompt_path.read_text()
        # Look for patterns like "BASE_SCORE = 5.0" or "BASE_RISK = 2.5"
        match = re.search(r"BASE_(?:SCORE|RISK)\s*=\s*(\d+\.?\d*)", content)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


def extract_base_score_from_formula(formula_path: Path) -> Optional[float]:
    """Extract BASE_SCORE or BASE_RISK from formula module."""
    try:
        content = formula_path.read_text()
        # Look for patterns like "BASE_SCORE = 5.0" or "BASE_RISK = 2.5"
        match = re.search(r"BASE_(?:SCORE|RISK)\s*=\s*(\d+\.?\d*)", content)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


def tier3_constant_sync() -> Dict[str, Any]:
    """Compare constants between prompts and formulas."""
    print("TIER 3: Constant Sync")
    print("-" * 40)

    results = {
        "base_score_match": [],
        "base_score_mismatch": [],
        "base_score_missing": [],
    }

    for task_id in ALL_TASKS:
        prompt_path = PROMPTS_DIR / f"{task_id}_prompt.txt"
        formula_path = FORMULAS_DIR / f"{task_id}.py"

        prompt_score = extract_base_score_from_prompt(prompt_path) if prompt_path.exists() else None
        formula_score = extract_base_score_from_formula(formula_path) if formula_path.exists() else None

        if prompt_score is None or formula_score is None:
            results["base_score_missing"].append((task_id, f"prompt={prompt_score}, formula={formula_score}"))
        elif abs(prompt_score - formula_score) < 0.001:
            results["base_score_match"].append(task_id)
        else:
            results["base_score_mismatch"].append((task_id, f"prompt={prompt_score}, formula={formula_score}"))

    # Print summary
    total = len(ALL_TASKS)
    match_count = len(results["base_score_match"])
    mismatch_count = len(results["base_score_mismatch"])
    missing_count = len(results["base_score_missing"])

    status = "PASS" if mismatch_count == 0 else "WARN"
    print(f"[{status}] {match_count}/{total} BASE_SCORE/BASE_RISK matches prompt")

    if results["base_score_mismatch"]:
        print(f"       ⚠ {mismatch_count} mismatches:")
        for task_id, msg in results["base_score_mismatch"][:5]:
            print(f"         {task_id}: {msg}")

    if missing_count > 0:
        print(f"[INFO] {missing_count} could not be compared (missing in prompt or formula)")

    print()
    return results


# =============================================================================
# Tier 4: Manual Review File Generation
# =============================================================================

def extract_formulas_section(prompt_path: Path) -> str:
    """Extract the Formulas & Decision Policy section from a prompt."""
    try:
        content = prompt_path.read_text()
        # Look for section starting with "Formulas" or "Formulas & Decision"
        # and ending at next section (====) or end of file
        match = re.search(
            r"(?:Formulas[^\n]*\n)(.*?)(?=\n={5,}|\Z)",
            content,
            re.DOTALL | re.IGNORECASE
        )
        if match:
            section = match.group(1).strip()
            lines = section.split("\n")
            # Filter to relevant lines (non-empty, contain =, <, >, or keywords)
            relevant = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if any(kw in stripped for kw in ["=", "<", ">", "=>", "iff", "Threshold", "Override", "BASE_"]):
                    relevant.append("  " + stripped)
            if relevant:
                return "\n".join(relevant[:12])
        # Fallback: try to find lines with formulas
        formula_lines = []
        for line in content.split("\n"):
            stripped = line.strip()
            if re.match(r"^[A-Z_]+ = ", stripped) and "=" in stripped:
                formula_lines.append("  " + stripped)
        if formula_lines:
            return "\n".join(formula_lines[:12])
    except Exception as e:
        return f"(Error: {e})"
    return "(Could not extract formulas section)"


def extract_formula_code(formula_path: Path) -> str:
    """Extract key formula computation lines from a formula module."""
    try:
        content = formula_path.read_text()
        lines = content.split("\n")

        # Find lines containing formula-like assignments
        formula_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Look for score/risk computations
            if any(kw in stripped.lower() for kw in ["score =", "risk =", "raw_", "final_", "verdict"]):
                if not stripped.startswith("#") and not stripped.startswith("def "):
                    formula_lines.append(f"  (L{i+1}) {stripped}")

        return "\n".join(formula_lines[:10]) if formula_lines else "(Could not extract formula lines)"
    except Exception:
        pass
    return "(Could not read formula file)"


def tier4_manual_review() -> None:
    """Generate manual_review.txt with sample formulas for human verification."""
    print("TIER 4: Manual Review File")
    print("-" * 40)

    lines = ["=" * 60]
    lines.append("Manual Review: 6 Formulas (1 per group)")
    lines.append("Generated by verify_formulas.py")
    lines.append("=" * 60)
    lines.append("")

    topic_names = {
        "G1a": "Allergy Safety",
        "G2a": "Romance Suitability",
        "G3a": "Price-Worth Value",
        "G4a": "Server Performance",
        "G5a": "Capacity Management",
        "G6a": "Uniqueness Differentiation",
    }

    for task_id in SAMPLE_TASKS:
        prompt_path = PROMPTS_DIR / f"{task_id}_prompt.txt"
        formula_path = FORMULAS_DIR / f"{task_id}.py"

        lines.append("-" * 60)
        lines.append(f"--- {task_id} ({topic_names.get(task_id, 'Unknown')}) ---")
        lines.append("-" * 60)
        lines.append("")

        # Prompt formulas
        lines.append("PROMPT FORMULA:")
        lines.append(extract_formulas_section(prompt_path))
        lines.append("")

        # Code formulas
        lines.append("CODE FORMULA:")
        lines.append(extract_formula_code(formula_path))
        lines.append("")

        # Run a test case
        topic_key = [k for k, v in TOPICS.items() if task_id in v][0] if any(task_id in v for v in TOPICS.values()) else None
        if topic_key:
            judgments, restaurant_meta = get_test_data_for_topic(topic_key)
            try:
                module = importlib.import_module(f"addm.tasks.formulas.{task_id}")
                output = module.compute_ground_truth(judgments, restaurant_meta)

                lines.append("TEST CASE OUTPUT:")
                # Show key output fields
                for key in ["FINAL_SCORE", "FINAL_RISK_SCORE", "RAW_SCORE", "RAW_RISK", "verdict", "base_verdict_by_score", "override_applied"]:
                    if key in output:
                        lines.append(f"  {key}: {output[key]}")
                lines.append("")
                lines.append("[ ] VERIFIED BY HUMAN")
            except Exception as e:
                lines.append(f"TEST CASE ERROR: {e}")

        lines.append("")

    # Write file
    MANUAL_REVIEW_FILE.write_text("\n".join(lines))
    print(f"→ {MANUAL_REVIEW_FILE.relative_to(PROJECT_ROOT)} generated ({len(SAMPLE_TASKS)} formulas)")
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Formula Verification Report")
    print("=" * 60)
    print()

    tier1_results = tier1_basic_checks()
    tier2_results = tier2_cross_variant()
    tier3_results = tier3_constant_sync()
    tier4_manual_review()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    tier1_pass = len(tier1_results["import_fail"]) == 0 and len(tier1_results["signature_fail"]) == 0
    tier2_pass = len(tier2_results["differentiation_fail"]) == 0
    tier3_pass = len(tier3_results["base_score_mismatch"]) == 0

    print(f"Tier 1 (Basic):        {'PASS' if tier1_pass else 'ISSUES FOUND'}")
    print(f"Tier 2 (Cross-variant): {'PASS' if tier2_pass else 'ISSUES FOUND'}")
    print(f"Tier 3 (Constant sync): {'PASS' if tier3_pass else 'ISSUES FOUND'}")
    print(f"Tier 4 (Manual review): → scripts/manual_review.txt")
    print()

    if tier1_pass and tier2_pass and tier3_pass:
        print("All automated checks passed!")
    else:
        print("Some issues found. Review output above.")

    return 0 if (tier1_pass and tier2_pass and tier3_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
