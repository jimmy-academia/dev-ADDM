#!/usr/bin/env python3
"""Generate EVIDENCE_FIELDS from Term Libraries.

This script reads the term library YAML files and generates the EVIDENCE_FIELDS
dictionary for eval/constants.py. This ensures evaluation constants stay in sync
with the source of truth (term libraries).

Usage:
    python scripts/generate_evidence_fields.py          # Print to stdout
    python scripts/generate_evidence_fields.py --check  # Validate current constants.py
"""

import argparse
import sys
from pathlib import Path

import yaml

# Map topic -> outcome field name in term library
# This is domain knowledge: which field represents the primary outcome for each topic
TOPIC_OUTCOME_FIELDS = {
    # G1: Customer Safety
    "allergy": "INCIDENT_SEVERITY",
    "dietary": "INCIDENT_SEVERITY",
    "hygiene": "ISSUE_SEVERITY",
    # G2: Customer Experience
    "romance": "DATE_OUTCOME",
    "business": "MEETING_OUTCOME",
    "group": "GROUP_OUTCOME",
    # G3: Customer Value
    "price_worth": "PRICE_PERCEPTION",
    "hidden_costs": "IMPACT_SEVERITY",
    "time_value": "WAIT_JUSTIFICATION",
    # G4: Owner Operations
    "server": "ATTENTIVENESS",
    "kitchen": "ISSUE_SEVERITY",
    "environment": "CLEANLINESS",  # environment.yaml has no ISSUE_SEVERITY
    # G5: Owner Performance
    "capacity": "SERVICE_DEGRADATION",
    "execution": "ISSUE_RESOLUTION",
    "consistency": "VISIT_COMPARISON",
    # G6: Owner Strategy
    "uniqueness": "MEMORABILITY",
    "comparison": "COMPARISON_OUTCOME",
    "loyalty": "LOYALTY_DRIVER",
}

# Map group -> topics
GROUP_TOPICS = {
    "G1": ["allergy", "dietary", "hygiene"],
    "G2": ["romance", "business", "group"],
    "G3": ["price_worth", "hidden_costs", "time_value"],
    "G4": ["server", "kitchen", "environment"],
    "G5": ["capacity", "execution", "consistency"],
    "G6": ["uniqueness", "comparison", "loyalty"],
}

TERMS_DIR = Path("src/addm/query/libraries/terms")


def load_term_library(topic: str) -> dict:
    """Load term library YAML for a topic."""
    yaml_path = TERMS_DIR / f"{topic}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Term library not found: {yaml_path}")

    with open(yaml_path) as f:
        return yaml.safe_load(f)


def get_evidence_values(term_def: dict) -> set:
    """Extract non-default values from a term definition.

    The default value is considered "neutral" and not evidence.
    We only exclude the explicit default - not all "none"/"unknown" values,
    as some fields use "none" to mean "no action taken" which IS evidence.
    """
    default_value = term_def.get("default", "")
    values = set()

    for value_def in term_def.get("values", []):
        value_id = value_def.get("id", "")
        # Only exclude the explicit default value
        if value_id and value_id != default_value:
            values.add(value_id)

    return values


def generate_evidence_fields() -> dict:
    """Generate EVIDENCE_FIELDS dict from term libraries."""
    evidence_fields = {}

    for group, topics in GROUP_TOPICS.items():
        for topic in topics:
            topic_key = f"{group}_{topic}"
            outcome_field_name = TOPIC_OUTCOME_FIELDS.get(topic)

            if not outcome_field_name:
                print(f"Warning: No outcome field defined for {topic}", file=sys.stderr)
                continue

            try:
                term_lib = load_term_library(topic)
            except FileNotFoundError as e:
                print(f"Warning: {e}", file=sys.stderr)
                continue

            if outcome_field_name not in term_lib:
                print(f"Warning: {outcome_field_name} not in {topic}.yaml", file=sys.stderr)
                continue

            term_def = term_lib[outcome_field_name]
            evidence_values = get_evidence_values(term_def)

            evidence_fields[topic_key] = {
                "field": outcome_field_name.lower(),
                "evidence_values": evidence_values,
            }

        # Add group fallback (use first topic's config)
        first_topic = topics[0]
        first_key = f"{group}_{first_topic}"
        if first_key in evidence_fields:
            evidence_fields[group] = evidence_fields[first_key].copy()

    return evidence_fields


def format_evidence_fields(evidence_fields: dict) -> str:
    """Format EVIDENCE_FIELDS as Python code."""
    lines = [
        "# Evidence field definitions per policy topic",
        "# AUTO-GENERATED from term libraries by scripts/generate_evidence_fields.py",
        "# DO NOT EDIT MANUALLY - regenerate with: python scripts/generate_evidence_fields.py",
        "#",
        "# Each topic has a primary outcome field and values that constitute \"evidence\"",
        "# (non-neutral values that indicate something noteworthy happened)",
        "EVIDENCE_FIELDS = {",
    ]

    current_group = None
    for key in sorted(evidence_fields.keys(), key=lambda k: (k[0:2] if k.startswith("G") else "Z", len(k), k)):
        config = evidence_fields[key]

        # Add group comment
        group = key[:2] if key.startswith("G") and "_" in key else key
        if group != current_group and key.startswith("G") and "_" in key:
            current_group = group
            group_names = {
                "G1": "Customer Safety",
                "G2": "Customer Experience",
                "G3": "Customer Value",
                "G4": "Owner Operations",
                "G5": "Owner Performance",
                "G6": "Owner Strategy",
            }
            lines.append(f"    # {group}: {group_names.get(group, '')}")

        # Format values as set literal
        values_str = ", ".join(f'"{v}"' for v in sorted(config["evidence_values"]))

        if key.startswith("G") and "_" not in key:
            # Group fallback
            lines.append(f'    "{key}": {{  # Fallback')
        else:
            lines.append(f'    "{key}": {{')

        lines.append(f'        "field": "{config["field"]}",')
        lines.append(f'        "evidence_values": {{{values_str}}},')
        lines.append("    },")

    lines.append("}")
    return "\n".join(lines)


def load_current_constants() -> dict:
    """Load current EVIDENCE_FIELDS from constants.py."""
    constants_path = Path("src/addm/eval/constants.py")

    # Import the module
    import importlib.util
    spec = importlib.util.spec_from_file_location("constants", constants_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.EVIDENCE_FIELDS


def compare_fields(generated: dict, current: dict) -> list:
    """Compare generated vs current EVIDENCE_FIELDS, return differences."""
    differences = []

    all_keys = set(generated.keys()) | set(current.keys())

    for key in sorted(all_keys):
        if key not in generated:
            differences.append(f"  {key}: in current but not generated")
            continue
        if key not in current:
            differences.append(f"  {key}: in generated but not current")
            continue

        gen_config = generated[key]
        cur_config = current[key]

        if gen_config["field"] != cur_config["field"]:
            differences.append(
                f"  {key}.field: generated={gen_config['field']}, "
                f"current={cur_config['field']}"
            )

        gen_values = set(gen_config["evidence_values"])
        cur_values = set(cur_config["evidence_values"])

        if gen_values != cur_values:
            only_in_gen = gen_values - cur_values
            only_in_cur = cur_values - gen_values
            diff_parts = []
            if only_in_gen:
                diff_parts.append(f"missing from current: {only_in_gen}")
            if only_in_cur:
                diff_parts.append(f"extra in current: {only_in_cur}")
            differences.append(f"  {key}.evidence_values: {', '.join(diff_parts)}")

    return differences


def main():
    parser = argparse.ArgumentParser(description="Generate EVIDENCE_FIELDS from term libraries")
    parser.add_argument("--check", action="store_true",
                        help="Check if current constants.py matches term libraries")
    args = parser.parse_args()

    evidence_fields = generate_evidence_fields()

    if args.check:
        try:
            current = load_current_constants()
            differences = compare_fields(evidence_fields, current)

            if differences:
                print("MISMATCH: constants.py differs from term libraries:")
                for diff in differences:
                    print(diff)
                sys.exit(1)
            else:
                print("OK: constants.py matches term libraries")
                sys.exit(0)
        except Exception as e:
            print(f"Error loading constants.py: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        print(format_evidence_fields(evidence_fields))


if __name__ == "__main__":
    main()
