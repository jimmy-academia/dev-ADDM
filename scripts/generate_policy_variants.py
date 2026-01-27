#!/usr/bin/env python3
"""Generate P2-P7 policy variants for T2-T5 topics.

Creates placeholder files based on P1 templates with format variations.
P4-P7 share the same decision logic as P1, just different NL formats.

Usage:
    python scripts/generate_policy_variants.py
"""

import os
from pathlib import Path

# Topic configurations
TOPICS = {
    "T2": {"name": "price_worth", "title": "Price-Worth Value Assessment"},
    "T3": {"name": "environment", "title": "Environment Quality Assessment"},
    "T4": {"name": "execution", "title": "Execution Quality Assessment"},
    "T5": {"name": "server", "title": "Server Performance Assessment"},
}

# Output directory
POLICIES_DIR = Path("src/addm/query/policies")


def create_p2_placeholder(topic_id: str, topic_name: str, title: str) -> str:
    """Create P2 (extended rules) placeholder."""
    return f'''# {topic_id}_{topic_name}_P2: {title} - Extended Conditions
# P2 = Extended: Multiple conditions per verdict (ANY triggers)

policy_id: {topic_id}_{topic_name}_P2
extends: {topic_id}_{topic_name}_P1
version: "1.0"
format: markdown

# Placeholder - P2 adds compound conditions to P1 rules
# To be migrated from G*_{topic_name}_V2 when finalized

# For now, inherits P1 structure - see P1.yaml for full spec
'''


def create_p3_placeholder(topic_id: str, topic_name: str, title: str) -> str:
    """Create P3 (TBD rule variation) placeholder."""
    return f'''# {topic_id}_{topic_name}_P3: {title} - TBD Rule Variation
# P3 = TBD: Rule variation to be defined (stricter thresholds, ALL logic, etc.)

policy_id: {topic_id}_{topic_name}_P3
extends: {topic_id}_{topic_name}_P1
version: "1.0"
format: markdown

# Placeholder - rule variation to be defined later
# Options to consider:
# - ALL logic instead of ANY for conditions
# - Stricter thresholds
# - Additional contextual factors

# For now, inherits P1 structure - see P1.yaml for full spec
'''


def create_format_variant(topic_id: str, topic_name: str, title: str, variant: str, format_type: str, description: str) -> str:
    """Create format variant (P4-P7) placeholder."""
    return f'''# {topic_id}_{topic_name}_{variant}: {title} - {description}
# {variant} = Same decision logic as P1, {format_type} format

policy_id: {topic_id}_{topic_name}_{variant}
extends: {topic_id}_{topic_name}_P1
version: "1.0"
format: {format_type}

# Ground truth: Use same GT as P1 (identical decision logic)
gt_source: {topic_id}_{topic_name}_P1

# Pre-rendered agenda in {format_type} format
# TODO: Generate from P1 when OBSERVE testing needed
agenda_override: |
  # Placeholder for {format_type} format
  # See T1/allergy/{variant}.yaml for example structure
'''


def main():
    for topic_id, config in TOPICS.items():
        topic_name = config["name"]
        title = config["title"]
        topic_dir = POLICIES_DIR / topic_id / topic_name

        # Create directory if needed
        topic_dir.mkdir(parents=True, exist_ok=True)

        # Create P2 placeholder
        p2_path = topic_dir / "P2.yaml"
        if not p2_path.exists():
            p2_path.write_text(create_p2_placeholder(topic_id, topic_name, title))
            print(f"Created {p2_path}")

        # Create P3 placeholder
        p3_path = topic_dir / "P3.yaml"
        if not p3_path.exists():
            p3_path.write_text(create_p3_placeholder(topic_id, topic_name, title))
            print(f"Created {p3_path}")

        # Create format variants P4-P7
        format_variants = [
            ("P4", "reorder_v1", "Reorder v1 (Verdicts before Definitions)"),
            ("P5", "reorder_v2", "Reorder v2 (Interleaved Structure)"),
            ("P6", "xml", "XML Format"),
            ("P7", "prose", "Prose Format"),
        ]

        for variant, format_type, description in format_variants:
            variant_path = topic_dir / f"{variant}.yaml"
            if not variant_path.exists():
                content = create_format_variant(topic_id, topic_name, title, variant, format_type, description)
                variant_path.write_text(content)
                print(f"Created {variant_path}")

    print("\nDone! Created policy variant placeholders for T2-T5.")


if __name__ == "__main__":
    main()
