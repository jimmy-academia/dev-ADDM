"""Parse L0 primitive definitions from agenda spec prompts."""

import re
from typing import Dict, List


def parse_l0_from_prompt(prompt_text: str) -> Dict[str, Dict[str, str]]:
    """
    Parse L0 primitive definitions from an agenda spec prompt.

    Looks for the section:
    ============================================================
    Policy Definitions (L0: Primitives)
    ============================================================

    FIELD_NAME
    - value1: Description
    - value2: Description

    Returns:
        Dict mapping field names (lowercase) to their value descriptions.
        Example: {"incident_severity": {"none": "No allergic...", "mild": "..."}}
    """
    # Find the L0 section
    l0_pattern = r"Policy Definitions \(L0: Primitives\)\s*\n=+\n(.*?)(?:\n=+|$)"
    match = re.search(l0_pattern, prompt_text, re.DOTALL)

    if not match:
        raise ValueError("Could not find L0 Primitives section in prompt")

    l0_section = match.group(1)

    # Parse each field
    result: Dict[str, Dict[str, str]] = {}
    current_field = None
    current_values: Dict[str, str] = {}

    for line in l0_section.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Check if this is a field name (ALL CAPS)
        if re.match(r"^[A-Z][A-Z_]+$", line):
            # Save previous field
            if current_field:
                result[current_field.lower()] = current_values

            current_field = line
            current_values = {}

        # Check if this is a value definition
        elif line.startswith("- ") and current_field:
            # Parse "- value: description"
            value_match = re.match(r"^- ([^:]+):\s*(.*)$", line)
            if value_match:
                value = value_match.group(1).strip()
                description = value_match.group(2).strip()
                current_values[value] = description

    # Save last field
    if current_field:
        result[current_field.lower()] = current_values

    return result


def get_l0_fields(l0_schema: Dict[str, Dict[str, str]]) -> List[str]:
    """Get ordered list of L0 field names."""
    # Maintain a consistent order
    preferred_order = ["incident_severity", "account_type", "assurance_claim", "staff_response"]
    fields = []
    for field in preferred_order:
        if field in l0_schema:
            fields.append(field)
    # Add any remaining fields
    for field in l0_schema:
        if field not in fields:
            fields.append(field)
    return fields
