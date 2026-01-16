"""Parse task prompts by section dividers.

Only enforces structure (sections exist), not content format.
Max flexibility for human-readable prompt content.
"""

import re
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ParsedPrompt:
    """Prompt split into sections. Content is raw text."""

    agenda: str
    l0_primitives: str
    l1_composites: str
    l1_5_grouping: Optional[str]  # Optional for tasks without semantic grouping
    l2_aggregates: str
    formulas: str
    output_schema: str

    # Full prompt for reference
    full_text: str


# Section header patterns (flexible matching)
SECTION_PATTERNS = {
    "agenda": r"Agenda",
    "l0_primitives": r"L0[:\s]|Primitives",
    "l1_composites": r"L1[:\s]|Composites",
    "l1_5_grouping": r"L1\.5|Semantic\s+Group",
    "l2_aggregates": r"L2[:\s]|Aggregates",
    "formulas": r"Formula|Decision|FL[:\s]",
    "output_schema": r"Output\s+Schema|OUTPUT",
}


def parse_prompt_sections(prompt_text: str) -> ParsedPrompt:
    """
    Parse prompt into sections by dividers.

    Expected format:
        ============================================================
        Section Name
        ============================================================
        content...

    Returns ParsedPrompt with raw text for each section.
    """
    # Split by divider lines (====...)
    divider_pattern = r"\n={10,}\n"
    parts = re.split(divider_pattern, prompt_text)

    # Build section map: header -> content
    sections: Dict[str, str] = {}
    i = 0
    while i < len(parts):
        part = parts[i].strip()

        # Check if this part is a section header
        for section_key, pattern in SECTION_PATTERNS.items():
            if re.search(pattern, part, re.IGNORECASE):
                # Next part is the content (if exists)
                content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                sections[section_key] = content
                i += 1
                break
        i += 1

    return ParsedPrompt(
        agenda=sections.get("agenda", ""),
        l0_primitives=sections.get("l0_primitives", ""),
        l1_composites=sections.get("l1_composites", ""),
        l1_5_grouping=sections.get("l1_5_grouping"),  # None if not present
        l2_aggregates=sections.get("l2_aggregates", ""),
        formulas=sections.get("formulas", ""),
        output_schema=sections.get("output_schema", ""),
        full_text=prompt_text,
    )


# =============================================================================
# L0 Helper (for extraction prompt building)
# =============================================================================


def parse_l0_fields(l0_text: str) -> Dict[str, Dict[str, str]]:
    """
    Parse L0 field definitions from the L0 section text.

    Expected format:
        FIELD_NAME
        - value1: Description
        - value2: Description

    Returns:
        Dict mapping field names (lowercase) to their value descriptions.
        Example: {"incident_severity": {"none": "No allergic...", "mild": "..."}}
    """
    result: Dict[str, Dict[str, str]] = {}
    current_field: Optional[str] = None
    current_values: Dict[str, str] = {}

    for line in l0_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Check if this is a field name (ALL CAPS with underscores)
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


def get_l0_field_order(l0_schema: Dict[str, Dict[str, str]]) -> list[str]:
    """Get ordered list of L0 field names."""
    # Common field order (task-agnostic)
    preferred_order = [
        "incident_severity",
        "account_type",
        "assurance_claim",
        "staff_response",
    ]
    fields = []
    for field in preferred_order:
        if field in l0_schema:
            fields.append(field)
    # Add any remaining fields
    for field in l0_schema:
        if field not in fields:
            fields.append(field)
    return fields
