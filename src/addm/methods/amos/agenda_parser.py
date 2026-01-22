"""Agenda Parser: Extract structured expectations from NL agenda text.

Parses natural language agenda text (generated from policy YAMLs) to extract:
1. Verdict labels (e.g., ["Needs Improvement", "Satisfactory", "Excellent"])
2. Evaluation order (which verdict is checked first)
3. Default verdict (the "otherwise" verdict)
4. Count thresholds (e.g., "2 or more reviews" -> threshold >= 2)
5. V2 scoring (point values, score thresholds)

This enables specification-based validation of LLM-generated Formula Seeds.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ScoringSystem:
    """V2/V3 scoring system extracted from agenda."""

    # Severity/outcome points: {"severe": 15, "moderate": 8, "mild": 3}
    severity_points: Dict[str, int] = field(default_factory=dict)

    # Modifier points: {"dismissive_staff": 3}
    modifier_points: Dict[str, int] = field(default_factory=dict)

    # Score thresholds: {"Critical Risk": 8, "High Risk": 4}
    thresholds: Dict[str, int] = field(default_factory=dict)


@dataclass
class ConditionExpectation:
    """Expected condition from a verdict rule."""

    # The condition text
    text: str

    # Expected threshold (e.g., 2 for "2 or more reviews")
    min_count: Optional[int] = None

    # Which verdict this condition belongs to
    verdict: Optional[str] = None


@dataclass
class AgendaExpectations:
    """Structured expectations parsed from agenda text."""

    # Verdict labels in order of appearance (e.g., ["Excellent", "Needs Improvement", "Satisfactory"])
    verdict_labels: List[str] = field(default_factory=list)

    # Evaluation order (which verdict is checked first based on rule precedence)
    # e.g., ["Excellent", "Needs Improvement", "Satisfactory"] if Excellent is checked first
    evaluation_order: List[str] = field(default_factory=list)

    # Default verdict (the "otherwise" verdict)
    default_verdict: Optional[str] = None

    # Count thresholds for each verdict: {verdict: [min_count values]}
    count_thresholds: Dict[str, List[int]] = field(default_factory=dict)

    # V2/V3 scoring system (optional)
    scoring_system: Optional[ScoringSystem] = None

    # Raw conditions with their expected thresholds
    conditions: List[ConditionExpectation] = field(default_factory=list)

    def has_scoring(self) -> bool:
        """Check if this is a scoring-based policy (V2/V3)."""
        return self.scoring_system is not None and bool(self.scoring_system.thresholds)


def _extract_verdict_labels(agenda: str) -> List[str]:
    """Extract verdict labels from agenda text.

    Looks for patterns like:
    - "Assign a label: X, Y, or Z"
    - "Assign a ... label: X, Y, or Z"
    - "**X** if" or "**X** otherwise"

    Args:
        agenda: Full agenda text

    Returns:
        List of verdict labels found
    """
    labels = []

    # Pattern 1: "Assign a [adjective] label: X, Y, or Z"
    assign_pattern = r"Assign\s+(?:a|an)\s+(?:\w+\s+)*label:\s*([^.\n]+)"
    assign_match = re.search(assign_pattern, agenda, re.IGNORECASE)
    if assign_match:
        label_text = assign_match.group(1)
        # Parse "X, Y, or Z" format - handle comma-or patterns
        # First normalize: "X, Y, or Z" -> "X|Y|Z"
        label_text = re.sub(r",\s*or\s+", "|", label_text)  # "X, or Y" -> "X|Y"
        label_text = re.sub(r"\s+or\s+", "|", label_text)    # "X or Y" -> "X|Y"
        label_text = re.sub(r",\s*", "|", label_text)        # "X, Y" -> "X|Y"

        parts = label_text.split("|")
        for part in parts:
            part = part.strip().rstrip(".")
            # Skip empty parts and fragments like "or" that got split wrong
            if part and len(part) > 2 and not part.lower().startswith("or "):
                if part not in labels:
                    labels.append(part)

    # Pattern 2: "**Verdict Name**" in rules section
    # Find all bold verdict names followed by "if" or "otherwise"
    # Only look in the Verdict Rules section
    rules_section = agenda
    rules_match = re.search(r"## Verdict Rules\s*\n(.*)", agenda, re.DOTALL)
    if rules_match:
        rules_section = rules_match.group(1)

    verdict_pattern = r"\*\*([^*]+)\*\*\s*(?:if|otherwise)"
    for match in re.finditer(verdict_pattern, rules_section):
        verdict = match.group(1).strip()
        if verdict and verdict not in labels:
            labels.append(verdict)

    return labels


def _extract_evaluation_order(agenda: str, verdict_labels: List[str]) -> List[str]:
    """Extract evaluation order based on rule precedence.

    Rules are checked in order of appearance. The first verdict in the rules
    section is checked first.

    Also handles "if X does not apply" patterns to determine order.

    Args:
        agenda: Full agenda text
        verdict_labels: List of known verdict labels

    Returns:
        Ordered list of verdict labels (first = checked first)
    """
    order = []

    # Find Verdict Rules section
    rules_match = re.search(r"## Verdict Rules\s*\n(.*)", agenda, re.DOTALL)
    if not rules_match:
        return verdict_labels  # Fall back to appearance order

    rules_text = rules_match.group(1)

    # Find all verdict appearances in order
    for label in verdict_labels:
        # Find position of this verdict in rules
        pattern = rf"\*\*{re.escape(label)}\*\*"
        match = re.search(pattern, rules_text)
        if match:
            order.append((match.start(), label))

    # Also handle "if X does not apply" which indicates X comes first
    precondition_pattern = r"if\s+([^,]+)\s+does not apply"
    for match in re.finditer(precondition_pattern, rules_text, re.IGNORECASE):
        prior_verdict = match.group(1).strip()
        # Find which verdict this precondition belongs to
        # The prior verdict should come before this one in evaluation order

    # Sort by position
    order.sort(key=lambda x: x[0])

    # Extract just the labels, excluding "otherwise" (default) verdict
    # Default verdict comes last in evaluation
    ordered = [label for _, label in order]

    # Find the default verdict ("otherwise")
    default_pattern = r"\*\*([^*]+)\*\*\s*otherwise"
    default_match = re.search(default_pattern, rules_text)
    if default_match:
        default = default_match.group(1).strip()
        if default in ordered:
            ordered.remove(default)
            ordered.append(default)  # Move to end

    return ordered


def _extract_default_verdict(agenda: str) -> Optional[str]:
    """Extract the default verdict (the "otherwise" verdict).

    Args:
        agenda: Full agenda text

    Returns:
        Default verdict label, or None if not found
    """
    # Pattern: "**Verdict** otherwise"
    pattern = r"\*\*([^*]+)\*\*\s*otherwise"
    match = re.search(pattern, agenda)
    if match:
        return match.group(1).strip()
    return None


def _extract_count_thresholds(agenda: str, verdict_labels: List[str]) -> Tuple[Dict[str, List[int]], List[ConditionExpectation]]:
    """Extract count thresholds from verdict conditions.

    Looks for patterns like:
    - "2 or more reviews"
    - "3 or more complaints"
    - "N or more X"

    Args:
        agenda: Full agenda text
        verdict_labels: List of known verdict labels

    Returns:
        Tuple of (thresholds dict, list of condition expectations)
        thresholds: {verdict: [min_count values]}
    """
    thresholds: Dict[str, List[int]] = {v: [] for v in verdict_labels}
    conditions: List[ConditionExpectation] = []

    # Only look in the Verdict Rules section to avoid matching term definitions
    rules_match = re.search(r"## Verdict Rules\s*\n(.*)", agenda, re.DOTALL)
    if not rules_match:
        return thresholds, conditions

    rules_text = rules_match.group(1)

    # Find each verdict section and its conditions
    for label in verdict_labels:
        # Find section for this verdict in the rules text
        # Pattern: **Verdict** if/otherwise ... until next **Verdict** or end
        # Handle "A restaurant is considered **Verdict** if" pattern
        section_pattern = rf"(?:is\s+considered\s+)?\*\*{re.escape(label)}\*\*.*?(?=(?:is\s+considered\s+)?\*\*[^*]+\*\*|\Z)"
        section_match = re.search(section_pattern, rules_text, re.DOTALL | re.IGNORECASE)
        if not section_match:
            continue

        section = section_match.group(0)

        # Find all "N or more" patterns in this section
        # Pattern: "- N or more ..." (condition lines)
        condition_pattern = r"-\s*(\d+)\s+or\s+more\s+([^\n]+)"
        for match in re.finditer(condition_pattern, section, re.IGNORECASE):
            count = int(match.group(1))
            condition_text = match.group(2).strip()

            thresholds[label].append(count)
            conditions.append(ConditionExpectation(
                text=f"{count} or more {condition_text}",
                min_count=count,
                verdict=label,
            ))

        # Also find legacy patterns like "[min_count=N]" (if still present)
        legacy_pattern = r"-\s*\[min_count=(\d+)\]\s*([^\n]+)"
        for match in re.finditer(legacy_pattern, section):
            count = int(match.group(1))
            condition_text = match.group(2).strip()

            thresholds[label].append(count)
            conditions.append(ConditionExpectation(
                text=condition_text,
                min_count=count,
                verdict=label,
            ))

    return thresholds, conditions


def _extract_scoring_system(agenda: str) -> Optional[ScoringSystem]:
    """Extract V2/V3 scoring system from agenda.

    Looks for patterns like:
    - "Severe: 15 points"
    - "if score >= 8" or "Critical Risk if score >= 8"

    Args:
        agenda: Full agenda text

    Returns:
        ScoringSystem if scoring patterns found, else None
    """
    # Check if this is a scoring-based policy
    if "score" not in agenda.lower() and "points" not in agenda.lower():
        return None

    scoring = ScoringSystem()

    # Extract severity points: "Severe: 15 points" or "severe = 15"
    severity_pattern = r"(?:^|\n)\s*[-*]?\s*(\w+):\s*(\d+)\s*points?"
    for match in re.finditer(severity_pattern, agenda, re.IGNORECASE | re.MULTILINE):
        severity = match.group(1).lower()
        points = int(match.group(2))
        scoring.severity_points[severity] = points

    # Extract score thresholds: "Critical Risk if score >= 8"
    threshold_pattern = r"(\w+(?:\s+\w+)?)\s+if\s+score\s*>=?\s*(\d+)"
    for match in re.finditer(threshold_pattern, agenda, re.IGNORECASE):
        verdict = match.group(1).strip()
        threshold = int(match.group(2))
        scoring.thresholds[verdict] = threshold

    # Also try reverse pattern: "if score >= 8, assign Critical Risk"
    reverse_pattern = r"if\s+score\s*>=?\s*(\d+).*?(?:assign|→|->)\s*['\"]?(\w+(?:\s+\w+)?)['\"]?"
    for match in re.finditer(reverse_pattern, agenda, re.IGNORECASE):
        threshold = int(match.group(1))
        verdict = match.group(2).strip()
        scoring.thresholds[verdict] = threshold

    # Only return if we found meaningful data
    if scoring.severity_points or scoring.thresholds:
        return scoring

    return None


def parse_agenda(agenda: str) -> AgendaExpectations:
    """Parse agenda text to extract structured expectations.

    Args:
        agenda: The natural language agenda text (from prompt generator)

    Returns:
        AgendaExpectations with parsed structure
    """
    expectations = AgendaExpectations()

    # Extract verdict labels
    expectations.verdict_labels = _extract_verdict_labels(agenda)
    logger.debug(f"Parsed verdict labels: {expectations.verdict_labels}")

    # Extract evaluation order
    expectations.evaluation_order = _extract_evaluation_order(
        agenda, expectations.verdict_labels
    )
    logger.debug(f"Parsed evaluation order: {expectations.evaluation_order}")

    # Extract default verdict
    expectations.default_verdict = _extract_default_verdict(agenda)
    logger.debug(f"Parsed default verdict: {expectations.default_verdict}")

    # Extract count thresholds
    thresholds, conditions = _extract_count_thresholds(
        agenda, expectations.verdict_labels
    )
    expectations.count_thresholds = thresholds
    expectations.conditions = conditions
    logger.debug(f"Parsed count thresholds: {expectations.count_thresholds}")

    # Extract scoring system (V2/V3)
    expectations.scoring_system = _extract_scoring_system(agenda)
    if expectations.scoring_system:
        logger.debug(f"Parsed scoring system: {expectations.scoring_system}")

    return expectations


# =============================================================================
# CLI Interface for Testing
# =============================================================================

def main():
    """CLI interface for testing agenda parsing."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Parse agenda text and show expectations")
    parser.add_argument("input", type=Path, help="Input agenda text file (prompt)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    with open(args.input) as f:
        agenda = f.read()

    expectations = parse_agenda(agenda)

    # Print results
    print("=== Agenda Expectations ===")
    print(f"Verdict Labels: {expectations.verdict_labels}")
    print(f"Evaluation Order: {expectations.evaluation_order}")
    print(f"Default Verdict: {expectations.default_verdict}")
    print(f"Count Thresholds: {expectations.count_thresholds}")
    print(f"Has Scoring: {expectations.has_scoring()}")

    if expectations.scoring_system:
        print(f"Severity Points: {expectations.scoring_system.severity_points}")
        print(f"Score Thresholds: {expectations.scoring_system.thresholds}")

    print("\n=== Conditions ===")
    for cond in expectations.conditions:
        print(f"  [{cond.verdict}] min_count={cond.min_count}: {cond.text[:50]}...")


if __name__ == "__main__":
    main()
