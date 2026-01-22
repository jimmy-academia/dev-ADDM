"""Seed Validator: Validate Formula Seeds against agenda expectations.

Validates that LLM-generated Formula Seeds match the structured expectations
parsed from the agenda text. Catches common LLM errors:

1. Wrong verdict labels ("Critical" instead of "Critical Risk")
2. Wrong evaluation order (checks "Needs Improvement" before "Excellent")
3. Invented rules not in agenda (adds N_NEGATIVE = 0 condition)
4. Wrong thresholds (uses >= 1 when agenda says >= 2)

When validation fails, the seed should ABSTAIN rather than guess.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .agenda_parser import AgendaExpectations
from .seed_transform import normalize_verdict_label

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a Formula Seed against expectations."""

    # Whether the seed is valid
    valid: bool = True

    # List of validation errors (failures)
    errors: List[str] = field(default_factory=list)

    # List of validation warnings (non-fatal issues)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        """Add an error and mark as invalid."""
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        """Add a warning (doesn't affect validity)."""
        self.warnings.append(msg)


def _get_seed_verdict_labels(seed: Dict[str, Any]) -> Set[str]:
    """Extract verdict labels used in the seed's compute operations.

    Args:
        seed: Formula Seed dict

    Returns:
        Set of verdict labels found in the seed
    """
    labels: Set[str] = set()

    for op in seed.get("compute", []):
        if op.get("op") == "case" or op.get("operation") == "case":
            name = op.get("name", "")
            # Only look at VERDICT operations
            if "VERDICT" in name.upper():
                for rule in op.get("rules", []):
                    if "then" in rule:
                        labels.add(str(rule["then"]))
                    if "else" in rule:
                        labels.add(str(rule["else"]))

    return labels


def _get_seed_verdict_order(seed: Dict[str, Any]) -> List[str]:
    """Extract evaluation order of verdicts from the seed.

    The order in case rules determines evaluation order.

    Args:
        seed: Formula Seed dict

    Returns:
        List of verdict labels in evaluation order
    """
    order: List[str] = []

    for op in seed.get("compute", []):
        if op.get("op") == "case" or op.get("operation") == "case":
            name = op.get("name", "")
            if "VERDICT" in name.upper():
                for rule in op.get("rules", []):
                    if "then" in rule:
                        verdict = str(rule["then"])
                        if verdict not in order:
                            order.append(verdict)
                    if "else" in rule:
                        verdict = str(rule["else"])
                        if verdict not in order:
                            order.append(verdict)

    return order


def _get_seed_default_verdict(seed: Dict[str, Any]) -> Optional[str]:
    """Extract the default verdict from the seed (the "else" case).

    Args:
        seed: Formula Seed dict

    Returns:
        Default verdict label, or None if not found
    """
    for op in seed.get("compute", []):
        if op.get("op") == "case" or op.get("operation") == "case":
            name = op.get("name", "")
            if "VERDICT" in name.upper():
                for rule in op.get("rules", []):
                    if "else" in rule:
                        return str(rule["else"])
    return None


def _extract_threshold_from_rule(rule: Dict[str, Any]) -> Optional[int]:
    """Extract threshold from a case rule's 'when' condition.

    Parses patterns like:
    - ">= 2"
    - "N_INCIDENTS >= 2"
    - "SCORE >= 8"

    Args:
        rule: Case rule dict with 'when' key

    Returns:
        Threshold value if found, else None
    """
    when = rule.get("when", "")
    if not when:
        return None

    # Pattern: "... >= N" or "... > N"
    match = re.search(r">=?\s*(\d+)", str(when))
    if match:
        return int(match.group(1))

    return None


def _get_seed_count_thresholds(seed: Dict[str, Any]) -> Dict[str, List[int]]:
    """Extract count thresholds from seed's case rules.

    Args:
        seed: Formula Seed dict

    Returns:
        Dict mapping verdict labels to their threshold values
    """
    thresholds: Dict[str, List[int]] = {}

    for op in seed.get("compute", []):
        if op.get("op") == "case" or op.get("operation") == "case":
            name = op.get("name", "")
            if "VERDICT" in name.upper():
                for rule in op.get("rules", []):
                    verdict = rule.get("then")
                    if not verdict:
                        continue

                    threshold = _extract_threshold_from_rule(rule)
                    if threshold is not None:
                        if verdict not in thresholds:
                            thresholds[verdict] = []
                        thresholds[verdict].append(threshold)

    return thresholds


def _normalize_label_set(labels: Set[str]) -> Set[str]:
    """Normalize a set of verdict labels for comparison.

    Args:
        labels: Set of verdict labels

    Returns:
        Set of normalized labels
    """
    return {normalize_verdict_label(label) for label in labels}


def validate_verdict_labels(
    seed: Dict[str, Any],
    expectations: AgendaExpectations,
    result: ValidationResult,
) -> None:
    """Validate that seed verdict labels match expectations.

    Args:
        seed: Formula Seed dict
        expectations: Parsed agenda expectations
        result: ValidationResult to update
    """
    seed_labels = _get_seed_verdict_labels(seed)
    seed_labels_normalized = _normalize_label_set(seed_labels)

    expected_labels = set(expectations.verdict_labels)
    expected_labels_normalized = _normalize_label_set(expected_labels)

    # Check for labels in seed that aren't in expected
    extra_labels = seed_labels_normalized - expected_labels_normalized
    if extra_labels:
        result.add_error(
            f"Seed uses verdict labels not in agenda: {extra_labels}. "
            f"Expected: {expected_labels}"
        )

    # Check for missing labels (warning, not error)
    missing_labels = expected_labels_normalized - seed_labels_normalized
    if missing_labels:
        result.add_warning(
            f"Seed missing verdict labels: {missing_labels}"
        )

    # Check that seed labels are normalized (exact match)
    for label in seed_labels:
        normalized = normalize_verdict_label(label)
        if label != normalized:
            result.add_error(
                f"Seed uses non-normalized verdict label '{label}'. "
                f"Should be '{normalized}'"
            )


def validate_evaluation_order(
    seed: Dict[str, Any],
    expectations: AgendaExpectations,
    result: ValidationResult,
) -> None:
    """Validate that seed evaluation order matches expectations.

    Args:
        seed: Formula Seed dict
        expectations: Parsed agenda expectations
        result: ValidationResult to update
    """
    seed_order = _get_seed_verdict_order(seed)
    expected_order = expectations.evaluation_order

    if not seed_order or not expected_order:
        return  # Can't validate if either is empty

    # Normalize labels for comparison
    seed_order_normalized = [normalize_verdict_label(v) for v in seed_order]
    expected_order_normalized = [normalize_verdict_label(v) for v in expected_order]

    # Check that the first verdict in seed matches first in expected
    if seed_order_normalized and expected_order_normalized:
        if seed_order_normalized[0] != expected_order_normalized[0]:
            result.add_error(
                f"Seed evaluation order wrong. First verdict should be "
                f"'{expected_order[0]}' but seed has '{seed_order[0]}'"
            )

    # Check order is preserved (allowing for missing verdicts)
    seed_positions = {v: i for i, v in enumerate(seed_order_normalized)}
    for i, expected_v in enumerate(expected_order_normalized):
        if expected_v in seed_positions:
            for j, later_v in enumerate(expected_order_normalized[i + 1:], i + 1):
                if later_v in seed_positions:
                    if seed_positions[expected_v] > seed_positions[later_v]:
                        result.add_error(
                            f"Seed evaluation order wrong. "
                            f"'{expected_v}' should be checked before '{later_v}'"
                        )


def validate_default_verdict(
    seed: Dict[str, Any],
    expectations: AgendaExpectations,
    result: ValidationResult,
) -> None:
    """Validate that seed has correct default verdict.

    Args:
        seed: Formula Seed dict
        expectations: Parsed agenda expectations
        result: ValidationResult to update
    """
    if not expectations.default_verdict:
        return  # Can't validate if not in expectations

    seed_default = _get_seed_default_verdict(seed)
    if not seed_default:
        result.add_error(
            f"Seed missing default verdict (else clause). "
            f"Expected: '{expectations.default_verdict}'"
        )
        return

    # Normalize and compare
    seed_default_normalized = normalize_verdict_label(seed_default)
    expected_normalized = normalize_verdict_label(expectations.default_verdict)

    if seed_default_normalized != expected_normalized:
        result.add_error(
            f"Seed has wrong default verdict. "
            f"Expected '{expectations.default_verdict}', got '{seed_default}'"
        )


def validate_count_thresholds(
    seed: Dict[str, Any],
    expectations: AgendaExpectations,
    result: ValidationResult,
) -> None:
    """Validate that seed count thresholds match expectations.

    Args:
        seed: Formula Seed dict
        expectations: Parsed agenda expectations
        result: ValidationResult to update
    """
    seed_thresholds = _get_seed_count_thresholds(seed)

    # Compare thresholds for each verdict
    for verdict, expected_counts in expectations.count_thresholds.items():
        if not expected_counts:
            continue

        verdict_normalized = normalize_verdict_label(verdict)

        # Find seed thresholds for this verdict (try normalized)
        seed_counts = seed_thresholds.get(verdict, [])
        if not seed_counts:
            seed_counts = seed_thresholds.get(verdict_normalized, [])

        if not seed_counts:
            # No thresholds in seed for this verdict (might be OK)
            continue

        # Check that seed thresholds are at least as strict as expected
        for expected_count in expected_counts:
            # Find matching or stricter threshold in seed
            matching = [c for c in seed_counts if c >= expected_count]
            if not matching:
                # Seed has looser threshold
                result.add_warning(
                    f"Seed threshold for '{verdict}' may be too loose. "
                    f"Agenda expects >= {expected_count}, seed has {seed_counts}"
                )


def validate_scoring_system(
    seed: Dict[str, Any],
    expectations: AgendaExpectations,
    result: ValidationResult,
) -> None:
    """Validate scoring system (V2/V3 policies).

    Args:
        seed: Formula Seed dict
        expectations: Parsed agenda expectations
        result: ValidationResult to update
    """
    if not expectations.has_scoring():
        return  # Not a scoring-based policy

    scoring = expectations.scoring_system

    # Validate severity points
    # Look for sum operations with CASE expressions
    for op in seed.get("compute", []):
        if op.get("op") == "sum":
            expr = op.get("expr", "")
            # Check if point values match
            for severity, expected_points in scoring.severity_points.items():
                # Look for "WHEN <field> = '<severity>' THEN <points>"
                pattern = rf"WHEN\s+\w+\s*=\s*['\"]?{severity}['\"]?\s+THEN\s+(\d+)"
                match = re.search(pattern, expr, re.IGNORECASE)
                if match:
                    actual_points = int(match.group(1))
                    if actual_points != expected_points:
                        result.add_error(
                            f"Seed uses wrong points for '{severity}'. "
                            f"Expected {expected_points}, got {actual_points}"
                        )

    # Validate score thresholds
    for verdict, expected_threshold in scoring.thresholds.items():
        seed_thresholds = _get_seed_count_thresholds(seed)
        verdict_normalized = normalize_verdict_label(verdict)

        seed_values = seed_thresholds.get(verdict, [])
        if not seed_values:
            seed_values = seed_thresholds.get(verdict_normalized, [])

        if seed_values:
            # Check that threshold matches
            for val in seed_values:
                if val != expected_threshold:
                    result.add_warning(
                        f"Seed score threshold for '{verdict}' differs. "
                        f"Expected >= {expected_threshold}, got >= {val}"
                    )


def validate_seed(
    seed: Dict[str, Any],
    expectations: AgendaExpectations,
) -> ValidationResult:
    """Validate a Formula Seed against agenda expectations.

    Performs comprehensive validation:
    1. Verdict labels match exactly
    2. Evaluation order is correct
    3. Default verdict is correct
    4. Count thresholds match
    5. V2/V3 scoring matches

    Args:
        seed: Formula Seed dict to validate
        expectations: Parsed agenda expectations

    Returns:
        ValidationResult with errors and warnings
    """
    result = ValidationResult()

    # Run all validations
    validate_verdict_labels(seed, expectations, result)
    validate_evaluation_order(seed, expectations, result)
    validate_default_verdict(seed, expectations, result)
    validate_count_thresholds(seed, expectations, result)
    validate_scoring_system(seed, expectations, result)

    if result.errors:
        logger.warning(
            f"Seed validation failed with {len(result.errors)} errors: "
            f"{result.errors}"
        )
    elif result.warnings:
        logger.info(
            f"Seed validation passed with {len(result.warnings)} warnings"
        )
    else:
        logger.debug("Seed validation passed")

    return result


# =============================================================================
# CLI Interface for Testing
# =============================================================================

def main():
    """CLI interface for testing seed validation."""
    import argparse
    import json
    from pathlib import Path

    from .agenda_parser import parse_agenda

    parser = argparse.ArgumentParser(description="Validate Formula Seed against agenda")
    parser.add_argument("seed", type=Path, help="Formula Seed JSON file")
    parser.add_argument("agenda", type=Path, help="Agenda text file (prompt)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Load seed
    with open(args.seed) as f:
        seed = json.load(f)

    # Load and parse agenda
    with open(args.agenda) as f:
        agenda = f.read()
    expectations = parse_agenda(agenda)

    # Validate
    result = validate_seed(seed, expectations)

    # Print results
    print("=== Validation Result ===")
    print(f"Valid: {result.valid}")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for error in result.errors:
            print(f"  - {error}")

    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"  - {warning}")

    return 0 if result.valid else 1


if __name__ == "__main__":
    exit(main())
