"""Policy IR (Intermediate Representation) for query construction.

A PolicyIR represents the complete specification of a task, including:
- Overview: purpose and evidence scope
- Normative: terms and verdict rules (compilable)
- Output: expected output format
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .logic import Condition, OperatorType


@dataclass
class Overview:
    """Overview section of a policy (informative)."""

    title: str  # Title for the prompt (e.g., "Allergy Safety Risk Assessment")
    purpose: str  # What this policy does
    incident_definition: str  # What counts as an incident (for Definitions section)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Overview":
        """Create Overview from dictionary."""
        return cls(
            title=data.get("title", ""),
            purpose=data.get("purpose", ""),
            incident_definition=data.get("incident_definition", ""),
        )


@dataclass
class ScoringPoint:
    """A single scoring item (severity or modifier)."""

    label: str  # Display label (e.g., "Mild incident")
    points: int  # Point value
    description: Optional[str] = None  # Optional explanation

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoringPoint":
        """Create ScoringPoint from dictionary."""
        return cls(
            label=data.get("label", ""),
            points=data.get("points", 0),
            description=data.get("description"),
        )


@dataclass
class ScoringThreshold:
    """A threshold for verdict based on score."""

    verdict: str  # Verdict name (e.g., "Critical Risk")
    min_score: int  # Minimum score for this verdict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoringThreshold":
        """Create ScoringThreshold from dictionary."""
        return cls(
            verdict=data.get("verdict", ""),
            min_score=data.get("min_score", 0),
        )


@dataclass
class RecencyRule:
    """A recency weighting rule."""

    age: str  # Age description (e.g., "within 2 years")
    weight: str  # Weight description (e.g., "full point value")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecencyRule":
        """Create RecencyRule from dictionary."""
        return cls(
            age=data.get("age", ""),
            weight=data.get("weight", ""),
        )


@dataclass
class ScoringSystem:
    """Point-based scoring system for V2+ policies.

    Defines:
    - Severity points: points per incident severity level
    - Modifiers: bonus/penalty points for specific conditions
    - Recency rules: how incident age affects scoring
    - Thresholds: score thresholds for each verdict
    """

    severity_points: List[ScoringPoint]  # Points by severity
    modifiers: List[ScoringPoint]  # Bonus/penalty modifiers
    thresholds: List[ScoringThreshold]  # Score thresholds for verdicts
    description: Optional[str] = None  # Intro text for scoring section
    recency_rules: List[RecencyRule] = field(default_factory=list)  # Recency weighting

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoringSystem":
        """Create ScoringSystem from dictionary."""
        return cls(
            severity_points=[
                ScoringPoint.from_dict(p) for p in data.get("severity_points", [])
            ],
            modifiers=[ScoringPoint.from_dict(p) for p in data.get("modifiers", [])],
            thresholds=[
                ScoringThreshold.from_dict(t) for t in data.get("thresholds", [])
            ],
            description=data.get("description"),
            recency_rules=[
                RecencyRule.from_dict(r) for r in data.get("recency_rules", [])
            ],
        )


@dataclass
class DecisionRule:
    """A single verdict rule in the decision policy.

    Rules follow a precedence ladder:
    - CRITICAL if any of: ...
    - HIGH if CRITICAL does not apply and any of: ...
    - LOW otherwise, especially when: ...
    """

    verdict: str  # Target verdict (e.g., "Critical Risk")
    label: str  # Short label for prompt (e.g., "CRITICAL")
    logic: OperatorType  # Logic type (ANY, ALL)
    conditions: List[str]  # List of condition texts
    precondition: Optional[str] = None  # e.g., "CRITICAL does not apply"
    default: bool = False  # Is this the default/fallback verdict?
    especially_when: List[str] = field(default_factory=list)  # For default verdicts

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionRule":
        """Create DecisionRule from dictionary."""
        # Handle logic field
        logic_str = data.get("logic", "ANY")
        try:
            logic = OperatorType(logic_str)
        except ValueError:
            logic = OperatorType.ANY

        return cls(
            verdict=data["verdict"],
            label=data.get("label", data["verdict"].upper().replace(" ", "_")),
            logic=logic,
            conditions=data.get("conditions", []),
            precondition=data.get("precondition"),
            default=data.get("default", False),
            especially_when=data.get("especially_when", []),
        )


@dataclass
class DecisionPolicy:
    """Complete decision policy with verdicts and rules."""

    verdicts: List[str]  # Ordered list of verdicts
    ordered: bool  # Whether verdicts are ordered (higher = worse)
    rules: List[DecisionRule]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionPolicy":
        """Create DecisionPolicy from dictionary."""
        rules = [DecisionRule.from_dict(r) for r in data.get("rules", [])]
        return cls(
            verdicts=data.get("verdicts", []),
            ordered=data.get("ordered", True),
            rules=rules,
        )


@dataclass
class OutputSpec:
    """Output specification for the policy."""

    fields: Dict[str, Dict[str, Any]]  # Field name -> spec

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputSpec":
        """Create OutputSpec from dictionary."""
        return cls(fields=data)


@dataclass
class NormativeCore:
    """Normative (compilable) part of a policy."""

    terms: List[str]  # Term references (e.g., "shared:ACCOUNT_TYPE")
    decision: DecisionPolicy
    output: OutputSpec
    scoring: Optional[ScoringSystem] = None  # Optional scoring system for V2+

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormativeCore":
        """Create NormativeCore from dictionary."""
        # Handle term references
        terms = []
        for t in data.get("terms", []):
            if isinstance(t, dict) and "ref" in t:
                terms.append(t["ref"])
            elif isinstance(t, str):
                terms.append(t)

        # Handle optional scoring system
        scoring = None
        if "scoring" in data:
            scoring = ScoringSystem.from_dict(data["scoring"])

        return cls(
            terms=terms,
            decision=DecisionPolicy.from_dict(data.get("decision", {})),
            output=OutputSpec.from_dict(data.get("output", {})),
            scoring=scoring,
        )


@dataclass
class PolicyIR:
    """Complete Policy Intermediate Representation.

    A PolicyIR contains everything needed to generate:
    - The NL agenda prompt (Overview + Definitions + Verdict Rules)
    - Evaluation metrics
    - Validation scripts
    """

    policy_id: str  # Unique ID (e.g., "G1_allergy_V1")
    overview: Overview
    normative: NormativeCore
    extends: Optional[str] = None  # Parent policy for inheritance
    version: str = "1.0"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyIR":
        """Create PolicyIR from dictionary."""
        return cls(
            policy_id=data.get("policy_id", ""),
            overview=Overview.from_dict(data.get("overview", {})),
            normative=NormativeCore.from_dict(data.get("normative", {})),
            extends=data.get("extends"),
            version=data.get("version", "1.0"),
        )

    @classmethod
    def load(cls, yaml_path: Path) -> "PolicyIR":
        """Load PolicyIR from a YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def get_term_refs(self) -> List[str]:
        """Get all term references from this policy."""
        return self.normative.terms

    def get_verdicts(self) -> List[str]:
        """Get ordered list of possible verdicts."""
        return self.normative.decision.verdicts
