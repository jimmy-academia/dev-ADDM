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
    """A threshold for verdict based on score.

    Supports K-specific thresholds via min_score_by_k/max_score_by_k.
    Example YAML:
        - verdict: "High Risk"
          min_score: 20
          min_score_by_k:
            25: 3
            50: 5
            100: 10
    """

    verdict: str  # Verdict name (e.g., "Critical Risk")
    min_score: Optional[int] = None  # Minimum score for this verdict (default, K=200)
    max_score: Optional[int] = None  # Maximum score for this verdict (default, K=200)
    min_score_by_k: Dict[int, int] = field(default_factory=dict)  # K-specific min thresholds
    max_score_by_k: Dict[int, int] = field(default_factory=dict)  # K-specific max thresholds

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoringThreshold":
        """Create ScoringThreshold from dictionary."""
        # Parse K-specific thresholds, converting string keys to int
        raw_min_by_k = data.get("min_score_by_k", {})
        min_score_by_k = {int(k): v for k, v in raw_min_by_k.items()}

        raw_max_by_k = data.get("max_score_by_k", {})
        max_score_by_k = {int(k): v for k, v in raw_max_by_k.items()}

        return cls(
            verdict=data.get("verdict", ""),
            min_score=data.get("min_score"),
            max_score=data.get("max_score"),
            min_score_by_k=min_score_by_k,
            max_score_by_k=max_score_by_k,
        )

    def get_min_score(self, k: int = 200) -> Optional[int]:
        """Get min_score for a specific K value."""
        return self.min_score_by_k.get(k, self.min_score)

    def get_max_score(self, k: int = 200) -> Optional[int]:
        """Get max_score for a specific K value."""
        return self.max_score_by_k.get(k, self.max_score)


@dataclass
class RecencyRule:
    """A recency weighting rule.

    Supports both human-readable (age, weight) and machine-readable
    (max_age_years, weight_multiplier) fields for V3 recency weighting.
    """

    age: str  # Human-readable age description (e.g., "Within 6 months")
    weight: str  # Human-readable weight description (e.g., "full point value")
    max_age_years: Optional[float] = None  # Machine-readable: max age in years (e.g., 0.5)
    weight_multiplier: Optional[float] = None  # Machine-readable: weight multiplier (e.g., 1.0)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecencyRule":
        """Create RecencyRule from dictionary."""
        return cls(
            age=data.get("age", ""),
            weight=data.get("weight", ""),
            max_age_years=data.get("max_age_years"),
            weight_multiplier=data.get("weight_multiplier"),
        )


@dataclass
class ModifierFieldMapping:
    """Maps a judgment field value to a scoring modifier label."""

    field: str  # Judgment field name (e.g., "named_positively")
    value: str  # Field value that triggers this modifier (e.g., "true")
    label: str  # Modifier label from scoring.modifiers (e.g., "Named with praise")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModifierFieldMapping":
        """Create ModifierFieldMapping from dictionary."""
        return cls(
            field=data.get("field", ""),
            value=data.get("value", ""),
            label=data.get("label", ""),
        )


@dataclass
class FieldMapping:
    """Maps judgment fields to severity labels for scoring.

    This allows V2+ policies to define how extracted judgment fields
    (from term libraries) map to the scoring system's severity_points.

    Example YAML:
        field_mapping:
          severity_field: "attentiveness"
          value_mappings:
            excellent: "Exceptional service experience"
            good: "Good service"
          modifier_mappings:
            - field: "named_positively"
              value: "true"
              label: "Named with praise"
    """

    severity_field: str  # Primary field for base score (e.g., "incident_severity")
    value_mappings: Dict[str, str]  # Field value -> severity label
    modifier_mappings: List[ModifierFieldMapping] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FieldMapping":
        """Create FieldMapping from dictionary."""
        return cls(
            severity_field=data.get("severity_field", ""),
            value_mappings=data.get("value_mappings", {}),
            modifier_mappings=[
                ModifierFieldMapping.from_dict(m)
                for m in data.get("modifier_mappings", [])
            ],
        )


@dataclass
class ScoringSystem:
    """Point-based scoring system for V2+ policies.

    Defines:
    - Severity points: points per incident severity level
    - Modifiers: bonus/penalty points for specific conditions
    - Recency rules: how incident age affects scoring
    - Thresholds: score thresholds for each verdict
    - Field mapping: how judgment fields map to severity labels
    - Reference date: date from which to calculate review age (V3)
    """

    severity_points: List[ScoringPoint]  # Points by severity
    modifiers: List[ScoringPoint]  # Bonus/penalty modifiers
    thresholds: List[ScoringThreshold]  # Score thresholds for verdicts
    description: Optional[str] = None  # Intro text for scoring section
    recency_rules: List[RecencyRule] = field(default_factory=list)  # Recency weighting
    reference_date: Optional[str] = None  # Reference date for V3 recency (e.g., "2022-01-01")
    field_mapping: Optional[FieldMapping] = None  # Maps judgment fields to severity labels

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoringSystem":
        """Create ScoringSystem from dictionary."""
        field_mapping = None
        if "field_mapping" in data:
            field_mapping = FieldMapping.from_dict(data["field_mapping"])

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
            reference_date=data.get("reference_date"),
            field_mapping=field_mapping,
        )


@dataclass
class StructuredCondition:
    """A structured condition for evaluation.

    Condition types:
    - count_threshold: count matching judgments >= min_count
    - exists: at least one matching judgment exists
    - compound: judgment matches filter AND has additional field

    Example YAML:
        - type: count_threshold
          filter:
            incident_severity: severe
            account_type: firsthand
          min_count: 30
          min_count_by_k:
            25: 4
            50: 8
            100: 15
    """

    type: str  # count_threshold, exists, compound
    filter: Dict[str, Any]  # Field -> value requirements
    min_count: int = 1  # For count_threshold (default, used for K=200)
    min_count_by_k: Dict[int, int] = field(default_factory=dict)  # K-specific thresholds
    requires: Optional[Dict[str, Any]] = None  # For compound conditions
    text: Optional[str] = None  # Optional NL description for prompts

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredCondition":
        """Create StructuredCondition from dictionary."""
        # Parse min_count_by_k, converting string keys to int
        raw_by_k = data.get("min_count_by_k", {})
        min_count_by_k = {int(k): v for k, v in raw_by_k.items()}

        return cls(
            type=data.get("type", "count_threshold"),
            filter=data.get("filter", {}),
            min_count=data.get("min_count", 1),
            min_count_by_k=min_count_by_k,
            requires=data.get("requires"),
            text=data.get("text"),
        )

    def get_min_count(self, k: int = 200) -> int:
        """Get min_count for a specific K value.

        Falls back to default min_count if K not specified in min_count_by_k.
        """
        return self.min_count_by_k.get(k, self.min_count)

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        result = {"type": self.type, "filter": self.filter}
        if self.type == "count_threshold":
            result["min_count"] = self.min_count
            if self.min_count_by_k:
                result["min_count_by_k"] = self.min_count_by_k
        if self.requires:
            result["requires"] = self.requires
        if self.text:
            result["text"] = self.text
        return result


# Type alias for conditions (can be string or structured)
ConditionType = Any  # Union[str, Dict, StructuredCondition]


@dataclass
class DecisionRule:
    """A single verdict rule in the decision policy.

    Rules follow a precedence ladder:
    - CRITICAL if any of: ...
    - HIGH if CRITICAL does not apply and any of: ...
    - LOW otherwise, especially when: ...

    Conditions can be:
    - String: NL text for prompt generation
    - Dict/StructuredCondition: Structured condition for evaluation
    """

    verdict: str  # Target verdict (e.g., "Critical Risk")
    label: str  # Short label for prompt (e.g., "CRITICAL")
    logic: OperatorType  # Logic type (ANY, ALL)
    conditions: List[ConditionType]  # List of conditions (str or structured)
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

        # Parse conditions - can be strings or structured dicts
        raw_conditions = data.get("conditions", [])
        conditions: List[ConditionType] = []
        for c in raw_conditions:
            if isinstance(c, str):
                conditions.append(c)
            elif isinstance(c, dict):
                # Keep as dict for now (evaluator will handle)
                conditions.append(c)

        return cls(
            verdict=data["verdict"],
            label=data.get("label", data["verdict"].upper().replace(" ", "_")),
            logic=logic,
            conditions=conditions,
            precondition=data.get("precondition"),
            default=data.get("default", False),
            especially_when=data.get("especially_when", []),
        )

    def get_nl_conditions(self) -> List[str]:
        """Get only the NL string conditions (for prompt generation)."""
        return [c for c in self.conditions if isinstance(c, str)]

    def get_structured_conditions(self) -> List[Dict[str, Any]]:
        """Get only the structured conditions (for evaluation)."""
        return [c for c in self.conditions if isinstance(c, dict)]


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

    Format options:
    - markdown (default): Definitions → Verdict Rules
    - reorder_v1: Verdict Rules → Definitions
    - reorder_v2: Interleaved structure
    - xml: XML-structured format
    - prose: Flowing narrative format
    """

    policy_id: str  # Unique ID (e.g., "G1_allergy_V1")
    overview: Overview
    normative: NormativeCore
    extends: Optional[str] = None  # Parent policy for inheritance
    version: str = "1.0"
    format: str = "markdown"  # Output format (markdown, reorder_v1, reorder_v2, xml, prose)
    agenda_override: Optional[str] = None  # Pre-rendered agenda to use instead of generating

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyIR":
        """Create PolicyIR from dictionary."""
        return cls(
            policy_id=data.get("policy_id", ""),
            overview=Overview.from_dict(data.get("overview", {})),
            normative=NormativeCore.from_dict(data.get("normative", {})),
            extends=data.get("extends"),
            version=data.get("version", "1.0"),
            format=data.get("format", "markdown"),
            agenda_override=data.get("agenda_override"),
        )

    @classmethod
    def load(cls, yaml_path: Path) -> "PolicyIR":
        """Load PolicyIR from a YAML file, handling inheritance via 'extends'.

        If the policy has 'extends: T1_P1', it will load T1/P1.yaml as the base
        and merge the child's overrides on top.
        """
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        # Handle inheritance
        extends = data.get("extends")
        if extends:
            # Parse extends reference: "T1_P1" -> T1/P1.yaml
            # Format: T{n}_P{m} or T{n}P{m}
            extends_clean = extends.replace("_", "")  # T1_P1 -> T1P1
            if len(extends_clean) >= 4 and extends_clean[0] == "T":
                tier = extends_clean[:2]  # T1
                variant = extends_clean[2:]  # P1
                parent_path = yaml_path.parent / f"../{tier}/{variant}.yaml"
                parent_path = parent_path.resolve()

                if parent_path.exists() and parent_path != yaml_path.resolve():
                    # Load parent (recursively handles nested inheritance)
                    parent_policy = cls.load(parent_path)

                    # Merge: child overrides parent
                    merged_data = cls._merge_policy_data(parent_policy, data)
                    return cls.from_dict(merged_data)

        return cls.from_dict(data)

    @classmethod
    def _merge_policy_data(cls, parent: "PolicyIR", child_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge child policy data with parent PolicyIR.

        Child values override parent. Missing sections inherit from parent.
        """
        # Start with parent data converted to dict
        merged = {
            "policy_id": child_data.get("policy_id", parent.policy_id),
            "version": child_data.get("version", parent.version),
            "format": child_data.get("format", parent.format),
            "extends": child_data.get("extends"),
            "overview": {
                "title": parent.overview.title,
                "purpose": parent.overview.purpose,
                "incident_definition": parent.overview.incident_definition,
            },
            "normative": {
                "terms": parent.normative.terms.copy() if parent.normative.terms else [],
                "decision": {
                    "verdicts": parent.normative.decision.verdicts.copy(),
                    "ordered": parent.normative.decision.ordered,
                    "rules": [
                        {
                            "verdict": r.verdict,
                            "label": r.label,
                            "logic": r.logic.value if hasattr(r.logic, 'value') else str(r.logic),
                            "conditions": list(r.conditions) if r.conditions else [],
                            "precondition": r.precondition,
                            "default": r.default,
                            "especially_when": list(r.especially_when) if r.especially_when else [],
                        }
                        for r in parent.normative.decision.rules
                    ],
                },
                "output": {
                    "fields": parent.normative.output.fields.copy() if parent.normative.output.fields else {},
                } if parent.normative.output else {},
            },
        }

        # Override with child data where present
        if "overview" in child_data:
            for key in ["title", "purpose", "incident_definition"]:
                if key in child_data["overview"]:
                    merged["overview"][key] = child_data["overview"][key]

        if "normative" in child_data:
            if "terms" in child_data["normative"]:
                merged["normative"]["terms"] = child_data["normative"]["terms"]
            if "decision" in child_data["normative"]:
                # Full override of decision section if provided
                merged["normative"]["decision"] = child_data["normative"]["decision"]
            if "output" in child_data["normative"]:
                merged["normative"]["output"] = child_data["normative"]["output"]

        return merged

    def get_term_refs(self) -> List[str]:
        """Get all term references from this policy."""
        return self.normative.terms

    def get_verdicts(self) -> List[str]:
        """Get ordered list of possible verdicts."""
        return self.normative.decision.verdicts
