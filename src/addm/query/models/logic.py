"""Logic operators for query construction.

Operators define the logical constructs used in verdict rules,
such as ANY, ALL, COUNT, EXISTS, etc.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class OperatorType(Enum):
    """Types of logical operators."""

    # Quantifiers
    ANY = "ANY"  # At least one condition true
    ALL = "ALL"  # All conditions true
    NONE = "NONE"  # No conditions true

    # Aggregates
    COUNT = "COUNT"  # Count matching items
    SUM = "SUM"  # Sum of values
    MAX = "MAX"  # Maximum value
    MIN = "MIN"  # Minimum value

    # Temporal
    WITHIN_WINDOW = "WITHIN_WINDOW"  # Within N days/months
    MOST_RECENT = "MOST_RECENT"  # Most recent N items
    SINCE = "SINCE"  # Since date/event

    # Existence
    EXISTS = "EXISTS"  # At least one exists

    # Exception
    EXCEPT = "EXCEPT"  # Exclude from consideration


@dataclass
class Operator:
    """An operator definition from the logic library."""

    id: str  # Operator ID (e.g., "ANY")
    op_type: OperatorType
    description: str
    template: Optional[str] = None  # NL template (e.g., "any of the following")
    params: List[str] = field(default_factory=list)  # Required parameters

    @classmethod
    def from_dict(cls, op_id: str, data: Dict[str, Any]) -> "Operator":
        """Create Operator from dictionary."""
        return cls(
            id=op_id,
            op_type=OperatorType(op_id),
            description=data.get("description", ""),
            template=data.get("template"),
            params=data.get("params", []),
        )


@dataclass
class Condition:
    """A single condition in a rule.

    Conditions are the atomic units of verdict rules. They can be:
    - Simple text conditions: "A single severe firsthand incident is reported"
    - Structured conditions: { field: ACCOUNT_TYPE, op: "==", value: "firsthand" }
    """

    text: str  # The condition text (for NL generation)
    field: Optional[str] = None  # Optional: field reference
    op: Optional[str] = None  # Optional: comparison operator
    value: Any = None  # Optional: comparison value

    @classmethod
    def from_str(cls, text: str) -> "Condition":
        """Create a text-only condition."""
        return cls(text=text)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Condition":
        """Create a structured condition."""
        if isinstance(data, str):
            return cls.from_str(data)
        return cls(
            text=data.get("text", ""),
            field=data.get("field"),
            op=data.get("op"),
            value=data.get("value"),
        )


class LogicLibrary:
    """Collection of operators loaded from YAML."""

    def __init__(self):
        self.operators: Dict[str, Operator] = {}

    def load(self, yaml_path: Path) -> None:
        """Load operators from a YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        for op_id, op_data in data.items():
            self.operators[op_id] = Operator.from_dict(op_id, op_data)

    def get(self, op_id: str) -> Operator:
        """Get an operator by ID."""
        if op_id not in self.operators:
            raise KeyError(f"Operator '{op_id}' not found")
        return self.operators[op_id]

    def list_operators(self) -> List[str]:
        """List all operator IDs."""
        return list(self.operators.keys())
