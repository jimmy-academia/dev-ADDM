"""Term definitions for query construction.

Terms are reusable definitions that appear in the Definitions section
of generated agenda prompts.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class TermType(Enum):
    """Type of term value."""

    ENUM = "enum"  # Categorical with fixed values
    BOOLEAN = "boolean"  # true/false
    NUMERIC = "numeric"  # Integer or float
    STRING = "string"  # Free text


@dataclass
class TermValue:
    """Single value in an enumeration term."""

    id: str  # Machine identifier (e.g., "mild")
    description: str  # Definition text for prompt
    label: Optional[str] = None  # Human-readable label (defaults to id)
    synonyms: List[str] = field(default_factory=list)  # Alternative names
    cues: List[str] = field(default_factory=list)  # Extraction hints

    def __post_init__(self):
        if self.label is None:
            self.label = self.id

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TermValue":
        """Create TermValue from dictionary."""
        return cls(
            id=data["id"],
            description=data["description"],
            label=data.get("label"),
            synonyms=data.get("synonyms", []),
            cues=data.get("cues", []),
        )


@dataclass
class Term:
    """A reusable term definition.

    Terms define the vocabulary used in agenda prompts. Each term has:
    - A unique ID (e.g., "INCIDENT_SEVERITY")
    - A human-readable name
    - A type (enum, boolean, numeric, string)
    - For enums: a list of allowed values with descriptions
    """

    id: str  # Unique ID (e.g., "INCIDENT_SEVERITY")
    name: str  # Human-readable name
    term_type: TermType
    description: Optional[str] = None  # Overall term description
    values: List[TermValue] = field(default_factory=list)  # For enums
    default: Optional[str] = None  # Default value
    domain: str = "shared"  # Which library it belongs to

    @classmethod
    def from_dict(cls, term_id: str, data: Dict[str, Any], domain: str = "shared") -> "Term":
        """Create Term from dictionary (as loaded from YAML)."""
        term_type = TermType(data.get("type", "enum"))

        values = []
        if "values" in data:
            for v in data["values"]:
                values.append(TermValue.from_dict(v))

        return cls(
            id=term_id,
            name=data.get("name", term_id.replace("_", " ").title()),
            term_type=term_type,
            description=data.get("description"),
            values=values,
            default=data.get("default"),
            domain=domain,
        )

    def get_value_ids(self) -> List[str]:
        """Get list of valid value IDs for enum terms."""
        return [v.id for v in self.values]


class TermLibrary:
    """Collection of terms, loaded from YAML files.

    Supports loading terms from multiple domains (shared, allergy, etc.)
    and resolving references like "shared:ACCOUNT_TYPE" or "allergy:INCIDENT_SEVERITY".
    """

    def __init__(self):
        self.domains: Dict[str, Dict[str, Term]] = {}

    def load_domain(self, domain: str, yaml_path: Path) -> None:
        """Load terms from a YAML file into a domain."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        self.domains[domain] = {}
        for term_id, term_data in data.items():
            self.domains[domain][term_id] = Term.from_dict(term_id, term_data, domain)

    def resolve(self, ref: str) -> Term:
        """Resolve a term reference.

        Args:
            ref: Reference like "shared:ACCOUNT_TYPE" or just "ACCOUNT_TYPE"

        Returns:
            The resolved Term

        Raises:
            KeyError: If term not found
        """
        if ":" in ref:
            domain, term_id = ref.split(":", 1)
        else:
            # Search all domains
            for domain, terms in self.domains.items():
                if ref in terms:
                    return terms[ref]
            raise KeyError(f"Term '{ref}' not found in any domain")

        if domain not in self.domains:
            raise KeyError(f"Domain '{domain}' not loaded")
        if term_id not in self.domains[domain]:
            raise KeyError(f"Term '{term_id}' not found in domain '{domain}'")

        return self.domains[domain][term_id]

    def get_all_terms(self, domain: Optional[str] = None) -> List[Term]:
        """Get all terms, optionally filtered by domain."""
        if domain:
            return list(self.domains.get(domain, {}).values())
        return [term for terms in self.domains.values() for term in terms.values()]

    def list_domains(self) -> List[str]:
        """List all loaded domains."""
        return list(self.domains.keys())
