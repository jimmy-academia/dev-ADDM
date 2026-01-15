"""Dataset types."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Sample:
    sample_id: str
    query: str
    context: Optional[str]
    metadata: Dict[str, Any]
    expected: Optional[Any] = None


@dataclass
class Dataset:
    name: str
    samples: List[Sample]
    metadata: Dict[str, Any]
