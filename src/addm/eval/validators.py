"""Validation utilities for outputs."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from addm.data.types import Sample


class Validator(ABC):
    name: str = "base"

    @abstractmethod
    def validate(self, sample: Sample, output: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class ExactMatchValidator(Validator):
    name = "exact"

    def validate(self, sample: Sample, output: Dict[str, Any]) -> Dict[str, Any]:
        expected = sample.expected
        prediction = output.get("output")
        return {
            "correct": expected is not None and prediction == expected,
            "expected": expected,
            "prediction": prediction,
        }
