"""Evaluator for experiment results."""

from typing import Any, Dict, List

from addm.data.types import Dataset
from addm.eval.validators import Validator


class Evaluator:
    def __init__(self, validator: Validator) -> None:
        self.validator = validator

    def evaluate(self, dataset: Dataset, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        outputs_by_id = {o["sample_id"]: o for o in outputs}
        total = 0
        correct = 0
        detailed = []
        for sample in dataset.samples:
            if sample.sample_id not in outputs_by_id:
                continue
            total += 1
            result = self.validator.validate(sample, outputs_by_id[sample.sample_id])
            detailed.append({"sample_id": sample.sample_id, **result})
            if result.get("correct"):
                correct += 1
        accuracy = correct / total if total else 0.0
        return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "details": detailed,
        }
