"""Task configuration and loading."""

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List

from addm.tasks.prompt_parser import (
    ParsedPrompt,
    parse_prompt_sections,
    parse_l0_fields,
    get_l0_field_order,
)


@dataclass
class TaskConfig:
    """Configuration for a task."""

    task_id: str
    domain: str
    prompt_text: str
    parsed_prompt: ParsedPrompt
    formula_module: ModuleType

    @property
    def l0_schema(self) -> Dict[str, Dict[str, str]]:
        """L0 field definitions parsed from prompt."""
        return parse_l0_fields(self.parsed_prompt.l0_primitives)

    @property
    def l0_fields(self) -> List[str]:
        """Ordered list of L0 field names."""
        return get_l0_field_order(self.l0_schema)

    @property
    def compute_ground_truth(self) -> Callable[..., Dict[str, Any]]:
        """Ground truth computation function from formula module."""
        return self.formula_module.compute_ground_truth

    @property
    def prompt_path(self) -> Path:
        """Path to the prompt file."""
        return Path(f"data/answers/{self.domain}/{self.task_id}_prompt.txt")

    def groundtruth_path(self, k: int = 50) -> Path:
        """Path to the ground truth file for given K."""
        return Path(f"data/answers/{self.domain}/{self.task_id}_K{k}_groundtruth.json")

    @property
    def cache_path(self) -> Path:
        """Path to the judgment cache."""
        return Path(f"data/answers/{self.domain}/cache.json")


def load_task(task_id: str, domain: str = "yelp") -> TaskConfig:
    """
    Load a task configuration.

    Args:
        task_id: Task identifier (e.g., "G1a")
        domain: Domain (e.g., "yelp", "amazon")

    Returns:
        TaskConfig with parsed prompt and formula module
    """
    # Load prompt
    prompt_path = Path(f"data/answers/{domain}/{task_id}_prompt.txt")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")

    prompt_text = prompt_path.read_text()

    # Parse prompt sections
    parsed_prompt = parse_prompt_sections(prompt_text)

    # Load formula module
    formula_module = importlib.import_module(f"addm.tasks.formulas.{task_id}")

    return TaskConfig(
        task_id=task_id,
        domain=domain,
        prompt_text=prompt_text,
        parsed_prompt=parsed_prompt,
        formula_module=formula_module,
    )
