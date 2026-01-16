"""Task configuration and loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from addm.tasks.prompt_parser import ParsedPrompt, parse_prompt_text, get_l0_fields


@dataclass
class TaskConfig:
    """Configuration for a task."""

    task_id: str
    domain: str
    prompt_text: str
    parsed_prompt: ParsedPrompt

    @property
    def l0_schema(self) -> Dict[str, Dict[str, str]]:
        """L0 field definitions."""
        return self.parsed_prompt.l0_fields

    @property
    def prompt_path(self) -> Path:
        """Path to the prompt file."""
        return Path(f"data/tasks/{self.domain}/{self.task_id}_prompt.txt")

    @property
    def groundtruth_path(self) -> Path:
        """Path to the ground truth file."""
        return Path(f"data/tasks/{self.domain}/{self.task_id}_groundtruth.json")

    @property
    def cache_path(self) -> Path:
        """Path to the judgment cache."""
        return Path(f"data/tasks/{self.domain}/cache.json")


def load_task(task_id: str, domain: str = "yelp") -> TaskConfig:
    """
    Load a task configuration.

    Args:
        task_id: Task identifier (e.g., "G1a")
        domain: Domain (e.g., "yelp", "amazon")

    Returns:
        TaskConfig with parsed prompt
    """
    # Load prompt
    prompt_path = Path(f"data/tasks/{domain}/{task_id}_prompt.txt")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")

    prompt_text = prompt_path.read_text()

    # Parse full prompt
    parsed_prompt = parse_prompt_text(prompt_text)

    return TaskConfig(
        task_id=task_id,
        domain=domain,
        prompt_text=prompt_text,
        parsed_prompt=parsed_prompt,
    )
