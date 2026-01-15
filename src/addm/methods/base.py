"""Method interfaces."""

from abc import ABC, abstractmethod
from typing import Dict, Any

from addm.data.types import Sample
from addm.llm import LLMService


class Method(ABC):
    name: str = "base"

    @abstractmethod
    async def run_sample(self, sample: Sample, llm: LLMService) -> Dict[str, Any]:
        raise NotImplementedError
