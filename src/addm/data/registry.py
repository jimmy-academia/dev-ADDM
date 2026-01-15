"""Dataset registry to map dataset names to loaders."""

from pathlib import Path
from typing import Callable, Dict, Optional

from addm.data.types import Dataset
from addm.data.loaders import load_jsonl_dataset, load_json_dataset

LoaderFn = Callable[[Path, Optional[str]], Dataset]


class DatasetRegistry:
    def __init__(self) -> None:
        self._loaders: Dict[str, LoaderFn] = {
            ".jsonl": load_jsonl_dataset,
            ".json": load_json_dataset,
        }

    def register_extension(self, ext: str, loader: LoaderFn) -> None:
        self._loaders[ext.lower()] = loader

    def load(self, path: Path, name: Optional[str] = None) -> Dataset:
        ext = path.suffix.lower()
        if ext not in self._loaders:
            raise ValueError(f"Unsupported dataset extension: {ext}")
        return self._loaders[ext](path, name)
