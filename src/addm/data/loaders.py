"""Dataset loaders."""

from pathlib import Path
from typing import Dict, Any, List, Optional

from addm.data.types import Dataset, Sample
from addm.utils.io import read_jsonl, read_json


def _build_samples(rows: List[Dict[str, Any]]) -> List[Sample]:
    samples = []
    for idx, row in enumerate(rows, start=1):
        sample_id = str(row.get("id") or row.get("sample_id") or idx)
        samples.append(
            Sample(
                sample_id=sample_id,
                query=row.get("query") or row.get("prompt") or "",
                context=row.get("context"),
                metadata=row.get("metadata", {}),
                expected=row.get("expected"),
            )
        )
    return samples


def load_jsonl_dataset(path: Path, name: Optional[str] = None) -> Dataset:
    rows = read_jsonl(path)
    samples = _build_samples(rows)
    return Dataset(name=name or path.stem, samples=samples, metadata={"path": str(path)})


def load_json_dataset(path: Path, name: Optional[str] = None) -> Dataset:
    data = read_json(path)
    rows = data.get("samples", data if isinstance(data, list) else [])
    samples = _build_samples(rows)
    metadata = {"path": str(path)}
    if isinstance(data, dict):
        metadata.update({k: v for k, v in data.items() if k != "samples"})
    return Dataset(name=name or path.stem, samples=samples, metadata=metadata)
