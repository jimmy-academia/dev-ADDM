import json
import tempfile
import unittest
from pathlib import Path

from addm.data.registry import DatasetRegistry


class TestDatasetLoader(unittest.TestCase):
    def test_load_jsonl_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            rows = [
                {"id": "1", "query": "q1", "context": "c1", "expected": "a1"},
                {"id": "2", "query": "q2", "context": "c2"},
            ]
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            registry = DatasetRegistry()
            dataset = registry.load(path)
            self.assertEqual(dataset.name, "data")
            self.assertEqual(len(dataset.samples), 2)
            self.assertEqual(dataset.samples[0].sample_id, "1")
