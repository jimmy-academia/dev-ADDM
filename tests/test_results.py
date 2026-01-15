import json
import tempfile
import unittest
from pathlib import Path

from addm.experiments.manager import ExperimentManager


class TestResults(unittest.TestCase):
    def test_save_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            manager = ExperimentManager(
                run_name="test",
                results_dir=base,
                benchmark=False,
                method="direct",
                dataset="demo",
                model="mock",
            )
            run_paths = manager.create_run()
            manager.save_config(run_paths.config_path, {"foo": "bar"})
            manager.save_results(run_paths.results_path, [{"sample_id": "1", "output": "x"}])
            manager.save_metrics(run_paths.metrics_path, {"accuracy": 1.0})

            self.assertTrue(run_paths.config_path.exists())
            self.assertTrue(run_paths.results_path.exists())
            self.assertTrue(run_paths.metrics_path.exists())

            data = json.loads(run_paths.config_path.read_text(encoding="utf-8"))
            self.assertEqual(data["foo"], "bar")
