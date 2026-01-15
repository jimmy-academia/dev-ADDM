import unittest
from pathlib import Path

from addm.cli import parse_args


class TestCLI(unittest.TestCase):
    def test_parse_args_defaults(self):
        args = parse_args(["--data", "sample.jsonl"])
        self.assertEqual(args.method, "direct")
        self.assertEqual(args.provider, "openai")
        self.assertEqual(args.model, "gpt-4o-mini")
        self.assertIsInstance(args.data_path, Path)
