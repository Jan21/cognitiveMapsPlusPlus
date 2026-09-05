"""CPU import-isolation and checkpoint-format checks for the lean CRTR runner."""

import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch


class SokobanRunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("distance_model.sokoban_solve"),
                             "The isolated Sokoban runner must exist")
        from distance_model import sokoban_solve
        self.runner = sokoban_solve

    def test_pinned_namespaces_ignore_shadowing_regular_module(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules):
            root = Path(temporary) / "crtr"
            shadow = Path(temporary) / "shadow"
            shadow.mkdir()
            (shadow / "search.py").write_text("raise RuntimeError('shadow module executed')\n")
            for name in ("search", "utils", "envs"):
                (root / name).mkdir(parents=True)
            (root / "networks.py").write_text("SOURCE = 'local'\n")
            (root / "search" / "marker.py").write_text("SOURCE = 'local solver'\n")
            for name in list(sys.modules):
                if name.split(".")[0] in ("search", "utils", "envs", "networks"):
                    del sys.modules[name]
            with patch.object(sys, "path", [str(shadow), *sys.path]):
                self.runner.pin_crtr_namespaces(root)
                module = importlib.import_module("search.marker")
            self.assertEqual(module.SOURCE, "local solver")
            self.assertEqual(sys.modules["search"].__path__, [str(root / "search")])

    def test_preloaded_shadow_is_rejected_before_pinning_any_namespace(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules):
            root = Path(temporary)
            for name in ("search", "utils", "envs"):
                (root / name).mkdir()
            (root / "networks.py").write_text("SOURCE = 'local'\n")
            for name in list(sys.modules):
                if name.split(".")[0] in ("search", "utils", "envs", "networks"):
                    del sys.modules[name]
            sys.modules["search"] = types.ModuleType("search")
            with self.assertRaisesRegex(RuntimeError, "already loaded"):
                self.runner.pin_crtr_namespaces(root)
            self.assertNotIn("utils", sys.modules)
            self.assertNotIn("envs", sys.modules)

    def test_file_sha256_hashes_full_bytes(self):
        self.assertTrue(hasattr(self.runner, "file_sha256"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.py"
            path.write_bytes(b"abc")
            self.assertEqual(self.runner.file_sha256(path),
                             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
            path.write_bytes(b"")
            self.assertEqual(self.runner.file_sha256(path),
                             "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")

    def test_summary_preserves_generated_and_expanded_budget_counts(self):
        self.assertTrue(hasattr(self.runner, "solve_metrics"))
        job = types.SimpleNamespace(
            solved_boards=2, all_boards=4,
            budget_solved={50: 0, 100: 1, 500: 2, 1000: 2},
            budget_exp_solved={50: 1, 100: 2, 500: 2, 1000: 2})
        summary = self.runner.solve_metrics(job)
        self.assertEqual(summary["solved"], 2)
        self.assertEqual(summary["boards_evaluated"], 4)
        self.assertEqual(summary["solved_rate"], 0.5)
        self.assertEqual(summary["solved_rate_by_node_budget"],
                         {"50": 0.0, "100": 0.25, "500": 0.5, "1000": 0.5})
        self.assertEqual(summary["solved_rate_by_expanded_node_budget"],
                         {"50": 0.25, "100": 0.5, "500": 0.5, "1000": 0.5})

    def test_supervised_wrapper_extracts_only_exact_network_prefix(self):
        weight = torch.ones(2, 3)
        checkpoint = {"model": "supervised", "state_dict": {"network.input_layer.weight": weight}}
        extracted = self.runner.raw_supervised_state(checkpoint)
        self.assertEqual(set(extracted), {"input_layer.weight"})
        self.assertIs(extracted["input_layer.weight"], weight)
        with self.assertRaises(ValueError):
            self.runner.raw_supervised_state({"model": "joint", "state_dict": checkpoint["state_dict"]})
        with self.assertRaises(ValueError):
            self.runner.raw_supervised_state({"model": "supervised", "state_dict": {"other.weight": weight}})


if __name__ == "__main__":
    unittest.main()

