"""CPU-only protocol tests; no subprocess training or GPU access."""

import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
import sys
import subprocess
from pathlib import Path


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("distance_model.autoresearch_loop"))
        from distance_model import autoresearch_loop as loop
        self.loop = loop
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.state = loop.new_campaign(self.root, 24, now=100)

    def complete_discovery(self):
        for task in self.state["tasks"]:
            if task["group"] == "dgx":
                task.update(status="completed", metrics={"test_corr": .8, "test_mae": 5., "nskip": 0})
        anchor = next(task for task in self.state["tasks"] if task["id"] == "dgx-discovery-coat64-s0")
        anchor["metrics"] = {"test_corr": .9, "test_mae": 3., "nskip": 0}

    def test_only_final_metrics_can_promote_and_confirmations_are_fixed(self):
        self.complete_discovery()
        candidate = self.state["tasks"][1]
        candidate["metrics"].update(test_corr=.906, test_mae=3.5, best_corr=.99)
        self.state["tasks"][2]["metrics"].update(best_corr=1., best_mae=0.)
        self.loop.advance_group(self.state, "dgx", now=101)
        confirmation = [task for task in self.state["tasks"] if task["phase"] == "confirmation"]
        self.assertEqual(len(confirmation), 8)
        self.assertEqual({task["G"] for task in confirmation}, {11, 13})
        self.assertEqual({task["seed"] for task in confirmation}, {0, 1})
        self.assertEqual({task["eval_bank"] for task in confirmation}, {"historical"})
        self.assertEqual({task["steps"] for task in confirmation}, {160000})
        self.assertEqual({task["model"] for task in confirmation}, {"coat64", candidate["model"]})
        for task in confirmation:
            command = task["command"]
            self.assertEqual(command[command.index("--eval-bank") + 1], "historical")
            self.assertEqual(command[command.index("--steps") + 1], "160000")
            self.assertEqual(command[command.index("--evalevery") + 1], "0")
            self.assertEqual(int(command[command.index("--G") + 1]), task["G"])
            self.assertEqual(int(command[command.index("--Rmax") + 1]), 36 if task["G"] == 11 else 44)

    def test_best_checkpoint_cannot_trigger_promotion_or_mutation(self):
        self.complete_discovery()
        self.state["tasks"][1]["metrics"].update(best_corr=1., best_mae=0.)
        self.loop.advance_group(self.state, "dgx", now=101)
        self.assertEqual(self.state["groups"]["dgx"]["phase"], "done")
        self.assertEqual(len(self.state["tasks"]), 18)

    def test_near_miss_gets_only_one_two_mutation_round(self):
        self.complete_discovery()
        self.state["tasks"][1]["metrics"].update(test_corr=.89, test_mae=3.1)
        self.loop.advance_group(self.state, "dgx", now=101)
        mutations = [task for task in self.state["tasks"] if task["phase"] == "mutation"]
        self.assertEqual(len(mutations), 2)
        for task in mutations:
            self.assertEqual(task["eval_bank"], "validation")
            self.assertEqual(task["steps"], 80000)
            task.update(status="completed", metrics={"test_corr": .89, "test_mae": 3.1, "nskip": 0})
        self.loop.advance_group(self.state, "dgx", now=102)
        self.assertEqual(self.state["groups"]["dgx"]["phase"], "done")
        self.assertEqual(len(self.state["tasks"]), 20)

    def test_nonzero_exit_nonfinite_skips_and_missing_final_result_are_rejected(self):
        path = self.root / "result.log"
        valid = {"joint": {"test_corr": .91, "test_mae": 2., "nskip": 0, "best_corr": .99}}
        path.write_text("RESULT " + json.dumps(valid) + "\n")
        metrics, result = self.loop.parse_result(path, 0, "joint")
        self.assertEqual(metrics["test_corr"], .91)
        with self.assertRaises(ValueError):
            self.loop.parse_result(path, 1, "joint")
        for bad in ({"test_corr": float("nan"), "test_mae": 2., "nskip": 0},
                    {"test_corr": .9, "test_mae": 2., "nskip": 1},
                    {"best_corr": .99, "best_mae": .1, "nskip": 0}):
            path.write_text("RESULT " + json.dumps({"joint": bad}) + "\n")
            with self.assertRaises(ValueError):
                self.loop.parse_result(path, 0, "joint")

    def test_failed_anchor_blocks_selection_and_decision_is_recorded(self):
        self.complete_discovery()
        self.state["tasks"][0].update(status="failed", error="CUDA failure")
        self.loop.advance_group(self.state, "dgx", now=101)
        self.assertEqual(self.state["groups"]["dgx"]["phase"], "done")
        self.assertIn("anchor", self.state["decisions"][-1]["reason"])

    def test_source_changes_are_rejected(self):
        (self.root / "src").mkdir()
        self.loop.verify_sources(self.state)
        (self.root / "src" / "model.py").write_text("changed = True\n")
        with self.assertRaises(ValueError):
            self.loop.verify_sources(self.state)

    def test_gpu_telemetry_fails_closed_and_existing_process_blocks_claim(self):
        for memory, utilization, display in (("NaN", "0", "Disabled"),
                ("-1", "0", "Disabled"), ("0", "NaN", "Disabled"),
                ("21", "0", "Disabled"), ("0", "1", "Disabled"),
                ("0", "0", "Unknown"), ("0", "0", "[N/A]")):
            output = f"GPU-test, NVIDIA A40, {memory}, {utilization}, {display}\n"
            with self.subTest(output=output), patch.object(self.loop.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout=output)):
                self.assertFalse(self.loop.gpu_idle("GPU-test")[0])
        telemetry = subprocess.CompletedProcess([], 0, stdout="GPU-test, NVIDIA A40, 3, 0, Disabled\n")
        for processes, expected in (("", True), ("GPU-test, 123\n", False)):
            with patch.object(self.loop.subprocess, "run", side_effect=[telemetry, subprocess.CompletedProcess([], 0, stdout=processes)]):
                self.assertEqual(self.loop.gpu_idle("GPU-test")[0], expected)

    def test_worker_records_real_child_and_final_result_without_gpu_access(self):
        for directory in ("src", "logs", "artifacts", "pools"):
            (self.root / directory).mkdir()
        state = self.loop.new_campaign(self.root, 24)
        for task in state["tasks"]:
            if task["group"] == "dgx":
                task.update(status="failed", error="test fixture")
        anchor = state["tasks"][0]
        anchor.update(status="pending", error=None)
        payload = {"coat": {"test_corr": .9, "test_mae": 3., "nskip": 0}}
        anchor["command"] = [sys.executable, "-c", "print(" + repr("RESULT " + json.dumps(payload)) + ")"]
        self.loop.atomic_json(self.root / "state.json", state)
        with patch.object(self.loop, "gpu_idle", return_value=(True, "test card")), patch.object(self.loop, "POLL_SECONDS", .01):
            self.assertEqual(self.loop.worker(self.root, "dgx", "GPU-test"), 0)
        saved = json.loads((self.root / "state.json").read_text())
        completed = saved["tasks"][0]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["pid"], completed["pgid"])
        self.assertIn("--signal=TERM", completed["launch_command"])
        self.assertIn("--kill-after=10s", completed["launch_command"])
        self.assertGreater(completed["timeout_seconds"], 0)
        self.assertLessEqual(completed["timeout_seconds"], self.loop.TRIAL_TIMEOUT)
        self.assertEqual(completed["result"], payload)
        self.assertGreaterEqual(completed["end"], completed["start"])
        self.assertEqual(next(iter(saved["workers"].values()))["status"], "exited")


if __name__ == "__main__":
    unittest.main()
