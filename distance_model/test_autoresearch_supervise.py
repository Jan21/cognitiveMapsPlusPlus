"""Pure CPU checks for bounded replacement scheduling; never launch jobs."""

import importlib.util
import unittest


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("distance_model.autoresearch_supervise"))
        from distance_model import autoresearch_supervise as module
        self.module = module

    def test_slurm_counts_only_tracked_expanded_array_elements(self):
        rows = "131967_0|ar-direct|R\n131967_3|ar-direct|R\n131975_1|ar-direct-retry|PD\n99999_0|ar-direct-auto|R\n"
        self.assertEqual(self.module.active_slurm_slots(rows, {"131967", "131975"}), 3)
        self.assertEqual(self.module.replacement_slots(7, 3), 5)
        self.assertEqual(self.module.replacement_slots(2, 7), 1)
        self.assertEqual(self.module.replacement_slots(9, 8), 0)

    def test_compacted_or_malformed_tracked_slurm_rows_fail_closed(self):
        with self.assertRaises(ValueError):
            self.module.active_slurm_slots("131967_[0-3]|ar-direct|PD\n", {"131967"})
        with self.assertRaises(ValueError):
            self.module.active_slurm_slots("131967_0\n", {"131967"})

    def test_replay_preserves_restart_budget_cooldown_and_submitted_ids(self):
        events = [{"event": "dgx_launch_attempt", "time": 12, "gpu": "GPU-one"},
                  {"event": "slurm_submit_attempt", "time": 30},
                  {"event": "slurm_submitted", "time": 31, "job_id": "42"}]
        budget = self.module.replay_budget(events)
        self.assertEqual(budget["dgx_restarts"], 1)
        self.assertEqual(budget["last_gpu_launch"]["GPU-one"], 12)
        self.assertEqual(budget["slurm_submissions"], 1)
        self.assertEqual(budget["last_slurm_submit"], 30)
        self.assertEqual(budget["submitted_jobs"], {"42"})
        self.assertFalse(budget["slurm_pending_submission"])

    def test_unresolved_slurm_attempt_blocks_submissions_after_restart(self):
        events = [{"event": "slurm_submit_attempt", "time": 30},
                  {"event": "error", "time": 31, "message": "sbatch timed out"}]
        budget = self.module.replay_budget(events)
        self.assertTrue(budget["slurm_pending_submission"])
        self.assertEqual(budget["slurm_submissions"], 1)
        self.assertEqual(budget["submitted_jobs"], set())

    def test_running_task_blocks_gpu_even_when_telemetry_may_be_idle(self):
        state = {"tasks": [{"status": "running", "gpu": "GPU-one"},
                           {"status": "completed", "gpu": "GPU-two"}]}
        self.assertEqual(self.module.unresolved_gpus(state), {"GPU-one"})


if __name__ == "__main__":
    unittest.main()
