"""CPU-only CRTR adapter checks; optional solver dependencies are mocked."""

import importlib
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import torch


class _RecordingDistance(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(self, states, goals):
        self.calls.append((states.clone(), goals.clone(), torch.is_grad_enabled()))
        return (states != goals).float().sum(dim=1) * self.scale


class ValueEstimatorSokobanJointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gin = types.ModuleType("gin")
        cls.gin.configurable = lambda value: value
        cls.loader_module = types.ModuleType("distance_model.sokoban_joint")
        cls.loader_module.load_joint_checkpoint = None
        with patch.dict(sys.modules, {"gin": cls.gin,
                                     "distance_model.sokoban_joint": cls.loader_module}):
            try:
                cls.adapter = importlib.import_module("distance_model.value_estimator_sokoban_joint")
            except ModuleNotFoundError:
                cls.adapter = None

    def setUp(self):
        self.assertIsNotNone(self.adapter, "The CRTR joint-distance solver adapter must exist")
        self.model = _RecordingDistance()
        self.checkpoint = {"config": {"width": 16, "T": 2, "attention_heads": 2}}
        self.loader = patch.object(self.adapter, "load_joint_checkpoint",
                                   return_value=(self.model, self.checkpoint))
        self.mock_load = self.loader.start()
        self.addCleanup(self.loader.stop)
        self.estimator = self.adapter.ValueEstimatorSokobanJoint(
            model=object(), metric="l2", checkpoint_path="joint.pt", device="cpu")
        self.states = np.zeros((2, 144), dtype=np.float32)
        self.states[0, :3] = 3
        self.states[1, :5] = 4
        self.goal = np.zeros((12, 12), dtype=np.float32)

    def construct(self):
        with redirect_stdout(io.StringIO()):
            self.estimator.construct_networks()

    def test_construct_uses_public_loader_and_sets_eval(self):
        self.construct()
        self.mock_load.assert_called_once_with("joint.pt", device=torch.device("cpu"))
        self.assertIs(self.estimator.model, self.model)
        self.assertIs(self.estimator.checkpoint, self.checkpoint)
        self.assertFalse(self.model.training)

    def test_single_distance_preserves_length_one_cpu_tensor(self):
        self.construct()
        distance = self.estimator.get_solved_distance(self.states[0], self.goal, action_in=2)
        torch.testing.assert_close(distance, torch.tensor([3.0]))
        self.assertEqual(distance.shape, (1,))
        self.assertEqual(distance.device.type, "cpu")
        self.assertFalse(distance.requires_grad)
        state, goal, gradients_enabled = self.model.calls[-1]
        self.assertEqual(state.shape, (1, 144))
        self.assertEqual(goal.shape, (1, 144))
        self.assertEqual(state.dtype, torch.uint8)
        self.assertFalse(gradients_enabled)

    def test_batch_one_is_not_squeezed_to_scalar(self):
        self.construct()
        distances = self.estimator.get_solved_distance_batch(self.states[:1], self.goal)
        self.assertEqual(distances.shape, (1,))
        self.assertEqual(len(distances), 1)
        torch.testing.assert_close(distances, torch.tensor([3.0]))

    def test_batch_repeats_goal_preserves_order_and_accepts_board_shape(self):
        self.construct()
        distances = self.estimator.get_solved_distance_batch(
            torch.from_numpy(self.states.reshape(2, 12, 12)), self.goal)
        torch.testing.assert_close(distances, torch.tensor([3.0, 5.0]))
        states, goals, gradients_enabled = self.model.calls[-1]
        self.assertEqual(states.shape, (2, 144))
        self.assertEqual(goals.shape, (2, 144))
        self.assertEqual(goals.dtype, torch.uint8)
        self.assertFalse(gradients_enabled)

    def test_empty_batch_returns_empty_cpu_vector_without_model_call(self):
        self.construct()
        distances = self.estimator.get_solved_distance_batch([], self.goal)
        self.assertEqual(distances.shape, (0,))
        self.assertEqual(distances.device.type, "cpu")
        self.assertEqual(len(self.model.calls), 0)

    def test_use_before_construct_and_missing_checkpoint_fail_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "construct_networks"):
            self.estimator.get_solved_distance(self.states[0], self.goal)
        missing = self.adapter.ValueEstimatorSokobanJoint(device="cpu")
        with self.assertRaisesRegex(ValueError, "checkpoint_path"):
            missing.construct_networks()

    def test_bindings_preserve_search_and_greedy_protocol(self):
        for mode, actions in (("search", 12), ("greedy", 1)):
            bindings = self.adapter.crtr_solve_bindings("joint.pt", mode=mode)
            self.assertIn("import value_estimator_sokoban_joint", bindings)
            self.assertIn("SolveJob.network = None", bindings)
            self.assertIn("BestFSSolver.value_estimator_class = @ValueEstimatorSokobanJoint", bindings)
            self.assertIn("BestFSSolver.checkpoint_path = 'joint.pt'", bindings)
            self.assertIn(f"SolveJob.n_actions = {actions}", bindings)
            self.assertIn("BestFSSolver.max_tree_size = 6000", bindings)
            self.assertIn("SolveJob.n_jobs = 1000", bindings)
            self.assertIn("SolveJob.n_parallel_workers = 1", bindings)
            self.assertIn("SolveJob.budget_checkpoints = [50, 100, 500, 1000]", bindings)
        with self.assertRaises(ValueError):
            self.adapter.crtr_solve_bindings("joint.pt", mode="unknown")


if __name__ == "__main__":
    unittest.main()
