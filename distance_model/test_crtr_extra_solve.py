"""Adapter checks; set CRTR_TEST_ROOT to run the actual original solver fixture."""

from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import random
import sys
import types
import unittest

import numpy as np


@contextmanager
def isolated_crtr_imports():
    names = ("search", "utils", "envs", "networks")
    saved = {name: value for name, value in sys.modules.items() if name.split(".")[0] in names}
    for name in saved:
        del sys.modules[name]
    original_gin = sys.modules.get("gin")
    shim = original_gin is None and importlib.util.find_spec("gin") is None
    if shim:
        # Only gin's class-registration decorator is absent locally; all actual
        # solver constructor arguments are explicit and algorithm code is original.
        gin = types.ModuleType("gin")
        gin.configurable = lambda value: value
        sys.modules["gin"] = gin
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name.split(".")[0] in names:
                del sys.modules[name]
        sys.modules.update(saved)
        if shim:
            sys.modules.pop("gin", None)


class ExtraSolverTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("distance_model.crtr_extra_solve"))
        from distance_model import crtr_extra_solve as module
        self.module = module

    def test_digit_invalid_moves_are_self_loops_and_goal_is_case_specific(self):
        from distance_model.crtr_extra_envs import digit_encode
        board = np.ones((3, 3), dtype=np.uint8)
        env = self.module.ExtraPuzzleEnv("digit", size=3)
        start, goal = digit_encode(board, 0), digit_encode(board, 2)
        env.set_problem(start, goal)
        self.assertEqual(env.get_all_actions(), [0, 1, 2, 3])
        unchanged, _, done, _ = env.step(env.reset(), 0)
        self.assertEqual(unchanged, tuple(start.flat))
        self.assertFalse(done)
        middle, _, _, _ = env.step(unchanged, 1)
        arrived, _, done, _ = env.step(middle, 1)
        self.assertTrue(done)
        self.assertEqual(arrived, tuple(goal.flat))

    def test_lights_actions_and_already_solved_root(self):
        env = self.module.ExtraPuzzleEnv("lights", size=3)
        goal = np.zeros((3, 3), dtype=np.uint8)
        env.set_problem(goal, goal)
        self.assertEqual(env.get_all_actions(), list(range(9)))
        record = self.module.solve_case(None, env, goal, goal, index=0)
        self.assertTrue(record["solved"])
        self.assertFalse(record["solver_invoked"])
        self.assertEqual((record["nodes"], record["expanded_nodes"], record["length"]), (1, 0, 0))

    def test_strict_generated_budget_is_distinct_from_expanded_budget(self):
        rows = [{"solved": True, "nodes": 50, "expanded_nodes": 2, "length": 1},
                {"solved": True, "nodes": 99, "expanded_nodes": 80, "length": 2},
                {"solved": False, "nodes": 1, "expanded_nodes": 1, "length": None}]
        metrics = self.module.aggregate_records(rows)
        self.assertEqual(metrics["solved_rate_by_node_budget"]["50"], 0)
        self.assertEqual(metrics["solved_rate_by_node_budget"]["100"], 2 / 3)
        self.assertEqual(metrics["solved_rate_by_expanded_node_budget"]["50"], 1 / 3)

    @unittest.skipUnless(os.environ.get("CRTR_TEST_ROOT"), "set CRTR_TEST_ROOT to an original CRTR checkout")
    def test_original_bestfs_solves_tiny_fixtures_with_exact_values_and_original_counters(self):
        from distance_model.crtr_extra_envs import lights_apply, lights_distances, digit_distances, digit_encode
        root = Path(os.environ["CRTR_TEST_ROOT"])
        with isolated_crtr_imports():
            solver_class, policy_class, sources = self.module.load_original_solver(root)
            self.assertEqual(Path(sources["search.solver"]).resolve(), (root / "search/solver.py").resolve())
            for kind in ("lights", "digit"):
                env = self.module.ExtraPuzzleEnv(kind, size=3)
                goal = np.zeros((3, 3), dtype=np.uint8)
                if kind == "lights":
                    initial = lights_apply(goal[None], np.array([4]))[0]
                else:
                    board = np.ones((3, 3), dtype=np.uint8)
                    initial, goal = digit_encode(board, 0), digit_encode(board, 1)
                env.set_problem(initial, goal)

                class ExactValues:
                    def construct_networks(self):
                        pass

                    def get_solved_distance(self, state, goal_state):
                        if kind == "lights":
                            return float(lights_distances(np.asarray(state).reshape(1, 3, 3), np.asarray(goal_state).reshape(3, 3), size=3)[0])
                        position = int(np.flatnonzero(np.asarray(state) > 6)[0])
                        target = int(np.flatnonzero(np.asarray(goal_state) > 6)[0])
                        return float(digit_distances(board, goal=target)[position])

                random.seed(0)
                solver = self.module.make_solver(solver_class, policy_class, env, ExactValues(),
                                                  n_actions=1, max_tree_size=50)
                result = self.module.solve_case(solver, env, initial, goal, index=0)
                self.assertTrue(result["solved"])
                self.assertTrue(result["solver_invoked"])
                self.assertEqual(result["length"], 1)
                # These are the original early-goal branch's counters, including
                # its additional expanded-node increment for the terminal node.
                self.assertEqual(result["nodes"], 2)
                self.assertEqual(result["expanded_nodes"], 2)


if __name__ == "__main__":
    unittest.main()
