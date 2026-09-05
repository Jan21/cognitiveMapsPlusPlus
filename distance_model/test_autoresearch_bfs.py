"""Exact ordered-graph equivalence checks for optional compiled BFS."""

import argparse
import copy
import unittest

import numpy as np

try:
    from . import switchyard as sw
except ImportError:
    import switchyard as sw

try:
    if __package__:
        from . import autoresearch_bfs as fast
    else:
        import autoresearch_bfs as fast
except ImportError:
    fast = None


class ExactBFSTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(fast, "The optional exact BFS accelerator must exist")

    def test_ordered_traversals_cover_mechanics_and_cap_overshoot(self):
        mechanics = [{}, {"nopush": True}, {"gatesopen": True},
                     {"noplate": True}, {"wire1": True},
                     {"nopush": True, "gatesopen": True}]
        for seed in range(24):
            ngate = seed % 5
            yard = sw.Yard(5 if seed % 2 else 7, ngate, 1 + seed % 3, seed % 3,
                           np.random.default_rng(seed), **mechanics[seed % len(mechanics)])
            rng = np.random.default_rng(seed + 91)
            for _ in range(3):
                src = yard.rand_state(rng)
                for cap in (0, 1, 2, 3, 4, 5, 8, 31, 150):
                    with self.subTest(seed=seed, src=src, cap=cap):
                        expected = yard.bfs(src, cap)
                        actual = fast.bfs(yard, src, cap)
                        self.assertEqual(list(actual.items()), list(expected.items()))

    def test_complete_reachable_sets_match_in_order(self):
        for seed in range(8):
            yard = sw.Yard(5, 3, 2, 1, np.random.default_rng(seed))
            src = yard.rand_state(np.random.default_rng(seed + 77))
            self.assertEqual(list(fast.bfs(yard, src, 100000).items()),
                             list(yard.bfs(src, 100000).items()))

    def test_pool_bytes_and_rng_state_match_reference(self):
        args = argparse.Namespace(G=7, ngate=3, nlever=2, nchute=1, seed=7,
            nmaps=8, split="map", wire1=False, noplate=False, nopush=False,
            gatesopen=False, poolq=12, bfsmax=500, T=4, ncot=0)
        yards, train_ids, _ = sw.make_yards(args)
        original = sw.Yard.bfs
        rng = np.random.default_rng(27)
        expected = sw.build_pool(args, rng, yards, train_ids, 20)
        state = copy.deepcopy(rng.bit_generator.state)
        try:
            sw.Yard.bfs = fast.bfs
            accelerated_rng = np.random.default_rng(27)
            actual = sw.build_pool(args, accelerated_rng, yards, train_ids, 20)
        finally:
            sw.Yard.bfs = original
        for left, right in zip(actual, expected):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(state, accelerated_rng.bit_generator.state)


if __name__ == "__main__":
    unittest.main()
