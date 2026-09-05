"""CPU checks for reusable pool generation and independent validation maps."""

import argparse
import copy
import tempfile
import unittest

import numpy as np

from distance_model import switchyard as sw

try:
    from distance_model import autoresearch_trial as trial
except ImportError:
    trial = None


def small_args():
    return argparse.Namespace(G=5, ngate=2, nlever=1, nchute=1, seed=3,
        nmaps=8, split="map", wire1=False, noplate=False, nopush=False,
        gatesopen=False, poolq=3, bfsmax=80, Rmax=8, Rtrain=0, T=3,
        ncot=0, gcurr=0, cotsup=0.0, curriculum=0, lencurr=0)


class PoolCacheTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(trial, "The autoresearch trial wrapper must exist")
        self.a = small_args()
        self.yards, self.tr, self.te = sw.make_yards(self.a)

    def test_cache_hit_matches_uncached_arrays_and_post_rng_state(self):
        direct_rng = np.random.default_rng(71)
        expected = sw.build_pool(self.a, direct_rng, self.yards, self.tr, 8)
        expected_next = direct_rng.integers(2**30, size=20)
        with tempfile.TemporaryDirectory() as directory:
            cache = trial.PoolCache(directory, sw.build_pool)
            for attempt in range(2):
                rng = np.random.default_rng(71)
                actual = cache(self.a, rng, self.yards, self.tr, 8)
                for left, right in zip(actual, expected):
                    np.testing.assert_array_equal(left, right)
                np.testing.assert_array_equal(rng.integers(2**30, size=20), expected_next)
            self.assertEqual(cache.builds, 1)
            self.assertEqual(cache.hits, 1)

    def test_validation_replaces_only_heldout_maps_and_reuses_train_key(self):
        validation, train_ids, eval_ids = trial.make_bank_yards(self.a, "validation")
        historical, _, _ = trial.make_bank_yards(self.a, "historical")
        self.assertEqual(train_ids, self.tr)
        self.assertEqual(eval_ids, self.te)
        with tempfile.TemporaryDirectory() as directory:
            cache = trial.PoolCache(directory, sw.build_pool)
            key = lambda yards, ids: cache.key(self.a, np.random.default_rng(19), yards, ids, 8)
            self.assertEqual(key(validation, self.tr), key(historical, self.tr))
            self.assertNotEqual(key(validation, self.te), key(historical, self.te))
            self.assertEqual(key(validation, self.te), key(trial.make_bank_yards(self.a, "validation")[0], self.te))

    def test_key_covers_map_structure_sampling_rng_order_and_cot(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = trial.PoolCache(directory, sw.build_pool)
            key = lambda a, yards, ids, seed=19, cap=8, cot=False: cache.key(
                a, np.random.default_rng(seed), yards, ids, cap, cot=cot)
            original = key(self.a, self.yards, self.tr)
            changed_yards = copy.deepcopy(self.yards)
            changed_yards[self.tr[0]].wall[0, 0] ^= True
            self.assertNotEqual(original, key(self.a, changed_yards, self.tr))
            self.assertNotEqual(original, key(self.a, self.yards, self.tr[::-1]))
            self.assertNotEqual(original, key(self.a, self.yards, self.tr, seed=20))
            self.assertNotEqual(original, key(self.a, self.yards, self.tr, cap=7))
            self.assertNotEqual(original, key(self.a, self.yards, self.tr, cot=True))
            for field in ("poolq", "bfsmax"):
                altered = copy.copy(self.a)
                setattr(altered, field, getattr(altered, field) + 1)
                self.assertNotEqual(original, key(altered, self.yards, self.tr))

    def test_cot_cache_preserves_waypoint_arrays(self):
        expected = sw.build_pool(self.a, np.random.default_rng(71), self.yards, self.tr, 8, cot=True)
        with tempfile.TemporaryDirectory() as directory:
            cache = trial.PoolCache(directory, sw.build_pool)
            cache(self.a, np.random.default_rng(71), self.yards, self.tr, 8, cot=True)
            actual = cache(self.a, np.random.default_rng(71), self.yards, self.tr, 8, cot=True)
            self.assertEqual(len(actual), 6)
            for left, right in zip(actual, expected):
                np.testing.assert_array_equal(left, right)


if __name__ == "__main__":
    unittest.main()
