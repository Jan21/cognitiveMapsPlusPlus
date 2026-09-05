"""CPU-only transfer checks using tiny synthetic banks, not performance claims."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


class SokobanTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("distance_model.sokoban_joint"))
        from distance_model import sokoban_joint as module
        self.module = module
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_bank(self, name="bank", offset=0):
        rng = np.random.default_rng(3 + offset)
        states = rng.integers(0, 7, size=(8, 144), dtype=np.uint8)
        goals = np.ones_like(states)
        goals[states == 0] = 0
        goals[np.isin(states, [2, 3, 6])] = 3
        path = self.root / (name + ".npz")
        np.savez(path, states=states, goals=goals, dist=np.arange(1, 9, dtype=np.float32),
                 metadata_json=np.array(json.dumps({"purpose": "synthetic CPU test"})))
        return path

    def test_tile_encoding_and_joint_d4_preserve_pair_alignment(self):
        tiles = torch.arange(144).remainder(7).reshape(1, 144).to(torch.uint8)
        grid = self.module.onehot_grid(tiles)
        self.assertEqual(grid.shape, (1, 7, 12, 12))
        torch.testing.assert_close(grid.argmax(1).flatten(1), tiles.long())
        repeated = tiles.expand(8, -1).clone()
        left, right = self.module.augment_pair(repeated, repeated.clone(), np.arange(8))
        torch.testing.assert_close(left, right)
        for code in range(8):
            expected = np.rot90(tiles.numpy().reshape(12, 12), code % 4)
            if code >= 4:
                expected = np.fliplr(expected)
            np.testing.assert_array_equal(left[code].numpy().reshape(12, 12), expected)

    def test_tie_aware_spearman_and_undefined_constant_correlations(self):
        result = self.module.distance_metrics(np.array([1, 1, 2, 3]), np.array([1, 2, 2, 3]))
        self.assertAlmostEqual(result["spearman"], 5 / 6)
        self.assertEqual(result["mae"], .25)
        self.assertEqual(result["bias"], .25)
        self.assertEqual(result["rmse"], .5)
        self.assertIsNone(self.module.distance_metrics(np.arange(3), np.ones(3))["pearson"])
        with self.assertRaises(ValueError):
            self.module.distance_metrics(np.arange(3), np.array([0, 1, np.nan]))

    def test_sampling_rng_is_independent_of_model_initialization(self):
        bank = self.module.FrozenBank(self.make_bank())
        first, second = np.random.default_rng(9), np.random.default_rng(9)
        batch1 = self.module.draw_batch(bank, first, 6)
        torch.manual_seed(100)
        self.module.SokobanJoint(width=8, T=1, attention_heads=0)
        batch2 = self.module.draw_batch(bank, second, 6)
        for left, right in zip(batch1, batch2):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
        self.assertEqual(len(bank.provenance["sha256"]), 64)

    def test_gap_banks_allow_unsolved_goals_but_clean_test_requires_solved_goals(self):
        source = self.make_bank()
        with np.load(source) as archive:
            states, dist = archive["states"], archive["dist"]
        gap = self.root / "gap.npz"
        np.savez(gap, states=states, goals=states.copy(), dist=dist)
        bank = self.module.FrozenBank(gap, require_solved_goal=False)
        self.assertFalse(bank.provenance["require_solved_goal"])
        with self.assertRaisesRegex(ValueError, "synthetic"):
            self.module.FrozenBank(gap)

    def test_cpu_train_final_checkpoint_reload_and_saved_predictions_match(self):
        train = self.make_bank("train")
        val = self.make_bank("val", 1)
        test = self.make_bank("test", 2)
        prefix = self.root / "tiny"
        result = self.module.main(["--model", "joint", "--train-bank", str(train),
            "--val-bank", str(val), "--test-bank", str(test), "--out", str(prefix),
            "--steps", "2", "--bs", "4", "--eval-bs", "4", "--evalevery", "1",
            "--warmup", "1", "--width", "8", "--T", "1", "--attention-heads", "0",
            "--device", "cpu", "--torch-threads", "1"])
        model, checkpoint = self.module.load_joint_checkpoint(str(prefix) + ".pt", device="cpu")
        bank = self.module.FrozenBank(test)
        with torch.no_grad():
            prediction = model(torch.from_numpy(bank.states), torch.from_numpy(bank.goals)).numpy()
        with np.load(str(prefix) + ".predictions.npz") as saved:
            np.testing.assert_allclose(saved["test_distance"], prediction, atol=1e-5, rtol=1e-5)
        self.assertEqual(checkpoint["config"]["T"], 1)
        self.assertEqual(checkpoint["training"]["steps"], 2)
        self.assertEqual(checkpoint["banks"]["test"]["sha256"], bank.provenance["sha256"])
        self.assertEqual(result["nskip"], 0)
        self.assertEqual(model(torch.from_numpy(bank.states[:1]), torch.from_numpy(bank.goals[:1])).shape, (1,))
        events = [json.loads(line) for line in Path(str(prefix) + ".validation.jsonl").read_text().splitlines()]
        self.assertEqual([event["step"] for event in events], [1, 2])
        self.assertTrue(all(event["bank"] == "validation" for event in events))


if __name__ == "__main__":
    unittest.main()
