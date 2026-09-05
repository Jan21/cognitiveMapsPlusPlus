"""CPU tests for frozen-bank LightsOut/DigitJumper distance transfer."""

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F


class ExtraTrainerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("distance_model.crtr_extra_train"))
        from distance_model import crtr_extra_train
        self.module = crtr_extra_train
        torch.set_num_threads(1)

    def bank(self, directory, name, env, rows=4):
        size, classes = (7, 2) if env == "lights" else (20, 13)
        rng = np.random.default_rng(sum(name.encode()))
        low = 0 if env == "lights" else 1
        states = rng.integers(low, classes if env == "lights" else 7,
                              size=(rows, size, size), dtype=np.uint8)
        goals = np.zeros_like(states) if env == "lights" else states.copy()
        if env == "digit":
            states[:, 0, 0] += 6
            goals[:, -1, -1] += 6
        path = Path(directory) / (name + ".npz")
        np.savez_compressed(path, states=states, goals=goals,
                            dist=np.arange(rows, dtype=np.float32),
                            gap=np.arange(rows, dtype=np.float32) + 1)
        return path

    def test_joint_both_envs_flat_grid_shape_and_exact_latent_motion(self):
        for env, size, classes in (("lights", 7, 2), ("digit", 20, 13)):
            with self.subTest(env=env):
                model = self.module.ExtraJoint(env, width=8, T=2, attention_heads=0)
                state = torch.randint(0, classes, (1, size, size), dtype=torch.uint8)
                goal = state.flip(-1)
                prediction, trajectory = model(state, goal, ret_states=True)
                movement = sum(torch.linalg.vector_norm(b - a, dim=1).sum((1, 2))
                               for a, b in zip(trajectory, trajectory[1:]))
                expected = F.softplus(model.core.scale) * movement
                torch.testing.assert_close(prediction, expected, rtol=0, atol=0)
                torch.testing.assert_close(model(state.flatten(1), goal.flatten(1)), prediction)
                torch.testing.assert_close(model(state[:, None], goal[:, None]), prediction)
                self.assertEqual(prediction.shape, (1,))
                self.assertEqual(model.core.in_channels, classes)
                self.assertTrue(all(name.startswith("core.") for name, _ in model.named_parameters()))
                self.assertTrue(torch.equal(model(state, goal, Trun=0), torch.zeros(1)))
                prediction.sum().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))

    def test_frozen_bank_gap_selection_and_invalid_tiles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.bank(directory, "bank", "lights")
            exact = self.module.FrozenExtraBank(path, "lights", targets="exact")
            gap = self.module.FrozenExtraBank(path, "lights", targets="gap")
            np.testing.assert_array_equal(gap.targets, exact.dist + 1)
            np.testing.assert_array_equal(exact.targets, exact.dist)
            self.assertEqual(exact.states.shape, (4, 7, 7))
            with np.load(path) as archive:
                data = dict(archive)
            data["states"][0, 0, 0] = 2
            np.savez_compressed(path, **data)
            with self.assertRaisesRegex(ValueError, "tile"):
                self.module.FrozenExtraBank(path, "lights")

    def test_iid_rows_do_not_depend_on_torch_model_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            bank = self.module.FrozenExtraBank(self.bank(directory, "bank", "digit"), "digit")
            rng1, rng2 = np.random.default_rng(42), np.random.default_rng(42)
            indices1 = self.module.draw_batch(bank, rng1, 11)[-1]
            self.module.ExtraJoint("digit", width=8, T=1, attention_heads=0)
            indices2 = self.module.draw_batch(bank, rng2, 11)[-1]
            np.testing.assert_array_equal(indices1, indices2)

    def test_lights_dense_exact_architecture_and_main_cnn_unchanged(self):
        self.assertEqual(self.module.supervised_config("lights", "dense"),
                         dict(network_class="LNDenseNet", input_size=98, repr_dim=50,
                              hidden_size=512, depth=8))
        self.assertEqual(self.module.supervised_config("lights", "supervised"),
                         dict(network_class="LNConvNet", input_size=4, repr_dim=50,
                              hidden_size=64, depth=8, baseline=True))
        with self.assertRaisesRegex(ValueError, "LightsOut-only"):
            self.module.supervised_config("digit", "dense")

    def test_exact_crtr_baselines_train_reload_and_match_joint_sampling(self):
        root = Path("/tmp/claude-1000/-home-jan-projects-CIIRC-colabs-Alma-cognitiveMapsPlusPlus/"
                    "2473a2ed-f5b5-4c80-a8a6-8423dc6e063f/scratchpad/CRTR")
        if not (root / "networks.py").is_file():
            self.skipTest("original CRTR networks.py reference is unavailable")
        gin = types.ModuleType("gin")
        gin.configurable = lambda target=None, **unused: target if target is not None else lambda value: value
        for env, kind, expected_bins in (("lights", "supervised", 50), ("lights", "dense", 50),
                                          ("digit", "supervised", 400), ("digit", "cnn", 400)):
            with self.subTest(env=env, kind=kind), tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"gin": gin}):
                paths = {name: self.bank(directory, name, env) for name in ("train", "val", "test")}
                output = Path(directory) / "baseline"
                args = ["--env", env, "--model", kind, "--crtr-root", str(root),
                        "--train-bank", str(paths["train"]), "--val-bank", str(paths["val"]),
                        "--test-bank", str(paths["test"]), "--out", str(output),
                        "--steps", "1", "--bs", "2", "--eval-bs", "2", "--evalevery", "0",
                        "--device", "cpu", "--torch-threads", "1"]
                with redirect_stdout(io.StringIO()):
                    result = self.module.main(args)
                model, checkpoint = self.module.load_extra_checkpoint(str(output) + ".pt", crtr_root=root)
                bank = self.module.FrozenExtraBank(paths["test"], env)
                states, goals = torch.from_numpy(bank.states[:1]), torch.from_numpy(bank.goals[:1])
                with torch.inference_mode():
                    logits = model(states, goals)
                    expected = model.network(torch.cat((states.flatten(1), goals.flatten(1)), dim=1).float())
                self.assertEqual(logits.shape, (1, expected_bins))
                torch.testing.assert_close(logits, expected, rtol=0, atol=0)
                _, predictions = self.module.evaluate(model, bank, kind, torch.device("cpu"), 2)
                with np.load(str(output) + ".predictions.npz") as saved:
                    for estimator in ("argmax", "expectation"):
                        np.testing.assert_array_equal(predictions[estimator], saved["test_" + estimator])
                import hashlib
                rng = np.random.default_rng(0)
                indices = rng.integers(0, 4, size=2)
                self.assertEqual(result["sample_trace_sha256"], hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest())
                self.assertEqual(checkpoint["config"], self.module.supervised_config(env, kind))

    def test_both_envs_cpu_train_final_checkpoint_before_test_and_reload(self):
        for env in ("lights", "digit"):
            with self.subTest(env=env), tempfile.TemporaryDirectory() as directory:
                paths = {name: self.bank(directory, name, env) for name in ("train", "val", "test")}
                output = Path(directory) / "tiny"
                real_evaluate = self.module.evaluate
                calls = []

                def checked_evaluate(model, bank, kind, device, batch_size):
                    if bank.path == paths["test"].resolve():
                        self.assertTrue(Path(str(output) + ".pt").is_file())
                    calls.append(bank.path.name)
                    return real_evaluate(model, bank, kind, device, batch_size)

                args = ["--env", env, "--model", "joint", "--train-bank", str(paths["train"]),
                        "--val-bank", str(paths["val"]), "--test-bank", str(paths["test"]),
                        "--out", str(output), "--steps", "2", "--bs", "2", "--eval-bs", "2",
                        "--evalevery", "1", "--width", "8", "--T", "1", "--attention-heads", "0",
                        "--device", "cpu", "--torch-threads", "1"]
                with patch.object(self.module, "evaluate", checked_evaluate), redirect_stdout(io.StringIO()):
                    result = self.module.main(args)
                self.assertEqual(calls.count("test.npz"), 1)
                self.assertEqual(calls[-1], "test.npz")
                self.assertEqual(result["training"]["lr"], 1e-4 if env == "lights" else 3e-4)
                model, checkpoint = self.module.load_extra_checkpoint(str(output) + ".pt", device="cpu")
                bank = self.module.FrozenExtraBank(paths["test"], env)
                _, predictions = real_evaluate(model, bank, "joint", torch.device("cpu"), 2)
                with np.load(str(output) + ".predictions.npz") as saved:
                    np.testing.assert_array_equal(predictions["distance"], saved["test_distance"])
                self.assertEqual(checkpoint["sample_trace_sha256"], result["sample_trace_sha256"])
                self.assertEqual(checkpoint["env"], env)
                self.assertTrue(Path(str(output) + ".json").is_file())


if __name__ == "__main__":
    unittest.main()


