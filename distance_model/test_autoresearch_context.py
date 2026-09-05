"""CPU invariants for the H3 static-context distance probe.

Run with: python3 -m unittest distance_model.test_autoresearch_context -v
"""

import unittest

import torch
import torch.nn.functional as F

try:
    from distance_model.autoresearch_context import ContextInteg
except ModuleNotFoundError:
    ContextInteg = None


def image_pair(batch=2, grid=7, ngate=3):
    """A same-layout pair in switchyard's 12-channel pureimage format."""
    a = torch.zeros(batch, 12, grid, grid)
    gates = [(1, 3), (3, 1), (3, 5)][:ngate]
    a[:, 0, 3, :] = 1
    a[:, 0, :, 3] = 1
    for row, col in gates:
        a[:, 0, row, col] = 0
        a[:, 3, row, col] = 1
    a[:, 5, 0, 1] = 1
    a[:, 6, 5, 0] = 1
    a[:, 7, 5, 5] = 1
    a[:, 8, 0, 5] = 1
    for gate, (row, col) in enumerate(gates):
        a[:, 5 + gate % 2, row, col] = 1
        a[:, 7, row, col] = float(gate == 0)
        a[:, 4, row, col] = float(gate == 1)
    a[:, 1, 0, 0] = 1
    a[:, 2, 1, 1] = 1
    b = a.clone()
    b[:, 1:3] = 0
    b[:, 1, 2, 2] = 1
    b[:, 2, 5, 4] = 1
    if gates:
        b[:, 4, gates[0][0], gates[0][1]] = 1
    return a, b


class ContextIntegChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.assertIsNotNone(ContextInteg, "The H3 ContextInteg model must exist")
        torch.manual_seed(7)
        self.model = ContextInteg(
            in_channels=12, d=32, heads=4, layers=2, T=3,
            cnnw=16, cnndepth=3, ngate=3,
        )
        self.a, self.b = image_pair()

    def test_readout_is_exactly_integrated_actual_motion(self):
        prediction, states = self.model(self.a, self.b, ret_states=True)
        self.assertEqual(prediction.shape, (2,))
        self.assertEqual(len(states), self.model.T + 1)
        self.assertEqual(states[0].shape, (2, 10, 32))
        motion = sum(
            (new - old).float().norm(dim=-1).sum(-1)
            for old, new in zip(states, states[1:])
        )
        torch.testing.assert_close(prediction, F.softplus(self.model.scale) * motion)
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue((prediction > 0).all())

    def test_no_iterations_means_no_distance_or_offset(self):
        prediction, states = self.model(self.a, self.b, Trun=0, ret_states=True)
        torch.testing.assert_close(prediction, torch.zeros_like(prediction), rtol=0, atol=0)
        self.assertEqual(len(states), 1)

    def test_stationary_recurrence_has_exactly_zero_readout(self):
        with torch.no_grad():
            for block in self.model.blocks:
                for parameter in block.parameters():
                    parameter.zero_()
        prediction = self.model(self.a, self.b)
        torch.testing.assert_close(prediction, torch.zeros_like(prediction), rtol=0, atol=0)

    def test_context_excludes_dynamic_channels_and_is_encoded_once(self):
        calls = []
        hook = self.model.context_encoder.register_forward_hook(
            lambda module, args, result: calls.append(result.detach().clone())
        )
        self.model(self.a, self.b)
        self.assertEqual(len(calls), 1)
        original = calls.pop()
        self.model(self.b, self.a)
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(original, calls.pop(), rtol=0, atol=0)
        changed = self.a.clone()
        changed[:, 5, 1, 3] = 1 - changed[:, 5, 1, 3]
        self.model(changed, self.b)
        self.assertFalse(torch.equal(original, calls.pop()))
        hook.remove()

    def test_gradients_reach_context_recurrence_and_scale(self):
        prediction = self.model(self.a, self.b)
        loss = (prediction - torch.tensor([4.0, 8.0])).square().mean()
        loss.backward()
        for group in (self.model.context_encoder, self.model.blocks):
            gradients = [p.grad for p in group.parameters() if p.requires_grad]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
        self.assertIsNotNone(self.model.scale.grad)
        self.assertGreater(float(self.model.scale.grad.abs()), 0)

    def test_batch_independence_and_dynamic_gate_sensitivity(self):
        self.model.eval()
        a, b = self.a.clone(), self.b.clone()
        a[1, 0, 6, 6] = 1
        b[1, 0, 6, 6] = 1
        batched = self.model(a, b)
        individual = torch.cat([self.model(a[i:i + 1], b[i:i + 1]) for i in range(2)])
        torch.testing.assert_close(batched, individual, atol=1e-5, rtol=1e-5)
        changed = b.clone()
        changed[:, 4, 1, 3] = 0
        self.assertFalse(torch.equal(self.model(a, b), self.model(a, changed)))

    def test_no_gate_maps_and_larger_grid(self):
        model = ContextInteg(in_channels=12, d=32, heads=4, layers=1, T=2,
                             cnnw=16, cnndepth=2, ngate=0)
        a, b = image_pair(batch=1, grid=11, ngate=0)
        prediction, states = model(a, b, ret_states=True)
        self.assertEqual(states[0].shape, (1, 4, 32))
        self.assertTrue(torch.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
