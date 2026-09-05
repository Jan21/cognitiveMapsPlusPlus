"""Small CPU checks for the joint spatial motion integrator.

Run: python -m unittest distance_model.test_autoresearch_joint -v
"""

import importlib.util
import unittest

import torch
import torch.nn.functional as F


class JointPixelIntegTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(3)
        self.assertIsNotNone(
            importlib.util.find_spec("distance_model.autoresearch_joint"),
            "The joint spatial motion model has not been implemented yet.",
        )
        from distance_model.autoresearch_joint import JointPixelInteg

        self.model_class = JointPixelInteg
        self.a = torch.rand(2, 3, 7, 9)
        self.b = torch.rand_like(self.a)

    def test_distance_is_exactly_accumulated_actual_state_motion(self):
        for tied, stride, heads in ((True, 1, 0), (False, 2, 2)):
            with self.subTest(tied=tied, stride=stride, heads=heads):
                model = self.model_class(
                    3, width=8, T=3, blocks=2, tied=tied,
                    stride=stride, attention_heads=heads,
                )
                prediction, states = model(self.a, self.b, ret_states=True)
                self.assertEqual(prediction.shape, (2,))
                self.assertEqual(len(states), 7)
                self.assertEqual(states[0].shape[-2:], (7, 9) if stride == 1 else (4, 5))
                motion = sum(
                    (after - before).norm(dim=1).sum(dim=(1, 2))
                    for before, after in zip(states, states[1:])
                )
                torch.testing.assert_close(prediction, F.softplus(model.scale) * motion)
                self.assertTrue(torch.all(prediction > 0))

    def test_zero_recurrent_updates_give_zero_distance_despite_nonzero_encoder(self):
        model = self.model_class(3, width=8, T=2, attention_heads=2)
        with torch.no_grad():
            for parameter in model.updates.parameters():
                parameter.zero_()
        prediction, states = model(self.a, self.b, ret_states=True)
        self.assertGreater(states[0].abs().sum().item(), 0)
        torch.testing.assert_close(prediction, torch.zeros(2), atol=0, rtol=0)
        prediction.sum().backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_gradients_reach_encoder_every_untied_update_inputs_and_scale(self):
        model = self.model_class(3, width=8, T=2, blocks=2, tied=False, attention_heads=2)
        a, b = self.a.requires_grad_(), self.b.requires_grad_()
        model(a, b).sum().backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        for value in (a, b):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_runtime_iterations_preserve_prefix_and_motion_is_nondecreasing(self):
        model = self.model_class(3, width=8, T=2, blocks=2)
        zero, states = model(self.a, self.b, Trun=0, ret_states=True)
        torch.testing.assert_close(zero, torch.zeros_like(zero))
        self.assertEqual(len(states), 1)
        short, short_states = model(self.a, self.b, Trun=1, ret_states=True)
        long, long_states = model(self.a, self.b, Trun=4, ret_states=True)
        self.assertTrue(torch.all(long >= short))
        for left, right in zip(short_states, long_states):
            torch.testing.assert_close(left, right)
        untied = self.model_class(3, width=8, T=2, tied=False, reinject=False)
        self.assertTrue(torch.isfinite(untied(self.a, self.b)).all())
        with self.assertRaises(ValueError):
            untied(self.a, self.b, Trun=3)

    def test_invalid_configuration_and_tensor_shapes_fail_clearly(self):
        for kwargs in ({"T": 0}, {"kernel_size": 2}, {"width": 0},
                       {"stride": 0}, {"blocks": 0}, {"attention_heads": 3},
                       {"step_scale": 0}, {"scale_init": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model_class(3, width=8, **kwargs) if "width" not in kwargs else self.model_class(3, **kwargs)
        model = self.model_class(3, width=8)
        for a, b in ((self.a, self.b[:, :, :-1]), (self.a[:, :2], self.b[:, :2]),
                     (self.a[0], self.b[0])):
            with self.assertRaises(ValueError):
                model(a, b)


if __name__ == "__main__":
    unittest.main()
