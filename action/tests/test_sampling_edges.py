import unittest

import torch

from spider.config import Config, get_noise_scale
from spider.optimizers.sampling import _compute_weights_impl


class SamplingEdgesTest(unittest.TestCase):
    def test_one_elite_has_finite_unit_weight(self):
        for count in [1, 8, 16]:
            rewards = torch.arange(count, dtype=torch.float32)
            weights, _ = _compute_weights_impl(rewards, count, 0.1)
            self.assertTrue(torch.isfinite(weights).all())
            self.assertEqual(weights.sum().item(), 1)
            self.assertEqual(weights[-1].item(), 1)

    def test_multiple_elites_keep_original_standard_deviation(self):
        rewards = torch.arange(32, dtype=torch.float32)
        weights, _ = _compute_weights_impl(rewards, 32, 0.1)
        top = rewards[-3:]
        expected = torch.softmax((top - top.mean()) / (top.std() + 0.01) / 0.1, dim=0)
        torch.testing.assert_close(weights[-3:], expected)

    def test_zero_exploit_samples_do_not_scale_every_candidate(self):
        cfg = Config(
            device="cpu", nu=18, embodiment_type="right", num_samples=8, horizon=0.1, knot_dt=0.05
        )
        scale = get_noise_scale(cfg)
        self.assertEqual(scale[0].count_nonzero().item(), 0)
        self.assertAlmostEqual(
            scale[1, 0, 0].item(), cfg.first_ctrl_noise_scale * cfg.pos_noise_scale, places=8
        )


if __name__ == "__main__":
    unittest.main()
