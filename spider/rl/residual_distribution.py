"""Bounded exploration and normalized residual actions used by CoDesign PPO."""

import torch
from rsl_rl.modules.distribution import GaussianDistribution


class ResidualGaussian(GaussianDistribution):
    def update(self, mlp_output):
        super().update(mlp_output.clamp(-1, 1))

    def sample(self):
        return super().sample().clamp(-1, 1)

    def deterministic_output(self, mlp_output):
        return mlp_output.clamp(-1, 1)

    def as_deterministic_output_module(self):
        return torch.nn.Hardtanh(-1, 1)
