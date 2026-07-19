import os
import sys
import unittest

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from phys_evolution import DifferentiableAdvection, MRLPhysEvolution  # noqa: E402


class PhysicsEvolutionTest(unittest.TestCase):
    def test_model_returns_explicit_motion_growth_and_decay(self):
        model = MRLPhysEvolution(
            base_channels=4,
            hidden_channels=6,
            output_steps=3,
        )
        reflectivity = torch.zeros(1, 4, 1, 32, 32)
        reflectivity[:, :, :, 14:18, 14:18] = 0.5
        valid_mask = torch.ones_like(reflectivity)

        forecast, diagnostics = model(reflectivity, valid_mask)

        self.assertEqual(forecast.shape, (1, 3, 1, 32, 32))
        self.assertEqual(diagnostics["motion"].shape, (1, 3, 2, 32, 32))
        self.assertEqual(diagnostics["growth"].shape, forecast.shape)
        self.assertEqual(diagnostics["decay"].shape, forecast.shape)
        self.assertTrue(torch.all(forecast >= 0.0))
        self.assertTrue(torch.all(forecast <= 1.0))

    def test_positive_x_motion_moves_echo_to_the_right(self):
        field = torch.zeros(1, 1, 9, 9)
        field[0, 0, 4, 3] = 1.0
        motion = torch.zeros(1, 2, 9, 9)
        motion[:, 0] = 1.0

        moved = DifferentiableAdvection()(field, motion)
        maximum = torch.nonzero(moved[0, 0] == moved[0, 0].max())[0]

        self.assertEqual(maximum.tolist(), [4, 4])


if __name__ == "__main__":
    unittest.main()
