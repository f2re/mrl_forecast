import os
import sys
import unittest

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from model_runtime import ModelRuntime  # noqa: E402
from radar_pipeline import CANONICAL_PIPELINE_VERSION  # noqa: E402


class ModelRuntimeTest(unittest.TestCase):
    def test_pipeline_uses_checkpoint_cadence(self):
        runtime = ModelRuntime(torch.device("cpu"))
        runtime.info = {
            "pipeline_version": CANONICAL_PIPELINE_VERSION,
            "forecast_step_minutes": 10,
        }

        pipeline = runtime.pipeline()

        self.assertEqual(pipeline.config.time_step_minutes, 10)
        self.assertEqual(pipeline.config.width, 512)
        self.assertEqual(pipeline.config.height, 512)

    def test_quality_masks_form_effective_model_mask(self):
        runtime = ModelRuntime(torch.device("cpu"))
        runtime.info = {
            "input_length": 2,
            "grid": {"width": 2, "height": 2},
        }
        values = np.full((2, 2, 2), 20.0, dtype=np.float32)
        valid = np.ones_like(values, dtype=bool)
        coverage = np.ones_like(valid)
        clutter = np.zeros_like(valid)
        clutter[:, 0, 1] = True
        weights = np.ones_like(values, dtype=np.float32)
        weights[:, 1, 0] = 0.0

        normalized, effective, quality = runtime._prepare_arrays(
            values,
            valid,
            coverage,
            clutter,
            weights,
        )

        self.assertFalse(effective[:, 0, 1].any())
        self.assertFalse(effective[:, 1, 0].any())
        self.assertTrue(effective[:, 0, 0].all())
        self.assertTrue((normalized[:, 0, 1] == 0.0).all())
        self.assertTrue(quality["clutter_mask"][:, 0, 1].all())


if __name__ == "__main__":
    unittest.main()
