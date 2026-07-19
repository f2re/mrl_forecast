import os
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
