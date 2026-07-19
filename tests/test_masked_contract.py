import datetime
import os
import sys
import unittest

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from losses import masked_mse  # noqa: E402
from radar_contract import CanonicalRadarFrame, DEFAULT_CANONICAL_GRID  # noqa: E402


class MaskedContractTest(unittest.TestCase):
    def test_canonical_frame_keeps_no_data_separate_from_no_echo(self):
        frame = CanonicalRadarFrame(
            reflectivity_dbz=np.array([[0.0, 30.0]], dtype=np.float32),
            valid_mask=np.array([[True, False]]),
            timestamp_utc=datetime.datetime(2026, 7, 19),
            station_id="demo",
            source_id="fixture",
        )

        values, mask = frame.to_model_arrays()
        self.assertEqual(DEFAULT_CANONICAL_GRID.shape, (512, 512))
        self.assertEqual(values.tolist(), [[0.0, 0.0]])
        self.assertEqual(mask.tolist(), [[1.0, 0.0]])
        self.assertTrue(np.isnan(frame.reflectivity_dbz[0, 1]))

    def test_masked_mse_ignores_invalid_pixel(self):
        prediction = torch.tensor([1.0, 100.0]).reshape(1, 1, 1, 2)
        target = torch.zeros_like(prediction)
        valid_mask = torch.tensor([1.0, 0.0]).reshape(1, 1, 1, 2)

        self.assertEqual(masked_mse(prediction, target, valid_mask).item(), 1.0)


if __name__ == "__main__":
    unittest.main()
