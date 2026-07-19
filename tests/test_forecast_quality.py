import os
import sys
import unittest

import numpy as np
from scipy.ndimage import shift

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from forecast_quality import (  # noqa: E402
    advection_forecast,
    block_motion_forecast,
    is_uniform_forecast,
    persistence_forecast,
    threshold_metrics_by_lead_time,
)


class ForecastQualityTest(unittest.TestCase):
    def test_persistence_repeats_last_history_frame(self):
        history = np.arange(3 * 2 * 2).reshape(3, 2, 2)

        forecast = persistence_forecast(history, output_steps=4)

        self.assertEqual(forecast.shape, (4, 2, 2))
        np.testing.assert_array_equal(forecast[0], history[-1])
        np.testing.assert_array_equal(forecast[-1], history[-1])

    def test_uniform_green_layer_is_rejected(self):
        forecast = np.full((4, 32, 32), 25.0, dtype=np.float32)

        self.assertTrue(is_uniform_forecast(forecast))

    def test_structured_forecast_is_not_uniform(self):
        forecast = np.zeros((4, 32, 32), dtype=np.float32)
        forecast[:, 10:20, 10:20] = 25.0

        self.assertFalse(is_uniform_forecast(forecast))

    def test_threshold_metrics_report_extended_categorical_scores(self):
        target = np.array([[[0.0, 10.0], [10.0, 0.0]]])
        forecast = np.array([[[0.0, 10.0], [0.0, 10.0]]])

        metrics = threshold_metrics_by_lead_time(forecast, target, thresholds=(5.0,))

        lead = metrics["5.0"][0]
        self.assertEqual(lead["hits"], 1)
        self.assertEqual(lead["misses"], 1)
        self.assertEqual(lead["false_alarms"], 1)
        self.assertAlmostEqual(lead["csi"], 1 / 3)
        self.assertAlmostEqual(lead["pod"], 1 / 2)
        self.assertAlmostEqual(lead["far"], 1 / 2)
        self.assertAlmostEqual(lead["frequency_bias"], 1.0)
        self.assertIsNotNone(lead["ets"])

    def test_advection_moves_last_frame_using_recent_motion(self):
        history = np.zeros((2, 9, 9), dtype=np.float32)
        history[0, 4, 3] = 10.0
        history[1, 4, 4] = 10.0

        forecast = advection_forecast(history, output_steps=2, search_radius=2)

        self.assertEqual(forecast[0, 4, 5], 10.0)
        self.assertEqual(forecast[1, 4, 6], 10.0)

    def test_block_motion_improves_shifted_gradient_over_persistence(self):
        yy, xx = np.mgrid[0:32, 0:32]
        previous = (xx + 2 * yy).astype(np.float32)
        current = shift(previous, (0, 1), order=1, mode="constant", cval=0.0)
        expected = shift(current, (0, 1), order=1, mode="constant", cval=0.0)
        history = np.stack([previous, current])

        forecast = block_motion_forecast(
            history,
            output_steps=1,
            downsample=1,
            block_size=8,
            search_radius=2,
        )[0]

        self.assertLess(np.mean((forecast - expected) ** 2), np.mean((current - expected) ** 2))


if __name__ == "__main__":
    unittest.main()
