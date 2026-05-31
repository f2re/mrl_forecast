import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from map_visualization import prepare_radar_overlay  # noqa: E402


class MapVisualizationTest(unittest.TestCase):
    def test_overlay_uses_lower_origin_without_flipping_north_south(self):
        data = np.array([[1.0, 2.0], [30.0, 40.0]])

        overlay, _, origin = prepare_radar_overlay(data, "kokx", 250.0)

        self.assertEqual(origin, "lower")
        self.assertEqual(overlay[1, 0], 30.0)

    def test_unknown_station_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown radar station"):
            prepare_radar_overlay(np.ones((2, 2)), "missing", 250.0)


if __name__ == "__main__":
    unittest.main()

