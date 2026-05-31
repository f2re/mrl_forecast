import os
import sys
import tempfile
import unittest
import datetime

import numpy as np
import xarray as xr

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from config import FORECAST_STEP_MINUTES  # noqa: E402
from export_utils import save_forecast_to_netcdf  # noqa: E402
from map_visualization import generate_sequence_plots  # noqa: E402
from radar_pipeline import RadarPipelineConfig  # noqa: E402


class TimeContractTest(unittest.TestCase):
    def test_pipeline_default_time_step_is_15_minutes(self):
        self.assertEqual(RadarPipelineConfig().time_step_minutes, 15)
        self.assertEqual(FORECAST_STEP_MINUTES, 15)

    def test_netcdf_default_lead_times_are_15_minute_steps(self):
        with tempfile.NamedTemporaryFile(suffix=".nc") as output:
            save_forecast_to_netcdf(
                forecast_data=np.zeros((4, 4, 4), dtype=np.float32),
                base_time=datetime.datetime(2026, 5, 31, 12, 0, tzinfo=datetime.UTC),
                station_id="KOKX",
                output_path=output.name,
            )
            with xr.open_dataset(output.name) as dataset:
                self.assertEqual(dataset["lead_time_minutes"].values.tolist(), [15, 30, 45, 60])

    def test_visualization_accepts_15_minute_interval_contract(self):
        images = generate_sequence_plots(
            np.zeros((1, 4, 4), dtype=np.float32),
            np.zeros((1, 4, 4), dtype=np.float32),
            input_len=1,
            station_code="kokx",
            start_datetime=datetime.datetime(2026, 5, 31, 12, 0),
            interval_minutes=FORECAST_STEP_MINUTES,
        )
        self.assertEqual(len(images), 2)
        self.assertGreater(len(images[0]), 0)


if __name__ == "__main__":
    unittest.main()
