import datetime
import os
import sys
import tempfile
import unittest

import numpy as np
import xarray as xr

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from config import FORECAST_STEP_MINUTES, PRODUCT_NAME  # noqa: E402
from export_utils import save_forecast_to_netcdf  # noqa: E402
from radar_pipeline import PIPELINE_VERSION  # noqa: E402


class ExportUtilsTest(unittest.TestCase):
    def test_netcdf_contains_crs_provenance_and_quality_masks(self):
        valid = np.ones((2, 4, 4), dtype=bool)
        valid[:, 0, 0] = False
        coverage = np.ones((4, 4), dtype=bool)
        clutter = np.zeros((4, 4), dtype=bool)
        clutter[1, 1] = True
        weights = np.ones((4, 4), dtype=np.float32)
        weights[0, 0] = 0.0

        with tempfile.NamedTemporaryFile(suffix=".nc") as output:
            save_forecast_to_netcdf(
                forecast_data=np.zeros((2, 4, 4), dtype=np.float32),
                base_time=datetime.datetime(2026, 5, 30, tzinfo=datetime.UTC),
                station_id="KOKX",
                output_path=output.name,
                station_lon=-72.86,
                station_lat=40.86,
                pipeline_version=PIPELINE_VERSION,
                model_id="fixture-model",
                valid_mask=valid,
                coverage_mask=coverage,
                clutter_mask=clutter,
                interpolation_weight=weights,
            )

            with xr.open_dataset(output.name) as dataset:
                self.assertIn("crs", dataset)
                self.assertIn("lead_time_minutes", dataset.coords)
                self.assertIn("valid_time_utc", dataset.coords)
                self.assertEqual(dataset.attrs["product"], PRODUCT_NAME)
                self.assertEqual(dataset.attrs["forecast_step_minutes"], FORECAST_STEP_MINUTES)
                self.assertEqual(dataset.attrs["pipeline_version"], PIPELINE_VERSION)
                self.assertEqual(dataset.attrs["model_id"], "fixture-model")
                self.assertEqual(dataset.attrs["not_official_warning"], "true")
                self.assertEqual(dataset["reflectivity"].attrs["grid_mapping"], "crs")
                self.assertEqual(dataset["lead_time_minutes"].values.tolist(), [15, 30])
                for name in (
                    "valid_mask",
                    "coverage_mask",
                    "clutter_mask",
                    "interpolation_weight",
                ):
                    self.assertIn(name, dataset)
                    self.assertEqual(dataset[name].shape, (2, 4, 4))
                self.assertTrue(np.isnan(dataset["reflectivity"].values[:, 0, 0]).all())


if __name__ == "__main__":
    unittest.main()
