import datetime
import os
import sys
import tempfile
import unittest

import numpy as np
import xarray as xr

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from export_utils import save_forecast_to_netcdf  # noqa: E402
from radar_pipeline import PIPELINE_VERSION  # noqa: E402


class ExportUtilsTest(unittest.TestCase):
    def test_netcdf_contains_crs_lead_time_and_provenance(self):
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
            )

            with xr.open_dataset(output.name) as dataset:
                self.assertIn("crs", dataset)
                self.assertIn("lead_time", dataset.coords)
                self.assertEqual(dataset.attrs["pipeline_version"], PIPELINE_VERSION)
                self.assertEqual(dataset.attrs["model_id"], "fixture-model")
                self.assertEqual(dataset["reflectivity"].attrs["grid_mapping"], "crs")


if __name__ == "__main__":
    unittest.main()

