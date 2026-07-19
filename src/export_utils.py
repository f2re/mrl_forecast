import datetime
from typing import Optional

import numpy as np
import xarray as xr

from config import FORECAST_STEP_MINUTES, PRODUCT_NAME


def save_forecast_to_netcdf(
    forecast_data: np.ndarray,
    base_time: datetime.datetime,
    station_id: str,
    output_path: str,
    grid_resolution: float = 1953.125,
    interval_minutes: int = FORECAST_STEP_MINUTES,
    station_lon: Optional[float] = None,
    station_lat: Optional[float] = None,
    pipeline_version: str = "unknown",
    model_id: str = "unknown",
    source: str = "unknown",
    model_architecture: str = "unknown",
    quality_gate_status: str = "unknown",
):
    """Export an experimental reflectivity forecast [T,H,W] to NetCDF4."""

    values = np.asarray(forecast_data, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"forecast_data must have shape [T,H,W], got {values.shape}")
    if grid_resolution <= 0 or interval_minutes <= 0:
        raise ValueError("grid_resolution and interval_minutes must be positive")

    time_count, height, width = values.shape
    if base_time.tzinfo is not None:
        base_time = base_time.astimezone(datetime.UTC).replace(tzinfo=None)
    valid_times = np.array(
        [base_time + datetime.timedelta(minutes=interval_minutes * (index + 1)) for index in range(time_count)],
        dtype="datetime64[ns]",
    )
    lead_times = [interval_minutes * (index + 1) for index in range(time_count)]

    x_coords = (np.arange(width, dtype=np.float64) - (width - 1) / 2.0) * grid_resolution
    y_coords = (np.arange(height, dtype=np.float64) - (height - 1) / 2.0) * grid_resolution

    dataset = xr.Dataset(
        data_vars={
            "reflectivity": (
                ("valid_time_utc", "y", "x"),
                values,
                {
                    "units": "dBZ",
                    "long_name": "Experimental radar reflectivity nowcast",
                    "grid_mapping": "crs",
                },
            ),
            "crs": (
                (),
                0,
                {
                    "grid_mapping_name": "azimuthal_equidistant",
                    "longitude_of_projection_origin": station_lon if station_lon is not None else float("nan"),
                    "latitude_of_projection_origin": station_lat if station_lat is not None else float("nan"),
                    "false_easting": 0.0,
                    "false_northing": 0.0,
                    "units": "m",
                },
            ),
        },
        coords={
            "valid_time_utc": ("valid_time_utc", valid_times, {"timezone": "UTC"}),
            "lead_time_minutes": ("valid_time_utc", lead_times, {"units": "minutes"}),
            "y": ("y", y_coords, {"units": "m", "axis": "Y"}),
            "x": ("x", x_coords, {"units": "m", "axis": "X"}),
        },
        attrs={
            "product": PRODUCT_NAME,
            "description": "Experimental radar reflectivity nowcast; not an official warning",
            "units": "dBZ",
            "station": station_id,
            "base_time_utc": base_time.isoformat(),
            "institution": "MRL Forecast Pro",
            "forecast_step_minutes": interval_minutes,
            "grid_resolution_m": float(grid_resolution),
            "pipeline_version": pipeline_version,
            "model_id": model_id,
            "model_architecture": model_architecture,
            "source": source,
            "quality_gate_status": quality_gate_status,
            "not_official_warning": "true",
            "reflectivity_only_no_nwp": "true",
        },
    )
    dataset.to_netcdf(output_path)
    return output_path
