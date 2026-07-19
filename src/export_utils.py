import datetime
from typing import Optional

import numpy as np
import xarray as xr

from config import FORECAST_STEP_MINUTES, PRODUCT_NAME


def _quality_cube(
    value: Optional[np.ndarray],
    shape: tuple[int, int, int],
    name: str,
    dtype,
) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 2:
        array = np.repeat(array[np.newaxis, ...], shape[0], axis=0)
    elif array.ndim == 3 and array.shape[0] == 1 and shape[0] > 1:
        array = np.repeat(array, shape[0], axis=0)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape [H,W] or {shape}, got {array.shape}")
    return array


def _mask_variable(values: np.ndarray, long_name: str) -> tuple:
    return (
        ("valid_time_utc", "y", "x"),
        values.astype(np.uint8),
        {
            "long_name": long_name,
            "flag_values": np.asarray([0, 1], dtype=np.uint8),
            "flag_meanings": "false true",
            "grid_mapping": "crs",
        },
    )


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
    valid_mask: Optional[np.ndarray] = None,
    coverage_mask: Optional[np.ndarray] = None,
    clutter_mask: Optional[np.ndarray] = None,
    interpolation_weight: Optional[np.ndarray] = None,
):
    """Export an experimental reflectivity forecast and quality masks to NetCDF4."""

    values = np.asarray(forecast_data, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"forecast_data must have shape [T,H,W], got {values.shape}")
    if grid_resolution <= 0 or interval_minutes <= 0:
        raise ValueError("grid_resolution and interval_minutes must be positive")

    time_count, height, width = values.shape
    shape = (time_count, height, width)
    valid = _quality_cube(valid_mask, shape, "valid_mask", bool)
    coverage = _quality_cube(coverage_mask, shape, "coverage_mask", bool)
    clutter = _quality_cube(clutter_mask, shape, "clutter_mask", bool)
    weights = _quality_cube(
        interpolation_weight,
        shape,
        "interpolation_weight",
        np.float32,
    )
    if valid is not None:
        values = np.where(valid, values, np.nan).astype(np.float32)

    if base_time.tzinfo is not None:
        base_time = base_time.astimezone(datetime.UTC).replace(tzinfo=None)
    valid_times = np.array(
        [base_time + datetime.timedelta(minutes=interval_minutes * (index + 1)) for index in range(time_count)],
        dtype="datetime64[ns]",
    )
    lead_times = [interval_minutes * (index + 1) for index in range(time_count)]

    x_coords = (np.arange(width, dtype=np.float64) - (width - 1) / 2.0) * grid_resolution
    y_coords = (np.arange(height, dtype=np.float64) - (height - 1) / 2.0) * grid_resolution

    data_vars = {
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
    }
    if valid is not None:
        data_vars["valid_mask"] = _mask_variable(valid, "Effective forecast validity mask")
    if coverage is not None:
        data_vars["coverage_mask"] = _mask_variable(coverage, "Radar geometric coverage mask")
    if clutter is not None:
        data_vars["clutter_mask"] = _mask_variable(clutter, "Pixels excluded as clutter")
    if weights is not None:
        data_vars["interpolation_weight"] = (
            ("valid_time_utc", "y", "x"),
            np.clip(np.nan_to_num(weights, nan=0.0), 0.0, 1.0).astype(np.float32),
            {
                "long_name": "Relative interpolation confidence",
                "units": "1",
                "valid_min": 0.0,
                "valid_max": 1.0,
                "grid_mapping": "crs",
            },
        )

    dataset = xr.Dataset(
        data_vars=data_vars,
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
            "quality_mask_policy": "last_observed_geometry_repeated_over_forecast_leads",
            "not_official_warning": "true",
            "reflectivity_only_no_nwp": "true",
        },
    )
    dataset.to_netcdf(output_path)
    return output_path
