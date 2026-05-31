import os
import numpy as np
import xarray as xr
import datetime
from typing import Optional

from config import FORECAST_STEP_MINUTES, PRODUCT_NAME


def save_forecast_to_netcdf(
    forecast_data: np.ndarray,
    base_time: datetime.datetime,
    station_id: str,
    output_path: str,
    grid_resolution: float = 1953.125,  # 500000 / 256
    interval_minutes: int = FORECAST_STEP_MINUTES,
    station_lon: Optional[float] = None,
    station_lat: Optional[float] = None,
    pipeline_version: str = "unknown",
    model_id: str = "unknown",
    source: str = "unknown",
    model_architecture: str = "unknown",
    quality_gate_status: str = "unknown",
):
    """
    Экспортирует экспериментальный прогноз отражаемости МРЛ [T, H, W] в NetCDF4.

    Выходная величина — reflectivity в dBZ. Это не официальный прогноз осадков и
    не предупреждение об опасных явлениях.
    """
    T, H, W = forecast_data.shape

    if base_time.tzinfo is not None:
        base_time = base_time.astimezone(datetime.UTC).replace(tzinfo=None)
    times = np.array(
        [base_time + datetime.timedelta(minutes=interval_minutes * (i + 1)) for i in range(T)],
        dtype="datetime64[ns]",
    )
    lead_times = [interval_minutes * (i + 1) for i in range(T)]

    x_coords = np.linspace(-250000, 250000, W)
    y_coords = np.linspace(-250000, 250000, H)

    ds = xr.Dataset(
        data_vars={
            "reflectivity": (
                ("valid_time_utc", "y", "x"),
                forecast_data,
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
            "valid_time_utc": ("valid_time_utc", times, {"timezone": "UTC"}),
            "lead_time_minutes": ("valid_time_utc", lead_times, {"units": "minutes"}),
            "y": y_coords,
            "x": x_coords,
        },
        attrs={
            "product": PRODUCT_NAME,
            "description": "Experimental radar reflectivity nowcast; not an official warning",
            "units": "dBZ",
            "station": station_id,
            "base_time_utc": base_time.isoformat(),
            "institution": "MRL Forecast Pro",
            "forecast_step_minutes": interval_minutes,
            "pipeline_version": pipeline_version,
            "model_id": model_id,
            "model_architecture": model_architecture,
            "source": source,
            "quality_gate_status": quality_gate_status,
            "not_official_warning": "true",
            "reflectivity_only_no_nwp": "true",
        },
    )

    ds.to_netcdf(output_path)
    return output_path
