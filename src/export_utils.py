import os
import numpy as np
import xarray as xr
import datetime
from typing import Optional

def save_forecast_to_netcdf(
    forecast_data: np.ndarray, 
    base_time: datetime.datetime, 
    station_id: str,
    output_path: str,
    grid_resolution: float = 1953.125, # 500000 / 256
    interval_minutes: int = 10,
    station_lon: Optional[float] = None,
    station_lat: Optional[float] = None,
    pipeline_version: str = "unknown",
    model_id: str = "unknown",
    source: str = "unknown",
):
    """
    Экспортирует прогноз (тензор [T, H, W]) в формат NetCDF4 с временными и пространственными координатами.
    """
    T, H, W = forecast_data.shape
    
    # Создаем координаты
    if base_time.tzinfo is not None:
        base_time = base_time.astimezone(datetime.UTC).replace(tzinfo=None)
    times = np.array(
        [base_time + datetime.timedelta(minutes=interval_minutes * (i + 1)) for i in range(T)],
        dtype="datetime64[ns]",
    )
    lead_times = [interval_minutes * (i + 1) for i in range(T)]
    
    # Центрируем сетку (предполагаем, что радар в центре 0,0)
    # Диапазон -250км до +250км
    x_coords = np.linspace(-250000, 250000, W)
    y_coords = np.linspace(-250000, 250000, H)
    
    ds = xr.Dataset(
        data_vars={
            "reflectivity": (
                ("time", "y", "x"),
                forecast_data,
                {
                    "units": "dBZ",
                    "long_name": "Radar Reflectivity",
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
            "time": ("time", times, {"timezone": "UTC"}),
            "lead_time": ("time", lead_times, {"units": "minutes"}),
            "y": y_coords,
            "x": x_coords
        },
        attrs={
            "description": "AI Precipitation Nowcast",
            "station": station_id,
            "base_time": base_time.isoformat(),
            "institution": "MRL Forecast Pro",
            "pipeline_version": pipeline_version,
            "model_id": model_id,
            "source": source,
        }
    )
    
    ds.to_netcdf(output_path)
    return output_path
