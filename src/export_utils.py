import os
import numpy as np
import xarray as xr
import datetime
from typing import List

def save_forecast_to_netcdf(
    forecast_data: np.ndarray, 
    base_time: datetime.datetime, 
    station_id: str,
    output_path: str,
    grid_resolution: float = 1953.125, # 500000 / 256
    interval_minutes: int = 15
):
    """
    Экспортирует прогноз (тензор [T, H, W]) в формат NetCDF4 с временными и пространственными координатами.
    """
    T, H, W = forecast_data.shape
    
    # Создаем координаты
    times = [base_time + datetime.timedelta(minutes=interval_minutes * (i + 1)) for i in range(T)]
    
    # Центрируем сетку (предполагаем, что радар в центре 0,0)
    # Диапазон -250км до +250км
    x_coords = np.linspace(-250000, 250000, W)
    y_coords = np.linspace(-250000, 250000, H)
    
    ds = xr.Dataset(
        data_vars={
            "reflectivity": (("time", "y", "x"), forecast_data, {"units": "dBZ", "long_name": "Radar Reflectivity"})
        },
        coords={
            "time": times,
            "y": y_coords,
            "x": x_coords
        },
        attrs={
            "description": "AI Precipitation Nowcast",
            "station": station_id,
            "base_time": base_time.isoformat(),
            "institution": "MRL Forecast Pro"
        }
    )
    
    ds.to_netcdf(output_path)
    return output_path
