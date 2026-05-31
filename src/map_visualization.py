"""
Enhanced radar map visualization utilities.

This module renders experimental radar reflectivity nowcasts with orientation
aids, range rings, azimuth labels, a dBZ legend and UTC timestamp annotations.
"""

import io
import math
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, patheffects

from config import FORECAST_STEP_MINUTES

try:
    import contextily as ctx
except Exception:
    ctx = None

try:
    from pyproj import CRS, Transformer
except Exception:
    CRS = None
    Transformer = None

RADAR_COORDS = {
    'kokx': (-72.86, 40.86),
    'kdtx': (-83.47, 42.69),
    'klot': (-88.08, 41.60),
    'kbgm': (-75.98, 42.20),
    'kewx': (-98.03, 29.70),
    'tjua': (-66.08, 18.12),
    'kffc': (-84.57, 33.36),
    'kamx': (-80.41, 25.61),
    'kbox': (-71.14, 41.96),
    'kgyx': (-70.26, 43.89),
    'kilx': (-89.34, 40.15),
    'klwx': (-77.49, 38.98),
}


def _local_to_web_mercator_transformer(station_code: str):
    """Create a local AEQD -> Web Mercator transformer for a radar station."""
    station = station_code.lower()
    if station not in RADAR_COORDS:
        raise ValueError(f"unknown radar station: {station_code}")
    if CRS is None or Transformer is None:
        raise ImportError("pyproj is required for coordinate transformations")
    lon, lat = RADAR_COORDS[station]
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs(local_crs, "epsg:3857", always_xy=True)


def _transform_local_points(station_code: str, x, y):
    transformer = _local_to_web_mercator_transformer(station_code)
    return transformer.transform(x, y)


def prepare_radar_overlay(
    data: np.ndarray,
    station_code: str,
    max_range_km: float,
) -> Tuple[np.ma.MaskedArray, List[float], str]:
    """Return a north-up radar overlay and Web Mercator extent."""
    offset_m = max_range_km * 1000.0
    corner_x, corner_y = _transform_local_points(
        station_code,
        [-offset_m, offset_m],
        [-offset_m, offset_m],
    )
    extent = [corner_x[0], corner_x[1], corner_y[0], corner_y[1]]
    overlay = np.ma.masked_where(~np.isfinite(data) | (data < 1.0), data)
    return overlay, extent, "lower"


def get_radar_colormap() -> Tuple[colors.Colormap, colors.BoundaryNorm]:
    """Return a common dBZ colormap and boundary normalisation."""
    clevs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
    ccols = [
        '#ffffff00',
        '#00ecec', '#01a0f6', '#0000f6', '#00ff00', '#00c800',
        '#009000', '#ffff00', '#e7c000', '#ff9000', '#ff0000',
        '#d60000', '#c00000', '#ff00ff', '#9955c9'
    ]
    cmap = colors.ListedColormap(ccols)
    norm = colors.BoundaryNorm(clevs, cmap.N)
    return cmap, norm


def _format_timestamp(dt: Optional[datetime]) -> Optional[str]:
    """Format a datetime into a human-readable Russian UTC timestamp."""
    if dt is None:
        return None
    months = [
        'января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
        'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря'
    ]
    return f"{dt.day} {months[dt.month - 1]} {dt.year} г., {dt:%H:%M} UTC"


def create_radar_plot(
    data: np.ndarray,
    title: str,
    station_code: str = 'kokx',
    max_range_km: float = 250.0,
    timestamp: Optional[datetime] = None,
    ring_intervals_km: Optional[List[float]] = None,
    azimuth_angles_deg: Optional[List[int]] = None,
) -> bytes:
    """Render a single radar reflectivity image with orientation aids."""
    cmap, norm = get_radar_colormap()
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    offset_m = max_range_km * 1000.0
    data, extent, origin = prepare_radar_overlay(data, station_code, max_range_km)
    center_x, center_y = _transform_local_points(station_code, 0.0, 0.0)

    im = ax.imshow(
        data,
        cmap=cmap,
        norm=norm,
        extent=extent,
        origin=origin,
        alpha=0.8,
        zorder=2,
    )

    if ctx is not None:
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zorder=1)
        except Exception:
            pass

    if ring_intervals_km is None:
        ring_intervals_km = [max_range_km * i / 5.0 for i in range(1, 6)]
    if azimuth_angles_deg is None:
        azimuth_angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]

    ring_color = '#666666'
    ring_alpha = 0.5
    line_style = '--'

    for dist_km in ring_intervals_km:
        radius_m = dist_km * 1000.0
        ring_angles = np.linspace(0.0, 2.0 * np.pi, 181)
        ring_x, ring_y = _transform_local_points(
            station_code,
            radius_m * np.sin(ring_angles),
            radius_m * np.cos(ring_angles),
        )
        ax.plot(
            ring_x,
            ring_y,
            color=ring_color,
            linewidth=0.8,
            linestyle=line_style,
            alpha=ring_alpha,
            zorder=2.5,
        )
        label_x, label_y = _transform_local_points(station_code, radius_m + (0.02 * offset_m), 0.0)
        ax.text(label_x, label_y, f"{int(dist_km)} км", color=ring_color, fontsize=8, va='center', ha='left', alpha=ring_alpha)

    for angle in azimuth_angles_deg:
        angle_rad = math.radians(angle)
        x_end, y_end = _transform_local_points(
            station_code,
            offset_m * math.sin(angle_rad),
            offset_m * math.cos(angle_rad),
        )
        ax.plot(
            [center_x, x_end],
            [center_y, y_end],
            color=ring_color,
            linewidth=0.8,
            linestyle=line_style,
            alpha=ring_alpha,
            zorder=2.5,
        )
        ax.text(
            x_end,
            y_end,
            f"{angle}°",
            color=ring_color,
            fontsize=7,
            va='center',
            ha='center',
            alpha=ring_alpha,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.4, pad=1),
        )

    ax.plot(center_x, center_y, 'k+', markersize=12, zorder=4, markeredgewidth=2)
    ax.text(
        center_x + 0.02 * offset_m,
        center_y + 0.02 * offset_m,
        station_code.upper(),
        weight='bold',
        fontsize=12,
        zorder=4,
        path_effects=[patheffects.withStroke(linewidth=3, foreground='white')],
    )

    ax.set_title(title, fontsize=14, pad=15)
    ax.axis('off')

    cax = fig.add_axes([0.9, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('дБЗ', rotation=90, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ts_str = _format_timestamp(timestamp)
    if ts_str:
        ax.text(
            0.01,
            0.01,
            ts_str,
            transform=ax.transAxes,
            fontsize=9,
            color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=2),
        )

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_sequence_plots(
    input_seq: np.ndarray,
    pred_seq: np.ndarray,
    input_len: int,
    station_code: str = 'kokx',
    start_datetime: Optional[datetime] = None,
    max_range_km: float = 250.0,
    history_timestamps: Optional[List[datetime]] = None,
    interval_minutes: int = FORECAST_STEP_MINUTES,
) -> List[bytes]:
    """Generate radar plots for history and forecast frames."""
    images: List[bytes] = []
    for i in range(input_seq.shape[0]):
        lead_time = (input_len - i - 1) * -interval_minutes
        label = f"История (T{lead_time} мин)" if lead_time != 0 else "Сейчас (T-0)"
        ts = None
        if history_timestamps and i < len(history_timestamps):
            ts = history_timestamps[i]
        elif start_datetime is not None:
            ts = start_datetime + timedelta(minutes=lead_time)
        images.append(
            create_radar_plot(
                input_seq[i] * 70.0,
                label,
                station_code=station_code,
                max_range_km=max_range_km,
                timestamp=ts,
            )
        )

    base_forecast_ts = start_datetime
    if history_timestamps and not base_forecast_ts:
        base_forecast_ts = history_timestamps[-1]

    for i in range(pred_seq.shape[0]):
        lead_time = (i + 1) * interval_minutes
        label = f"Прогноз ИИ (T+{lead_time} мин)"
        ts = None
        if base_forecast_ts is not None:
            ts = base_forecast_ts + timedelta(minutes=lead_time)
        images.append(
            create_radar_plot(
                pred_seq[i] * 70.0,
                label,
                station_code=station_code,
                max_range_km=max_range_km,
                timestamp=ts,
            )
        )
    return images
