"""
Enhanced radar map visualization utilities.

This module extends the basic radar plotting functions used in the
``mrl_forecast`` project.  In addition to overlaying reflectivity data on a
CartoDB Positron base map, the visualisation now includes optional
orientation aids such as concentric range rings and azimuth lines, a
colour scale legend for dBZ values, and timestamp annotations.  These
elements help viewers quickly judge distances and directions on the
radar images while maintaining a clean and unobtrusive design.

The design decisions implemented here are informed by cartographic and
UI/UX guidelines.  For example, a neutral base map was chosen so the
data overlay stands out clearly, and the colour scale
reflects the standard placement on operational radar products.  Range
rings and radial lines follow conventions used in radar software where
concentric circles labelled by distance and radials annotated with
angles aid orientation.

Note: ``contextily`` and ``pyproj`` must be installed in the environment where this
code runs.  The functions gracefully catch errors from these libraries
so that the module can still be imported for testing.
"""

import io
import math
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for headless environments
import matplotlib.pyplot as plt
from matplotlib import colors, patheffects
from matplotlib.patches import Circle

try:
    import contextily as ctx
except Exception:
    ctx = None  # Fallback if contextily is not available

try:
    from pyproj import Transformer
except Exception:
    Transformer = None  # Fallback if pyproj is not available

# Географические координаты радаров (Долгота, Широта)
RADAR_COORDS = {
    'kokx': (-72.86, 40.86),  # Нью‑Йорк
    'kdtx': (-83.47, 42.69),  # Детройт
    'klot': (-88.08, 41.60),  # Чикаго
    'kamx': (-80.41, 25.61),  # Майами
    'default': (0.0, 0.0)     # Fallback
}

def get_radar_colormap() -> Tuple[colors.Colormap, colors.BoundaryNorm]:
    """Return a matplotlib colour map and normalisation for reflectivity.

    Colours correspond to typical dBZ thresholds used by many meteorological
    services.  The first colour is transparent so that areas with no data do
    not obscure the base map.
    """
    clevs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
    ccols = [
        '#ffffff00',  # transparent for missing data
        '#00ecec', '#01a0f6', '#0000f6', '#00ff00', '#00c800',
        '#009000', '#ffff00', '#e7c000', '#ff9000', '#ff0000',
        '#d60000', '#c00000', '#ff00ff', '#9955c9'
    ]
    cmap = colors.ListedColormap(ccols)
    norm = colors.BoundaryNorm(clevs, cmap.N)
    return cmap, norm

def _format_timestamp(dt: Optional[datetime]) -> Optional[str]:
    """Format a datetime into a human‑readable Russian string.

    If ``dt`` is ``None``, returns ``None``.  Otherwise the returned string
    will look like "30 мая 2026 г., 14:45 UTC".  Month names are in
    nominative case for clarity.
    """
    if dt is None:
        return None
    months = [
        'января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
        'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря'
    ]
    month_name = months[dt.month - 1]
    return f"{dt.day} {month_name} {dt.year} г., {dt:%H:%M} UTC"

def create_radar_plot(
    data: np.ndarray,
    title: str,
    station_code: str = 'kokx',
    max_range_km: float = 250.0,
    timestamp: Optional[datetime] = None,
    ring_intervals_km: Optional[List[float]] = None,
    azimuth_angles_deg: Optional[List[int]] = None
) -> bytes:
    """Render a single radar image with orientation aids and a legend.

    Parameters
    ----------
    data : np.ndarray
        Two‑dimensional array of reflectivity values.  Values < 1 are
        considered no‑data and masked out.  The data should already be
        scaled to dBZ units if desired.
    title : str
        Title displayed above the plot.  Avoid long sentences; use concise
        phrases to prevent clutter.
    station_code : str, default 'kokx'
        Identifier for the radar site.  Must exist in RADAR_COORDS.  The
        marker and orientation aids will be centred on this station.
    max_range_km : float, default 250.0
        Maximum range for the plot.  Circles and radial lines will extend to
        this distance.
    timestamp : datetime, optional
        When provided, a timestamp annotation will be added at the bottom
        of the figure in Russian locale.  Use naive UTC datetimes or
        timezone‑aware objects (only the time component is displayed).
    ring_intervals_km : list[float], optional
        Distances in kilometres at which to draw concentric rings.  If
        ``None``, defaults to five evenly spaced rings.
    azimuth_angles_deg : list[int], optional
        Angles in degrees for radial lines, measured clockwise from north.
        Common values are [0, 45, 90, …].  If ``None``, uses eight
        compass points.

    Returns
    -------
    bytes
        PNG image bytes of the rendered plot.
    """
    cmap, norm = get_radar_colormap()

    # Mask zeros to make them transparent
    data = np.ma.masked_where(data < 1.0, data)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    # Transform radar coordinates to Web Mercator
    lon, lat = RADAR_COORDS.get(station_code.lower(), RADAR_COORDS['default'])
    if Transformer is None:
        raise ImportError("pyproj is required for coordinate transformations")
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    center_x, center_y = transformer.transform(lon, lat)

    offset_m = max_range_km * 1000.0
    extent = [center_x - offset_m, center_x + offset_m, center_y - offset_m, center_y + offset_m]

    # Plot reflectivity overlay
    im = ax.imshow(
        data,
        cmap=cmap,
        norm=norm,
        extent=extent,
        origin='upper',
        alpha=0.8,
        zorder=2
    )

    # Add base map underlay
    if ctx is not None:
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zorder=1)
        except Exception:
            # If tile fetching fails (e.g. offline), ignore and continue
            pass

    # Draw range rings and radial lines
    if ring_intervals_km is None:
        # Choose five rings at 20%, 40%, … of max_range
        ring_intervals_km = [max_range_km * i / 5.0 for i in range(1, 6)]
    if azimuth_angles_deg is None:
        azimuth_angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]

    # A subtle grey for orientation aids
    ring_color = '#666666'
    ring_alpha = 0.5
    line_style = '--'

    # Draw concentric circles with labels
    for dist_km in ring_intervals_km:
        radius_m = dist_km * 1000.0
        circle = Circle(
            (center_x, center_y),
            radius_m,
            edgecolor=ring_color,
            facecolor='none',
            linewidth=0.8,
            linestyle=line_style,
            alpha=ring_alpha,
            zorder=2.5
        )
        ax.add_patch(circle)
        # Place distance label slightly outside the circle on the east side
        ax.text(
            center_x + radius_m + (0.02 * offset_m),
            center_y,
            f"{int(dist_km)} км",
            color=ring_color,
            fontsize=8,
            va='center',
            ha='left',
            alpha=ring_alpha
        )

    # Draw radial lines and azimuth labels
    for angle in azimuth_angles_deg:
        angle_rad = math.radians(angle)
        x_end = center_x + offset_m * math.sin(angle_rad)  # note: y increases northwards, x eastwards
        y_end = center_y + offset_m * math.cos(angle_rad)
        ax.plot(
            [center_x, x_end],
            [center_y, y_end],
            color=ring_color,
            linewidth=0.8,
            linestyle=line_style,
            alpha=ring_alpha,
            zorder=2.5
        )
        # Label azimuth at end of line
        ax.text(
            x_end,
            y_end,
            f"{angle}°",
            color=ring_color,
            fontsize=7,
            va='center',
            ha='center',
            alpha=ring_alpha,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.4, pad=1)
        )

    # Mark the radar site
    ax.plot(center_x, center_y, 'k+', markersize=12, zorder=4, markeredgewidth=2)
    ax.text(
        center_x + 0.02 * offset_m,
        center_y + 0.02 * offset_m,
        station_code.upper(),
        weight='bold',
        fontsize=12,
        zorder=4,
        path_effects=[patheffects.withStroke(linewidth=3, foreground='white')]
    )

    # Title
    ax.set_title(title, fontsize=14, pad=15)

    # Hide axes for a cleaner look
    ax.axis('off')

    # Add colourbar legend on the right
    cax = fig.add_axes([0.9, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('дБЗ', rotation=90, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Timestamp annotation
    ts_str = _format_timestamp(timestamp)
    if ts_str:
        # Place at bottom left inside the figure
        ax.text(
            0.01,
            0.01,
            ts_str,
            transform=ax.transAxes,
            fontsize=9,
            color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=2)
        )

    # Save to buffer
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
    history_timestamps: Optional[List[datetime]] = None
) -> List[bytes]:
    """Generate a sequence of radar plots for history and forecast.

    Parameters
    ----------
    input_seq : np.ndarray
        Array with shape (T_in, H, W) representing the history frames.
    pred_seq : np.ndarray
        Array with shape (T_out, H, W) representing forecast frames.
    input_len : int
        Length of the input sequence (used to compute lead times).
    station_code : str, default 'kokx'
        Radar identifier to centre the plots.
    start_datetime : datetime, optional
        Timestamp corresponding to the LAST element of ``input_seq`` (T-0).
    max_range_km : float, default 250.0
        The maximum range used for plotting and orientation aids.
    history_timestamps : list of datetime, optional
        Explicit timestamps for each frame in ``input_seq``. If provided,
        takes precedence over ``start_datetime`` for history frames.

    Returns
    -------
    list of bytes
        A list of PNG images representing the historical and forecast frames.
    """
    images: List[bytes] = []
    # History (past frames)
    for i in range(input_seq.shape[0]):
        lead_time = (input_len - i - 1) * -15
        label = f"История (T{lead_time} мин)" if lead_time != 0 else "Сейчас (T-0)"
        
        # Compute timestamp for this frame
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
                timestamp=ts
            )
        )
    # Forecast (future frames)
    # Note: start_datetime for forecast ALWAYS refers to the time of the LAST history frame (T-0)
    base_forecast_ts = start_datetime
    if history_timestamps and not base_forecast_ts:
        base_forecast_ts = history_timestamps[-1]

    for i in range(pred_seq.shape[0]):
        lead_time = (i + 1) * 15
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
                timestamp=ts
            )
        )
    return images
