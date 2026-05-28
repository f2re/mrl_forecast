import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from typing import List, Tuple

def get_radar_colormap():
    """Returns a standard professional dBZ colormap and norm."""
    clevs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
    ccols = [
        '#ffffff', # 0-5
        '#00ecec', # 5-10
        '#01a0f6', # 10-15
        '#0000f6', # 15-20
        '#00ff00', # 20-25
        '#00c800', # 25-30
        '#009000', # 30-35
        '#ffff00', # 35-40
        '#e7c000', # 40-45
        '#ff9000', # 45-50
        '#ff0000', # 50-55
        '#d60000', # 55-60
        '#c00000', # 60-65
        '#ff00ff', # 65-70
        '#9955c9'  # 70+
    ]
    cmap = colors.ListedColormap(ccols)
    norm = colors.BoundaryNorm(clevs, cmap.N)
    return cmap, norm

def create_radar_plot(
    data: np.ndarray, 
    title: str, 
    center: Tuple[float, float] = (0, 0), 
    max_range_km: float = 250.0
) -> bytes:
    """
    Generates a professional radar plot with MRL circles and radii.
    Returns PNG bytes.
    """
    cmap, norm = get_radar_colormap()
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    
    # Plot radar data
    # Assuming data is square grid covering [-max_range, max_range]
    extent = [-max_range_km, max_range_km, -max_range_km, max_range_km]
    im = ax.imshow(data, cmap=cmap, norm=norm, extent=extent, origin='upper')
    
    # Draw MRL range rings
    rings = [50, 100, 150, 200, 250]
    for r in rings:
        circle = plt.Circle((0, 0), r, color='gray', fill=False, linestyle='--', alpha=0.5, linewidth=0.8)
        ax.add_patch(circle)
        ax.text(0, r + 2, f"{r} km", color='gray', fontsize=8, ha='center')

    # Draw radii (azimuth lines)
    for angle in range(0, 360, 45):
        rad = np.radians(angle)
        ax.plot([0, max_range_km * np.sin(rad)], [0, max_range_km * np.cos(rad)], color='gray', alpha=0.3, linewidth=0.5)

    # Station marker
    ax.plot(0, 0, 'k+', markersize=10)
    
    ax.set_title(title)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Distance (km)")
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Reflectivity (dBZ)')
    
    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def generate_sequence_plots(
    input_seq: np.ndarray, 
    pred_seq: np.ndarray, 
    input_len: int
) -> List[bytes]:
    """Generates a list of PNG images for history and forecast."""
    images = []
    
    # History
    for i in range(input_seq.shape[0]):
        lead_time = (input_len - i - 1) * -15
        img = create_radar_plot(input_seq[i] * 70.0, f"History T{lead_time if lead_time != 0 else '-0'} min")
        images.append(img)
        
    # Forecast
    for i in range(pred_seq.shape[0]):
        lead_time = (i + 1) * 15
        img = create_radar_plot(pred_seq[i] * 70.0, f"Forecast T+{lead_time} min")
        images.append(img)
        
    return images
