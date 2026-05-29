print("DEBUG: map_visualization: starting imports")
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
print("DEBUG: map_visualization: importing contextily")
import contextily as ctx
print("DEBUG: map_visualization: importing pyproj")
from pyproj import Transformer
from typing import List, Tuple
print("DEBUG: map_visualization: imports finished")

# Географические координаты радаров (Долгота, Широта)
RADAR_COORDS = {
    'kokx': (-72.86, 40.86),  # Нью-Йорк
    'kdtx': (-83.47, 42.69),  # Детройт
    'klot': (-88.08, 41.60),  # Чикаго
    'kamx': (-80.41, 25.61),  # Майами
    'default': (0.0, 0.0)     # Fallback
}

def get_radar_colormap():
    clevs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
    ccols = ['#ffffff00', '#00ecec', '#01a0f6', '#0000f6', '#00ff00', '#00c800', 
             '#009000', '#ffff00', '#e7c000', '#ff9000', '#ff0000', '#d60000', 
             '#c00000', '#ff00ff', '#9955c9'] # Первый цвет прозрачный для пустых зон
    cmap = colors.ListedColormap(ccols)
    norm = colors.BoundaryNorm(clevs, cmap.N)
    return cmap, norm

def create_radar_plot(
    data: np.ndarray, 
    title: str, 
    station_code: str = 'kokx', 
    max_range_km: float = 250.0
) -> bytes:
    cmap, norm = get_radar_colormap()
    
    # Скрываем нули (делаем прозрачными для наложения на карту)
    data = np.ma.masked_where(data < 1.0, data)
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    
    # Трансформация в Web Mercator (EPSG:3857) для Contextily
    lon, lat = RADAR_COORDS.get(station_code.lower(), RADAR_COORDS['default'])
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    center_x, center_y = transformer.transform(lon, lat)
    
    offset_m = max_range_km * 1000 # Перевод км в метры
    extent = [center_x - offset_m, center_x + offset_m, center_y - offset_m, center_y + offset_m]
    
    # Отрисовка МРЛ поверх карты
    im = ax.imshow(data, cmap=cmap, norm=norm, extent=extent, origin='upper', alpha=0.6, zorder=2)
    
    # Добавление картографической основы (с городами и рельефом)
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zorder=1)
    except Exception as e:
        print(f"Ошибка загрузки карты: {e}")

    # Маркер станции
    ax.plot(center_x, center_y, 'k+', markersize=12, zorder=3)
    ax.text(center_x + 5000, center_y + 5000, station_code.upper(), weight='bold', fontsize=12, zorder=3)

    ax.set_title(title, fontsize=14, pad=15)
    ax.axis('off') # Отключаем оси координат
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def generate_sequence_plots(
    input_seq: np.ndarray, 
    pred_seq: np.ndarray, 
    input_len: int,
    station_code: str = 'kokx'
) -> List[bytes]:
    images = []
    # История
    for i in range(input_seq.shape[0]):
        lead_time = (input_len - i - 1) * -15
        label = f"История (T{lead_time} мин)" if lead_time != 0 else "Сейчас (T-0)"
        images.append(create_radar_plot(input_seq[i] * 70.0, label, station_code))
        
    # Прогноз
    for i in range(pred_seq.shape[0]):
        lead_time = (i + 1) * 15
        images.append(create_radar_plot(pred_seq[i] * 70.0, f"Прогноз ИИ (T+{lead_time} мин)", station_code))
        
    return images
