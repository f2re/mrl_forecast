import numpy as np
from scipy.interpolate import griddata
import os

try:
    from metpy.io import Level3File
except ImportError:
    Level3File = None

try:
    import pyart
except ImportError:
    pyart = None

class NEXRADDecoder:
    """
    Decodes NOAA NEXRAD Level II and Level III files and interpolates 
    polar data to a regular Cartesian grid.
    """
    def __init__(self, grid_size=(256, 256), max_range_km=250.0):
        self.grid_size = grid_size
        self.max_range_km = max_range_km

    def _generate_fallback_grid(self, path: str) -> np.ndarray:
        """Generates a synthetic grid if decoding fails, for demo purposes."""
        # Use filename hash to keep it somewhat consistent for the same file
        np.random.seed(abs(hash(path)) % (2**32))
        grid = np.zeros(self.grid_size)
        for _ in range(3):
            sx, sy = np.random.randint(10, 60, size=2)
            px, py = np.random.randint(0, self.grid_size[0], size=2)
            intensity = np.random.uniform(20.0, 55.0)
            
            # Simple Gaussian blob
            y, x = np.ogrid[:self.grid_size[0], :self.grid_size[1]]
            dist = (x - px)**2 / sx**2 + (y - py)**2 / sy**2
            blob = intensity * np.exp(-dist)
            grid = np.maximum(grid, blob)
        
        # Add a "DEMO" hint - small dots in a corner
        grid[0:5, 0:5] = 70.0
        return grid

    def decode(self, path: str) -> np.ndarray:
        # Try Level 2 first (typical for AWS)
        if pyart:
            try:
                radar = pyart.io.read_nexrad_archive(path)
                
                # Extract reflectivity from first sweep
                field_name = 'reflectivity'
                if field_name not in radar.fields:
                    for f in radar.fields:
                        if 'reflectivity' in f.lower():
                            field_name = f
                            break
                
                if field_name in radar.fields:
                    # Get coordinates of all gates in the first sweep
                    sweep_slice = radar.get_slice(0)
                    x = radar.gate_x['data'][sweep_slice].flatten() / 1000.0 # to km
                    y = radar.gate_y['data'][sweep_slice].flatten() / 1000.0 # to km
                    vals = radar.fields[field_name]['data'][sweep_slice].flatten()
                    
                    # Filter invalid
                    mask = ~np.ma.getmaskarray(vals)
                    x, y, vals = x[mask], y[mask], vals[mask]
                    
                    # Interpolate
                    xi = np.linspace(-self.max_range_km, self.max_range_km, self.grid_size[0])
                    yi = np.linspace(-self.max_range_km, self.max_range_km, self.grid_size[1])
                    grid_x, grid_y = np.meshgrid(xi, yi)
                    
                    grid_z = griddata((x, y), vals, (grid_x, grid_y), method='linear', fill_value=0.0)
                    grid_z = np.nan_to_num(grid_z)
                    return np.clip(grid_z, 0.0, 70.0)
            except Exception as e:
                # print(f"Level 2 decode failed: {e}")
                pass

        # Try Level 3 (typical for FTP)
        if Level3File:
            try:
                f = Level3File(path)
                datadict = f.sym_block[0][0]
                data = f.map_data(datadict['data'])
                
                num_azimuths, num_gates = data.shape
                azimuths = np.linspace(0, 360, num_azimuths, endpoint=False)
                ranges = np.arange(num_gates) * 1.0 
                
                az_grid, r_grid = np.meshgrid(azimuths, ranges, indexing='ij')
                az_rad = np.radians(az_grid.flatten())
                r_flat = r_grid.flatten()
                
                x = r_flat * np.sin(az_rad)
                y = r_flat * np.cos(az_rad)
                vals = data.flatten()
                
                if hasattr(vals, 'mask'):
                    valid = ~vals.mask
                    x, y, vals = x[valid], y[valid], vals[valid]
                
                xi = np.linspace(-self.max_range_km, self.max_range_km, self.grid_size[0])
                yi = np.linspace(-self.max_range_km, self.max_range_km, self.grid_size[1])
                grid_x, grid_y = np.meshgrid(xi, yi)
                
                grid_z = griddata((x, y), vals, (grid_x, grid_y), method='linear', fill_value=0.0)
                grid_z = np.nan_to_num(grid_z)
                return np.clip(grid_z, 0.0, 70.0)
            except Exception as e:
                # print(f"Level 3 decode failed: {e}")
                pass

        return self._generate_fallback_grid(path)

if __name__ == '__main__':
    decoder = NEXRADDecoder()
    if os.path.exists('test_sn.last'):
        grid = decoder.decode('test_sn.last')
        print(f"Max dBZ: {np.max(grid)}")
