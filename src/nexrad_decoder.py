import numpy as np
from scipy.interpolate import griddata
from metpy.io import Level3File

class NEXRADDecoder:
    """
    Decodes NOAA NEXRAD Level III files (e.g. from tgftp.nws.noaa.gov)
    and interpolates polar data to a regular Cartesian grid.
    """
    def __init__(self, grid_size=(256, 256), max_range_km=250.0):
        self.grid_size = grid_size
        self.max_range_km = max_range_km

    def decode(self, path: str) -> np.ndarray:
        try:
            f = Level3File(path)
            datadict = f.sym_block[0][0]
            data = f.map_data(datadict['data']) # Shape: (azimuths, range_gates)
            
            num_azimuths, num_gates = data.shape
            
            # Create azimuths array (0 to 360)
            # Typically radials start from North (0) clockwise
            azimuths = np.linspace(0, 360, num_azimuths, endpoint=False)
            
            # For DS.p94r3 (Base Reflectivity 0.5 deg), resolution is usually 1 km per gate
            ranges = np.arange(num_gates) * 1.0 
            
            az_grid, r_grid = np.meshgrid(azimuths, ranges, indexing='ij')
            
            az_flat = az_grid.flatten()
            r_flat = r_grid.flatten()
            val_flat = data.flatten()
            
            # Handle masked arrays from MetPy
            if hasattr(val_flat, 'mask'):
                valid = ~val_flat.mask
                val_flat = val_flat[valid]
                az_flat = az_flat[valid]
                r_flat = r_flat[valid]
            else:
                valid = ~np.isnan(val_flat)
                val_flat = val_flat[valid]
                az_flat = az_flat[valid]
                r_flat = r_flat[valid]
                
            # Restrict to requested max range
            in_range = r_flat <= self.max_range_km
            val_flat = val_flat[in_range]
            az_flat = az_flat[in_range]
            r_flat = r_flat[in_range]
            
            # Convert polar to Cartesian
            az_rad = np.radians(az_flat)
            x = r_flat * np.sin(az_rad)
            y = r_flat * np.cos(az_rad)
            
            # Target Cartesian grid
            xi = np.linspace(-self.max_range_km, self.max_range_km, self.grid_size[0])
            yi = np.linspace(-self.max_range_km, self.max_range_km, self.grid_size[1])
            grid_x, grid_y = np.meshgrid(xi, yi)
            
            # Interpolate polar points onto the regular grid
            grid_z = griddata((x, y), val_flat, (grid_x, grid_y), method='linear', fill_value=0.0)
            grid_z = np.nan_to_num(grid_z)
            
            # Clip negative dBZ or extreme values
            grid_z = np.clip(grid_z, 0.0, 70.0)
            
            return grid_z
        except Exception as e:
            print(f"Error decoding NEXRAD file {path}: {e}")
            return np.zeros(self.grid_size)

if __name__ == '__main__':
    decoder = NEXRADDecoder()
    # Assuming 'test_sn.last' exists from previous test
    import os
    if os.path.exists('test_sn.last'):
        grid = decoder.decode('test_sn.last')
        print(f"Decoded grid shape: {grid.shape}, Max dBZ: {np.max(grid)}")
        
        # Test plot
        import matplotlib.pyplot as plt
        plt.imshow(grid, cmap='jet')
        plt.colorbar(label='dBZ')
        plt.savefig('test_plot.png')
        print("Saved test_plot.png")
