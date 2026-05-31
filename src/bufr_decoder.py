import os
import numpy as np
try:
    import eccodes
except ImportError:
    eccodes = None
from scipy.interpolate import griddata
import argparse
from typing import Optional, Tuple
from radar_pipeline import RadarDecodeError

class MRLBufrDecoder:
    """
    Decodes MRL radar data from BUFR format and interpolates it to a regular grid.
    Handles standard radar descriptors (reflectivity, azimuth, range).
    """
    def __init__(self, grid_size=(256, 256), max_range_km=250.0):
        self.grid_size = grid_size
        self.max_range_km = max_range_km

    def decode(self, bufr_path: str) -> np.ndarray:
        """
        Reads a BUFR file and returns a decoded Cartesian grid (dBZ).
        """
        if eccodes is None:
            raise ImportError("eccodes library not found. Please install it with 'pip install eccodes'.")
            
        reflectivity = []
        azimuths = []
        ranges = []
        
        with open(bufr_path, 'rb') as f:
            # Check for BUFR marker
            header = f.read(4)
            if header != b'BUFR':
                raise RadarDecodeError(f"{bufr_path} is not a valid BUFR file")
            f.seek(0)
            
            while True:
                bufr = eccodes.codes_bufr_new_from_file(f)
                if bufr is None:
                    break
                
                try:
                    eccodes.codes_set(bufr, 'unpack', 1)
                    
                    try:
                        data = eccodes.codes_get_array(bufr, 'snrReflectivity')
                        az = eccodes.codes_get_array(bufr, 'bearing')
                        dist = eccodes.codes_get_array(bufr, 'range')
                        
                        reflectivity.extend(data)
                        azimuths.extend(az)
                        ranges.extend(dist)
                    except eccodes.KeyValueNotFoundError:
                        try:
                            data = eccodes.codes_get_array(bufr, 'reflectivity')
                            az = eccodes.codes_get_array(bufr, 'bearing')
                            dist = eccodes.codes_get_array(bufr, 'range')
                            reflectivity.extend(data)
                            azimuths.extend(az)
                            ranges.extend(dist)
                        except:
                            continue
                finally:
                    eccodes.codes_release(bufr)
        
        if not reflectivity:
            raise RadarDecodeError(f"No reflectivity descriptors found in {bufr_path}")
            
        return self._polar_to_cartesian(
            np.array(azimuths), 
            np.array(ranges) / 1000.0, 
            np.array(reflectivity)
        )

    def _polar_to_cartesian(self, azimuths: np.ndarray, ranges: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Converts polar radar data to a regular Cartesian grid.
        """
        mask = (values >= 0) & (ranges <= self.max_range_km)
        az_filt = azimuths[mask]
        ra_filt = ranges[mask]
        va_filt = values[mask]
        
        if len(va_filt) == 0:
            return np.zeros(self.grid_size)

        az_rad = np.radians(az_filt)
        x = ra_filt * np.sin(az_rad)
        y = ra_filt * np.cos(az_rad)
        
        xi = np.linspace(-self.max_range_km, self.max_range_km, self.grid_size[0])
        yi = np.linspace(-self.max_range_km, self.max_range_km, self.grid_size[1])
        grid_x, grid_y = np.meshgrid(xi, yi)
        
        grid_z = griddata((x, y), va_filt, (grid_x, grid_y), method='linear', fill_value=0)
        grid_z = np.nan_to_num(grid_z)
        
        return grid_z

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Path to BUFR file')
    parser.add_argument('--output', default='decoded_radar.npy', help='Output .npy file')
    args = parser.parse_args()
    
    decoder = MRLBufrDecoder()
    try:
        grid = decoder.decode(args.input)
        np.save(args.output, grid)
        print(f"Decoded grid saved to {args.output}")
    except Exception as e:
        print(f"Error: {e}")
