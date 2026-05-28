import os
import numpy as np
import pathlib
import requests
import io
from PIL import Image
from abc import ABC, abstractmethod
from bufr_decoder import MRLBufrDecoder

class BaseRadarAdapter(ABC):
    """Abstract base class for all radar data adapters."""
    @abstractmethod
    def get_latest_sequence(self, seq_length: int) -> np.ndarray:
        """Fetch the latest sequence of radar frames."""
        pass

class LocalDirectoryAdapter(BaseRadarAdapter):
    """Adapter for loading radar data from a local directory."""
    def __init__(self, directory: str, grid_size=(256, 256)):
        self.directory = pathlib.Path(directory)
        self.grid_size = grid_size
        self.decoder = MRLBufrDecoder(grid_size=grid_size)

    def get_latest_sequence(self, seq_length: int) -> np.ndarray:
        # Find all .bufr, .npy, .npz files and sort by modification time
        files = sorted(
            [p for p in self.directory.iterdir() if p.suffix in ('.bufr', '.npy', '.npz')],
            key=os.path.getmtime,
            reverse=True
        )
        
        if len(files) < seq_length:
            raise ValueError(f"Not enough files in {self.directory}. Found {len(files)}, need {seq_length}.")
        
        # Take latest N files and reverse to get chronological order
        latest_files = files[:seq_length][::-1]
        
        sequence = []
        for f in latest_files:
            if f.suffix == '.bufr':
                grid = self.decoder.decode(str(f))
            elif f.suffix == '.npy':
                grid = np.load(f)
            elif f.suffix == '.npz':
                grid = np.load(f)['arr_0']
            else:
                continue
            
            # Ensure grid size matches
            if grid.shape != self.grid_size:
                # Basic resizing if needed could be added here
                pass
            sequence.append(grid)
            
        return np.stack(sequence, axis=0)

class RainViewerAdapter(BaseRadarAdapter):
    """Adapter for fetching latest radar data from RainViewer API."""
    def __init__(self, grid_size=(256, 256)):
        self.api_url = "https://api.rainviewer.com/public/weather-maps.json"
        self.grid_size = grid_size

    def get_latest_sequence(self, seq_length: int) -> np.ndarray:
        response = requests.get(self.api_url)
        response.raise_for_status()
        data = response.json()
        
        # 'radar' section contains historical composites
        past_frames = data.get('radar', {}).get('past', [])
        if len(past_frames) < seq_length:
            raise ValueError(f"RainViewer does not have enough past frames. Found {len(past_frames)}.")
            
        latest_frames = past_frames[-seq_length:]
        
        sequence = []
        for frame in latest_frames:
            ts = frame['time']
            # Fetch a 256x256 composite for the whole world or a specific region
            # We'll fetch a "product" or a single tile that covers a broad area for demonstration
            # Here we use the global coverage image if available or a specific zoom level tile
            # For simplicity, we'll fetch a single tile (Z=0, X=0, Y=0) or a larger one.
            # Zoom 0 is 1 tile for the whole world.
            tile_url = f"https://tilecache.rainviewer.com/v2/radar/{ts}/256/0/0/0/1/1_1.png"
            
            img_resp = requests.get(tile_url)
            img_resp.raise_for_status()
            
            img = Image.open(io.BytesIO(img_resp.content)).convert('L') # Convert to Grayscale
            img = img.resize(self.grid_size)
            
            # Convert to reflectivity-like values (0-70 range)
            # RainViewer images are colored, 'L' conversion gives intensity. 
            # We map 0-255 to 0-70 dBZ
            grid = np.array(img, dtype=np.float32) * (70.0 / 255.0)
            sequence.append(grid)
            
        return np.stack(sequence, axis=0)
