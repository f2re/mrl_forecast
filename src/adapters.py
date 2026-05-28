import os
import numpy as np
import pathlib
import requests
import io
import ftplib
import tempfile
from PIL import Image
from abc import ABC, abstractmethod
from typing import Tuple
from bufr_decoder import MRLBufrDecoder

class BaseRadarAdapter(ABC):
    """Abstract base class for all radar data adapters."""
    @abstractmethod
    def get_latest_sequence(self, seq_length: int) -> Tuple[np.ndarray, str]:
        """Fetch the latest sequence of radar frames. Returns (sequence, status_message)."""
        pass

class LocalDirectoryAdapter(BaseRadarAdapter):
    """Adapter for loading radar data from a local directory."""
    def __init__(self, directory: str, grid_size=(256, 256)):
        self.directory = pathlib.Path(directory)
        self.grid_size = grid_size
        self.decoder = MRLBufrDecoder(grid_size=grid_size)

    def get_latest_sequence(self, seq_length: int) -> Tuple[np.ndarray, str]:
        if not self.directory.exists():
            raise ValueError(f"Директория {self.directory} не существует.")
            
        files = sorted(
            [p for p in self.directory.iterdir() if p.suffix in ('.bufr', '.npy', '.npz')],
            key=os.path.getmtime,
            reverse=True
        )
        
        if len(files) < seq_length:
            # If not enough files, use what we have and fallback for the rest
            msg = f"В папке всего {len(files)} файлов. Использован демо-режим для дополнения."
            dummy_seq = self.decoder._generate_fallback_grid("dummy")
            sequence = [dummy_seq] * seq_length
            return np.stack(sequence, axis=0), msg
        
        latest_files = files[:seq_length][::-1]
        sequence = []
        for f in latest_files:
            try:
                if f.suffix == '.bufr':
                    grid = self.decoder.decode(str(f))
                elif f.suffix == '.npy':
                    grid = np.load(f)
                elif f.suffix == '.npz':
                    grid = np.load(f)['arr_0']
                sequence.append(grid)
            except:
                sequence.append(self.decoder._generate_fallback_grid(str(f)))
            
        return np.stack(sequence, axis=0), "Данные загружены из локальной папки."

class NOAAFTPAdapter(BaseRadarAdapter):
    """Adapter for fetching latest radar data from NOAA NWS FTP."""
    def __init__(self, grid_size=(256, 256)):
        self.host = "tgftp.nws.noaa.gov"
        # Verified working path for station KOKX
        self.base_path = "/SL.us008001/DF.of/DC.radar/DS.p94r3/SI.kokx/"
        self.grid_size = grid_size
        self.decoder = MRLBufrDecoder(grid_size=grid_size)

    def get_latest_sequence(self, seq_length: int) -> Tuple[np.ndarray, str]:
        try:
            ftp = ftplib.FTP(self.host, timeout=10)
            ftp.login() 
            ftp.cwd(self.base_path)
            
            # Get files starting with 'sn.'
            files = sorted([f for f in ftp.nlst() if f.startswith("sn.") and f != "sn.last"])
            if len(files) < seq_length:
                ftp.quit()
                return self._get_fallback_sequence(seq_length, "Недостаточно файлов на FTP.")
                
            latest_files = files[-seq_length:]
            sequence = []
            with tempfile.TemporaryDirectory() as tmpdir:
                for f_name in latest_files:
                    local_path = os.path.join(tmpdir, f_name)
                    with open(local_path, 'wb') as local_file:
                        ftp.retrbinary(f"RETR {f_name}", local_file.write)
                    grid = self.decoder.decode(local_path)
                    sequence.append(grid)
                    
            ftp.quit()
            return np.stack(sequence, axis=0), "Данные успешно получены с NOAA FTP (KOKX)."
        except Exception as e:
            return self._get_fallback_sequence(seq_length, f"Ошибка FTP: {str(e)}. Активирован демо-режим.")

    def _get_fallback_sequence(self, seq_length: int, reason: str) -> Tuple[np.ndarray, str]:
        sequence = []
        for i in range(seq_length):
            grid = self.decoder._generate_fallback_grid(f"fallback_{i}")
            sequence.append(grid)
        return np.stack(sequence, axis=0), reason

class RainViewerAdapter(BaseRadarAdapter):
    """Adapter for fetching latest radar data from RainViewer API."""
    def __init__(self, grid_size=(256, 256)):
        self.api_url = "https://api.rainviewer.com/public/weather-maps.json"
        self.grid_size = grid_size
        self.decoder = MRLBufrDecoder(grid_size=grid_size)

    def get_latest_sequence(self, seq_length: int) -> Tuple[np.ndarray, str]:
        try:
            response = requests.get(self.api_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            host = data.get('host', 'https://tilecache.rainviewer.com')
            past_frames = data.get('radar', {}).get('past', [])
            if len(past_frames) < seq_length:
                return self._get_fallback_sequence(seq_length, "Недостаточно кадров в RainViewer.")
                
            latest_frames = past_frames[-seq_length:]
            sequence = []
            for frame in latest_frames:
                path = frame['path']
                tile_url = f"{host}{path}/256/0/0/0/1/1_1.png"
                img_resp = requests.get(tile_url, timeout=5)
                if img_resp.status_code != 200:
                    grid = np.zeros(self.grid_size, dtype=np.float32)
                else:
                    img = Image.open(io.BytesIO(img_resp.content)).convert('L')
                    img = img.resize(self.grid_size)
                    grid = np.array(img, dtype=np.float32) * (70.0 / 255.0)
                sequence.append(grid)
                
            return np.stack(sequence, axis=0), "Данные получены через RainViewer API."
        except Exception as e:
            return self._get_fallback_sequence(seq_length, f"Ошибка API: {str(e)}. Активирован демо-режим.")

    def _get_fallback_sequence(self, seq_length: int, reason: str) -> Tuple[np.ndarray, str]:
        sequence = []
        for i in range(seq_length):
            grid = self.decoder._generate_fallback_grid(f"fallback_rv_{i}")
            sequence.append(grid)
        return np.stack(sequence, axis=0), reason

from typing import Tuple
