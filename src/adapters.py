import os
import numpy as np
import pathlib
import requests
import io
import ftplib
import tempfile
import datetime
from PIL import Image
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
from bufr_decoder import MRLBufrDecoder
from nexrad_decoder import NEXRADDecoder

# Mapping of major NOAA NEXRAD stations to readable names
NEXRAD_STATIONS = {
    'kokx': 'Нью-Йорк Сити, NY',
    'kdtx': 'Детройт, MI',
    'klot': 'Чикаго, IL',
    'kbgm': 'Бингемтон, NY',
    'kewx': 'Остин / Сан-Антонио, TX',
    'tjua': 'Сан-Хуан, Пуэрто-Рико',
    'kffc': 'Атланта, GA',
    'kamx': 'Майами, FL',
    'kbox': 'Бостон, MA',
    'kgyx': 'Портленд, ME',
    'kilx': 'Чикаго (Lincoln), IL',
    'klwx': 'Вашингтон, DC'
}

class BaseRadarAdapter(ABC):
    """Abstract base class for all radar data adapters."""
    @abstractmethod
    def get_latest_sequence(self, seq_length: int) -> Tuple[np.ndarray, List[datetime.datetime], str]:
        """Fetch the latest sequence of radar frames. Returns (sequence, timestamps, status_message)."""
        pass

class LocalDirectoryAdapter(BaseRadarAdapter):
    """Adapter for loading radar data from a local directory."""
    def __init__(self, directory: str, grid_size=(256, 256)):
        self.directory = pathlib.Path(directory)
        self.grid_size = grid_size
        self.decoder = MRLBufrDecoder(grid_size=grid_size)

    def get_latest_sequence(self, seq_length: int) -> Tuple[np.ndarray, List[datetime.datetime], str]:
        if not self.directory.exists():
            raise ValueError(f"Директория {self.directory} не существует.")
            
        files = sorted(
            [p for p in self.directory.iterdir() if p.suffix in ('.bufr', '.npy', '.npz')],
            key=os.path.getmtime,
            reverse=True
        )
        
        if len(files) < seq_length:
            msg = f"В папке всего {len(files)} файлов. Использован демо-режим."
            dummy_seq, dummy_ts = self._get_fallback_data(seq_length)
            return dummy_seq, dummy_ts, msg
        
        latest_files = files[:seq_length][::-1]
        sequence = []
        timestamps = []
        for f in latest_files:
            try:
                # In a real app, we'd extract time from file content or metadata
                ts = datetime.datetime.fromtimestamp(os.path.getmtime(f))
                if f.suffix == '.bufr':
                    grid = self.decoder.decode(str(f))
                elif f.suffix == '.npy':
                    grid = np.load(f)
                elif f.suffix == '.npz':
                    grid = np.load(f)['arr_0']
                sequence.append(grid)
                timestamps.append(ts)
            except:
                sequence.append(self.decoder._generate_fallback_grid(str(f)))
                timestamps.append(datetime.datetime.now())
            
        return np.stack(sequence, axis=0), timestamps, "Данные загружены из локальной папки."

    def _get_fallback_data(self, seq_length: int):
        sequence = []
        timestamps = []
        now = datetime.datetime.now()
        for i in range(seq_length):
            sequence.append(self.decoder._generate_fallback_grid(f"fallback_{i}"))
            timestamps.append(now - datetime.timedelta(minutes=(seq_length-i-1)*15))
        return np.stack(sequence, axis=0), timestamps

class NOAAAWSAdapter(BaseRadarAdapter):
    """Modern adapter for fetching radar data from Amazon S3 via nexradaws."""
    STATION_MAP = NEXRAD_STATIONS

    def __init__(self, grid_size=(256, 256)):
        import nexradaws
        self.conn = nexradaws.NexradAwsInterface()
        self.grid_size = grid_size
        self.decoder = NEXRADDecoder(grid_size=grid_size)

    def get_latest_sequence(self, seq_length: int, station_code: str = 'kokx', end_time: Optional[datetime.datetime] = None) -> Tuple[np.ndarray, List[datetime.datetime], str]:
        try:
            now = end_time or datetime.datetime.now(datetime.UTC)
            # Fetch scans for the last 6 hours to find a continuous sequence
            scans = self.conn.get_avail_scans(now.year, now.month, now.day, station_code.upper())
            # Filter out metadata files
            scans = [s for s in scans if not s.filename.endswith('_MDM')]
            
            # If not enough scans today, check yesterday
            if len(scans) < seq_length + 2:
                yesterday = now - datetime.timedelta(days=1)
                y_scans = self.conn.get_avail_scans(yesterday.year, yesterday.month, yesterday.day, station_code.upper())
                y_scans = [s for s in y_scans if not s.filename.endswith('_MDM')]
                scans = y_scans + scans

            # Filter out scans newer than 'now' if end_time was specified
            if end_time:
                scans = [s for s in scans if s.scan_time <= end_time]

            # Sort by time
            scans.sort(key=lambda x: x.scan_time)
            
            # Find the best continuous sequence ending at the latest available scan
            # We want seq_length scans with ~15 min intervals
            target_interval = datetime.timedelta(minutes=15)
            tolerance = datetime.timedelta(minutes=5)
            
            selected_scans = []
            if not scans:
                return self._get_fallback_data(seq_length, "Нет доступных сканов на AWS.")

            # Backwards search for a sequence
            curr_idx = len(scans) - 1
            selected_scans.append(scans[curr_idx])
            
            while len(selected_scans) < seq_length and curr_idx > 0:
                prev_idx = curr_idx - 1
                time_diff = selected_scans[-1].scan_time - scans[prev_idx].scan_time
                
                if time_diff >= (target_interval - tolerance) and time_diff <= (target_interval + tolerance):
                    selected_scans.append(scans[prev_idx])
                    curr_idx = prev_idx
                elif time_diff < (target_interval - tolerance):
                    # Scan too close, skip it
                    curr_idx = prev_idx
                else:
                    # Gap too large! Try to find another sequence or just stop
                    break
            
            if len(selected_scans) < seq_length:
                # If we couldn't find a perfect sequence, just take the last N
                selected_scans = scans[-seq_length:]
                status = f"Внимание: последовательность может быть нерегулярной (пробелы в данных)."
            else:
                status = "Последовательность успешно верифицирована (15-мин интервалы)."

            selected_scans.sort(key=lambda x: x.scan_time)
            
            sequence = []
            timestamps = []
            
            with tempfile.TemporaryDirectory() as tmpdir:
                results = self.conn.download(selected_scans, tmpdir)
                for scan in selected_scans:
                    local_path = [r.filepath for r in results.success if r.scan.filename == scan.filename]
                    if local_path:
                        try:
                            grid = self.decoder.decode(local_path[0])
                        except:
                            grid = self.decoder._generate_fallback_grid(local_path[0])
                        sequence.append(grid)
                        timestamps.append(scan.scan_time)
                    else:
                        sequence.append(self.decoder._generate_fallback_grid(scan.filename))
                        timestamps.append(scan.scan_time)
            
            station_name = self.STATION_MAP.get(station_code.lower(), station_code.upper())
            return np.stack(sequence, axis=0), timestamps, f"AWS S3: {station_name} | {status}"

        except Exception as e:
            return self._get_fallback_data(seq_length, f"Ошибка AWS ({station_code}): {str(e)}")

    def _get_fallback_data(self, seq_length: int, reason: str):
        sequence = []
        timestamps = []
        now = datetime.datetime.now()
        for i in range(seq_length):
            sequence.append(self.decoder._generate_fallback_grid(f"aws_fallback_{i}"))
            timestamps.append(now - datetime.timedelta(minutes=(seq_length-i-1)*15))
        full_reason = f"РЕЖИМ ДЕМО (Данные не получены: {reason})"
        return np.stack(sequence, axis=0), timestamps, full_reason

class NOAAFTPAdapter(BaseRadarAdapter):
    """Adapter for fetching latest radar data from NOAA NWS FTP."""
    
    STATION_MAP = NEXRAD_STATIONS

    def __init__(self, grid_size=(256, 256)):
        self.host = "tgftp.nws.noaa.gov"
        self.base_path = "/SL.us008001/DF.of/DC.radar/DS.p94r3/"
        self.grid_size = grid_size
        self.decoder = NEXRADDecoder(grid_size=grid_size)

    def get_available_stations(self) -> list:
        """Returns a list of dictionaries with station codes and names."""
        try:
            ftp = ftplib.FTP(self.host, timeout=10)
            ftp.login()
            ftp.cwd(self.base_path)
            dirs = [d for d in ftp.nlst() if d.startswith("SI.")]
            ftp.quit()
            
            stations = []
            for d in dirs:
                code = d.replace("SI.", "").lower()
                name = self.STATION_MAP.get(code, f"Радар {code.upper()}")
                stations.append({'code': code, 'name': name})
            
            # Sort with known mapped stations first
            stations.sort(key=lambda x: (x['code'] not in self.STATION_MAP, x['name']))
            return stations
        except Exception as e:
            print(f"Error fetching stations: {e}")
            return []

    def get_available_times(self, station_code: str) -> list:
        """Returns a list of recent files (representing time) for a station."""
        try:
            ftp = ftplib.FTP(self.host, timeout=10)
            ftp.login()
            station_path = f"{self.base_path}SI.{station_code}/"
            ftp.cwd(station_path)
            files = sorted([f for f in ftp.nlst() if f.startswith("sn.") and f != "sn.last"])
            ftp.quit()
            
            # Return last 20 files as available times (they represent recent slices)
            # In a real app, we'd parse MDTM to show actual HH:MM, but filenames indicate order
            times = []
            for f in files[-20:]:
                times.append({'id': f, 'label': f"Срез {f.split('.')[-1]}"})
            return times[::-1] # Newest first
        except Exception as e:
            print(f"Error fetching times: {e}")
            return []

    def get_latest_sequence(self, seq_length: int, station_code: str = 'kokx', end_file_id: str = 'latest') -> Tuple[np.ndarray, List[datetime.datetime], str]:
        try:
            ftp = ftplib.FTP(self.host, timeout=10)
            ftp.login() 
            station_path = f"{self.base_path}SI.{station_code.lower()}/"
            ftp.cwd(station_path)
            
            files = sorted([f for f in ftp.nlst() if f.startswith("sn.") and f != "sn.last"])
            if len(files) < seq_length:
                ftp.quit()
                return self._get_fallback_data(seq_length, f"Недостаточно файлов для станции {station_code.upper()}.")
            
            if end_file_id == 'latest':
                target_files = files[-seq_length:]
            else:
                if end_file_id not in files:
                    ftp.quit()
                    return self._get_fallback_data(seq_length, f"Файл {end_file_id} не найден.")
                end_idx = files.index(end_file_id)
                start_idx = end_idx - seq_length + 1
                if start_idx < 0:
                    ftp.quit()
                    return self._get_fallback_data(seq_length, f"Недостаточно истории перед файлом {end_file_id}.")
                target_files = files[start_idx:end_idx+1]
                
            sequence = []
            timestamps = []
            with tempfile.TemporaryDirectory() as tmpdir:
                for f_name in target_files:
                    local_path = os.path.join(tmpdir, f_name)
                    with open(local_path, 'wb') as local_file:
                        ftp.retrbinary(f"RETR {f_name}", local_file.write)
                    
                    # Try to get timestamp from FTP if possible, else use modification time
                    try:
                        mdtm = ftp.voidcmd(f"MDTM {f_name}")[4:].strip()
                        ts = datetime.datetime.strptime(mdtm, '%Y%m%d%H%M%S')
                    except:
                        ts = datetime.datetime.now()

                    try:
                        grid = self.decoder.decode(local_path)
                    except:
                        grid = self.decoder._generate_fallback_grid(local_path)
                    sequence.append(grid)
                    timestamps.append(ts)
                    
            ftp.quit()
            station_name = self.STATION_MAP.get(station_code.lower(), station_code.upper())
            time_label = "Последние данные" if end_file_id == 'latest' else f"Срез {end_file_id}"
            return np.stack(sequence, axis=0), timestamps, f"NOAA FTP: {station_name} | {time_label}"
        except Exception as e:
            return self._get_fallback_data(seq_length, f"Ошибка FTP ({station_code}): {str(e)}. Демо-режим.")

    def _get_fallback_data(self, seq_length: int, reason: str) -> Tuple[np.ndarray, List[datetime.datetime], str]:
        sequence = []
        timestamps = []
        now = datetime.datetime.now()
        for i in range(seq_length):
            grid = self.decoder._generate_fallback_grid(f"fallback_{i}")
            sequence.append(grid)
            timestamps.append(now - datetime.timedelta(minutes=(seq_length-i-1)*15))
        full_reason = f"РЕЖИМ ДЕМО (FTP ошибка: {reason})"
        return np.stack(sequence, axis=0), timestamps, full_reason

class RainViewerAdapter(BaseRadarAdapter):
    """Adapter for fetching latest radar data from RainViewer API."""
    def __init__(self, grid_size=(256, 256)):
        self.api_url = "https://api.rainviewer.com/public/weather-maps.json"
        self.grid_size = grid_size
        self.decoder = MRLBufrDecoder(grid_size=grid_size)

    def get_latest_sequence(self, seq_length: int) -> Tuple[np.ndarray, List[datetime.datetime], str]:
        try:
            response = requests.get(self.api_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            host = data.get('host', 'https://tilecache.rainviewer.com')
            past_frames = data.get('radar', {}).get('past', [])
            if len(past_frames) < seq_length:
                return self._get_fallback_data(seq_length, "Недостаточно кадров в RainViewer.")
                
            latest_frames = past_frames[-seq_length:]
            sequence = []
            timestamps = []
            for frame in latest_frames:
                path = frame['path']
                ts = datetime.datetime.fromtimestamp(frame['time'], datetime.UTC)
                tile_url = f"{host}{path}/256/0/0/0/1/1_1.png"
                img_resp = requests.get(tile_url, timeout=5)
                if img_resp.status_code != 200:
                    grid = np.zeros(self.grid_size, dtype=np.float32)
                else:
                    img = Image.open(io.BytesIO(img_resp.content)).convert('L')
                    img = img.resize(self.grid_size)
                    grid = np.array(img, dtype=np.float32) * (70.0 / 255.0)
                sequence.append(grid)
                timestamps.append(ts)
                
            return np.stack(sequence, axis=0), timestamps, "Данные получены через RainViewer API."
        except Exception as e:
            return self._get_fallback_data(seq_length, f"Ошибка API: {str(e)}. Активирован демо-режим.")

    def _get_fallback_data(self, seq_length: int, reason: str) -> Tuple[np.ndarray, List[datetime.datetime], str]:
        sequence = []
        timestamps = []
        now = datetime.datetime.now()
        for i in range(seq_length):
            grid = self.decoder._generate_fallback_grid(f"fallback_rv_{i}")
            sequence.append(grid)
            timestamps.append(now - datetime.timedelta(minutes=(seq_length-i-1)*15))
        full_reason = f"РЕЖИМ ДЕМО (API ошибка: {reason})"
        return np.stack(sequence, axis=0), timestamps, full_reason

from typing import Tuple
