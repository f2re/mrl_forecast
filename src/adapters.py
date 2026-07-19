"""Radar source adapters with explicit observation trust boundaries."""

from __future__ import annotations

import datetime
import ftplib
import os
import pathlib
import re
import tempfile
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from bufr_decoder import MRLBufrDecoder
from nexrad_decoder import NEXRADDecoder
from radar_pipeline import (
    DemoRadarAdapter,
    RadarDecodeError,
    RadarFrame,
    RadarPipeline,
    RadarSequence,
    RadarSourceError,
)

NEXRAD_STATIONS = {
    "kokx": "Нью-Йорк Сити, NY",
    "kdtx": "Детройт, MI",
    "klot": "Чикаго, IL",
    "kbgm": "Бингемтон, NY",
    "kewx": "Остин / Сан-Антонио, TX",
    "tjua": "Сан-Хуан, Пуэрто-Рико",
    "kffc": "Атланта, GA",
    "kamx": "Майами, FL",
    "kbox": "Бостон, MA",
    "kgyx": "Портленд, ME",
    "kilx": "Чикаго (Lincoln), IL",
    "klwx": "Вашингтон, DC",
}

AWS_PUBLIC_REGION = "us-east-1"
FILE_TIMESTAMP = re.compile(r"(?P<date>\d{8})[_-]?(?P<time>\d{6})")


def configure_public_aws_region() -> None:
    """Keep public NOAA S3 access independent from a user's local AWS profile."""
    os.environ["AWS_DEFAULT_REGION"] = AWS_PUBLIC_REGION
    os.environ["AWS_REGION"] = AWS_PUBLIC_REGION


class BaseRadarAdapter(ABC):
    @abstractmethod
    def get_latest_sequence(self, seq_length: int) -> RadarSequence:
        """Fetch a timestamped sequence or raise a typed radar error."""


class LocalDirectoryAdapter(BaseRadarAdapter):
    """Load trusted BUFR or NumPy observations from a local directory."""

    def __init__(
        self,
        directory: str,
        grid_size: tuple[int, int] = (256, 256),
        pipeline: Optional[RadarPipeline] = None,
    ):
        self.directory = pathlib.Path(directory)
        self.pipeline = pipeline or RadarPipeline()
        self.decoder = MRLBufrDecoder(grid_size=grid_size)

    def get_latest_sequence(self, seq_length: int) -> RadarSequence:
        if not self.directory.exists():
            raise RadarSourceError(f"Директория {self.directory} не существует")

        paths = [
            path
            for path in self.directory.iterdir()
            if path.is_file() and path.suffix.lower() in (".bufr", ".npy", ".npz")
        ]
        frames: list[RadarFrame] = []
        errors = []
        for path in paths:
            try:
                frames.extend(self._load_frames(path))
            except RadarDecodeError as exc:
                errors.append(f"{path.name}: {exc}")

        frames.sort(key=lambda frame: frame.timestamp_utc)
        if len(frames) < seq_length:
            details = "; ".join(errors[-3:]) if errors else "нет дополнительных диагностик"
            raise RadarSourceError(
                f"В {self.directory} найдено {len(frames)} доверенных сроков, "
                f"требуется {seq_length}. {details}"
            )
        selected = frames[-seq_length:]
        return RadarSequence(
            frames=selected,
            source="local",
            message=f"Локальная папка: {len(selected)} наблюдаемых сроков",
        )

    def _load_frames(self, path: pathlib.Path) -> list[RadarFrame]:
        try:
            if path.suffix.lower() == ".npz":
                return self._load_npz_frames(path)
            if path.suffix.lower() == ".bufr":
                grid = self.decoder.decode(str(path))
            else:
                grid = np.load(path, allow_pickle=False)
            if np.asarray(grid).ndim != 2:
                raise RadarDecodeError(
                    f"Файл {path.name} без timestamps_utc должен содержать один растр [H,W]"
                )
            return [self._frame_from_array(grid, self._timestamp_from_name(path), path)]
        except RadarDecodeError:
            raise
        except Exception as exc:
            raise RadarDecodeError(f"Failed to load local radar data {path}: {exc}") from exc

    def _load_npz_frames(self, path: pathlib.Path) -> list[RadarFrame]:
        with np.load(path, allow_pickle=False) as payload:
            if "reflectivity" in payload:
                values = np.asarray(payload["reflectivity"], dtype=np.float32)
            elif "arr_0" in payload:
                values = np.asarray(payload["arr_0"], dtype=np.float32)
            else:
                raise RadarDecodeError(f"NPZ {path.name} has no reflectivity or arr_0 array")

            if values.ndim == 2:
                values = values[np.newaxis, ...]
            if values.ndim != 3:
                raise RadarDecodeError(f"Expected [T,H,W] in {path.name}, got {values.shape}")

            if "valid_mask" in payload:
                masks = np.asarray(payload["valid_mask"], dtype=bool)
                if masks.ndim == 2:
                    masks = masks[np.newaxis, ...]
            else:
                masks = np.isfinite(values)
            if masks.shape != values.shape:
                raise RadarDecodeError(
                    f"valid_mask shape {masks.shape} does not match reflectivity {values.shape}"
                )

            if "timestamps_utc" in payload:
                raw_timestamps = list(payload["timestamps_utc"])
                if len(raw_timestamps) != values.shape[0]:
                    raise RadarDecodeError("timestamps_utc length does not match sequence length")
                timestamps = [self._parse_timestamp(value) for value in raw_timestamps]
            elif values.shape[0] == 1:
                timestamps = [self._timestamp_from_name(path)]
            else:
                raise RadarDecodeError(
                    f"Sequence {path.name} requires timestamps_utc; file mtime is not accepted"
                )

            return [
                self._frame_from_array(values[index], timestamps[index], path, masks[index], index)
                for index in range(values.shape[0])
            ]

    def _frame_from_array(
        self,
        grid: np.ndarray,
        timestamp: datetime.datetime,
        path: pathlib.Path,
        valid_mask: Optional[np.ndarray] = None,
        sequence_index: Optional[int] = None,
    ) -> RadarFrame:
        values = np.asarray(grid, dtype=np.float32)
        expected = (self.pipeline.config.height, self.pipeline.config.width)
        if values.shape != expected:
            raise RadarDecodeError(
                f"Grid {values.shape} in {path.name} is incompatible with pipeline {expected}"
            )
        mask = np.isfinite(values) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
        if mask.shape != values.shape:
            raise RadarDecodeError("Local valid_mask does not match grid shape")
        masked_grid = np.ma.array(values, mask=~mask)
        provenance = {"path": str(path)}
        if sequence_index is not None:
            provenance["sequence_index"] = sequence_index
        return self.pipeline.frame_from_grid(
            masked_grid,
            timestamp_utc=timestamp,
            station="LOCAL",
            source="local",
            provenance=provenance,
        )

    @staticmethod
    def _parse_timestamp(value) -> datetime.datetime:
        text = str(value.decode("utf-8") if isinstance(value, bytes) else value)
        try:
            timestamp = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise RadarDecodeError(f"Invalid timestamp {text}") from exc
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=datetime.UTC)
        return timestamp.astimezone(datetime.UTC)

    @staticmethod
    def _timestamp_from_name(path: pathlib.Path) -> datetime.datetime:
        match = FILE_TIMESTAMP.search(path.name)
        if not match:
            raise RadarDecodeError(
                f"Observation timestamp is unavailable for {path.name}; file mtime is not accepted"
            )
        return datetime.datetime.strptime(
            match.group("date") + match.group("time"),
            "%Y%m%d%H%M%S",
        ).replace(tzinfo=datetime.UTC)


class NOAAAWSAdapter(BaseRadarAdapter):
    """Fetch NOAA NEXRAD Level II observations from the public AWS S3 archive."""

    STATION_MAP = NEXRAD_STATIONS

    def __init__(
        self,
        grid_size: tuple[int, int] = (256, 256),
        conn=None,
        pipeline: Optional[RadarPipeline] = None,
    ):
        configure_public_aws_region()
        if conn is None:
            import nexradaws

            conn = nexradaws.NexradAwsInterface()
        self.conn = conn
        self.pipeline = pipeline or RadarPipeline()
        self.grid_size = grid_size

    def get_latest_sequence(
        self,
        seq_length: int,
        station_code: str = "kokx",
        end_time: Optional[datetime.datetime] = None,
    ) -> RadarSequence:
        station = station_code.upper()
        now = end_time or datetime.datetime.now(datetime.UTC)
        try:
            scans = self._available_scans(now, station)
            selected_scans = self._select_scans(scans, seq_length, end_time=end_time)
            frames = self._download_and_process(selected_scans, station)
        except (RadarSourceError, RadarDecodeError):
            raise
        except Exception as exc:
            raise RadarSourceError(f"AWS source failed for {station}: {exc}") from exc

        station_name = self.STATION_MAP.get(station_code.lower(), station)
        return RadarSequence(
            frames=frames,
            source="aws",
            message=f"AWS S3: {station_name} | {len(frames)} observed frames",
        )

    def _available_scans(self, now: datetime.datetime, station: str) -> list:
        scans = self.conn.get_avail_scans(now.year, now.month, now.day, station)
        scans = [scan for scan in scans if not scan.filename.endswith("_MDM")]
        if len(scans) < 2:
            yesterday = now - datetime.timedelta(days=1)
            older = self.conn.get_avail_scans(
                yesterday.year,
                yesterday.month,
                yesterday.day,
                station,
            )
            scans = [scan for scan in older if not scan.filename.endswith("_MDM")] + scans
        return scans

    def _select_scans(
        self,
        scans: list,
        seq_length: int,
        *,
        end_time: Optional[datetime.datetime],
    ) -> list:
        if end_time is not None:
            scans = [scan for scan in scans if scan.scan_time <= end_time]
        scans = sorted(scans, key=lambda scan: scan.scan_time)
        if not scans:
            raise RadarSourceError("Нет доступных сканов AWS")

        selected = [scans[-1]]
        remaining = scans[:-1]
        target_step = datetime.timedelta(minutes=self.pipeline.config.time_step_minutes)
        tolerance = datetime.timedelta(minutes=4)
        while remaining and len(selected) < seq_length:
            target_time = selected[-1].scan_time - target_step
            candidate = min(remaining, key=lambda scan: abs(scan.scan_time - target_time))
            if abs(candidate.scan_time - target_time) > tolerance:
                break
            selected.append(candidate)
            remaining = [scan for scan in remaining if scan.scan_time < candidate.scan_time]
        if len(selected) != seq_length:
            raise RadarSourceError(
                f"Недостаточно регулярных AWS сканов: найдено {len(selected)}, требуется {seq_length}"
            )
        return sorted(selected, key=lambda scan: scan.scan_time)

    def _download_and_process(self, scans: list, station: str) -> List[RadarFrame]:
        frames = []
        with tempfile.TemporaryDirectory() as tmpdir:
            results = self.conn.download(scans, tmpdir)
            downloaded = {item.scan.filename: item.filepath for item in results.success}
            for scan in scans:
                path = downloaded.get(scan.filename)
                if path is None:
                    raise RadarSourceError(f"AWS download failed for {scan.filename}")
                frames.append(
                    self.pipeline.process_file(
                        path,
                        timestamp_utc=scan.scan_time,
                        station=station,
                        source="aws",
                    )
                )
        return frames


class NOAAFTPAdapter(BaseRadarAdapter):
    """Fetch NEXRAD Level III observations from NOAA FTP."""

    STATION_MAP = NEXRAD_STATIONS

    def __init__(self, grid_size: tuple[int, int] = (256, 256)):
        self.host = "tgftp.nws.noaa.gov"
        self.base_path = "/SL.us008001/DF.of/DC.radar/DS.p94r3/"
        self.pipeline = RadarPipeline()
        self.decoder = NEXRADDecoder(grid_size=grid_size)

    def get_available_stations(self) -> list:
        try:
            with ftplib.FTP(self.host, timeout=10) as ftp:
                ftp.login()
                ftp.cwd(self.base_path)
                directories = [item for item in ftp.nlst() if item.startswith("SI.")]
            stations = []
            for directory in directories:
                code = directory.replace("SI.", "").lower()
                stations.append({"code": code, "name": self.STATION_MAP.get(code, f"Радар {code.upper()}")})
            stations.sort(key=lambda item: (item["code"] not in self.STATION_MAP, item["name"]))
            return stations
        except Exception:
            return []

    def get_available_times(self, station_code: str) -> list:
        try:
            with ftplib.FTP(self.host, timeout=10) as ftp:
                ftp.login()
                ftp.cwd(f"{self.base_path}SI.{station_code}/")
                files = sorted([item for item in ftp.nlst() if item.startswith("sn.") and item != "sn.last"])
            return [{"id": name, "label": f"Срез {name.split('.')[-1]}"} for name in files[-20:][::-1]]
        except Exception:
            return []

    def get_latest_sequence(
        self,
        seq_length: int,
        station_code: str = "kokx",
        end_file_id: str = "latest",
    ) -> RadarSequence:
        try:
            with ftplib.FTP(self.host, timeout=10) as ftp:
                ftp.login()
                ftp.cwd(f"{self.base_path}SI.{station_code.lower()}/")
                files = sorted([item for item in ftp.nlst() if item.startswith("sn.") and item != "sn.last"])
                target_files = self._select_files(files, seq_length, end_file_id)
                frames = self._download_frames(ftp, target_files, station_code.upper())
        except (RadarSourceError, RadarDecodeError):
            raise
        except Exception as exc:
            raise RadarSourceError(f"FTP source failed for {station_code.upper()}: {exc}") from exc
        station_name = self.STATION_MAP.get(station_code.lower(), station_code.upper())
        return RadarSequence(
            frames=frames,
            source="ftp",
            message=f"NOAA FTP: {station_name} | {len(frames)} observed frames",
        )

    @staticmethod
    def _select_files(files: list[str], seq_length: int, end_file_id: str) -> list[str]:
        if len(files) < seq_length:
            raise RadarSourceError(f"Недостаточно FTP файлов: найдено {len(files)}, требуется {seq_length}")
        if end_file_id == "latest":
            return files[-seq_length:]
        if end_file_id not in files:
            raise RadarSourceError(f"FTP файл {end_file_id} не найден")
        end_index = files.index(end_file_id)
        start_index = end_index - seq_length + 1
        if start_index < 0:
            raise RadarSourceError(f"Недостаточно истории перед FTP файлом {end_file_id}")
        return files[start_index : end_index + 1]

    def _download_frames(self, ftp: ftplib.FTP, files: list[str], station: str) -> List[RadarFrame]:
        frames = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for filename in files:
                local_path = os.path.join(tmpdir, filename)
                with open(local_path, "wb") as local_file:
                    ftp.retrbinary(f"RETR {filename}", local_file.write)
                timestamp = self._ftp_timestamp(ftp, filename)
                grid = self.decoder.decode(local_path)
                frames.append(
                    self.pipeline.frame_from_grid(
                        grid,
                        timestamp_utc=timestamp,
                        station=station,
                        source="ftp",
                        provenance={"filename": filename},
                    )
                )
        return frames

    @staticmethod
    def _ftp_timestamp(ftp: ftplib.FTP, filename: str) -> datetime.datetime:
        try:
            value = ftp.voidcmd(f"MDTM {filename}")[4:].strip()
            return datetime.datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=datetime.UTC)
        except Exception as exc:
            raise RadarSourceError(f"FTP timestamp unavailable for {filename}: {exc}") from exc


class RainViewerAdapter(BaseRadarAdapter):
    """Reserved for a future location-aware RainViewer tile decoder."""

    def get_latest_sequence(self, seq_length: int) -> RadarSequence:
        raise RadarSourceError(
            "RainViewer operational ingestion is disabled until tile coordinates and palette decoding are configured"
        )
