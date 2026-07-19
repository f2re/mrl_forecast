"""DWD Open Data adapter for quantitative ODIM HDF5 reflectivity sweeps."""

from __future__ import annotations

import datetime
import pathlib
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urljoin

import requests

from radar_contract import RadarSourceCapabilities
from radar_pipeline import RadarDecodeError, RadarPipeline, RadarSequence, RadarSourceError
from source_access import RemoteRadarFile, SourceProbeResult, download_http, verify_http_download


@dataclass(frozen=True)
class DWDScanRef:
    filename: str
    url: str
    scan_time: datetime.datetime


class DWDOpenDataAdapter:
    """Load DWD precipitation reflectivity sweeps through the canonical pipeline."""

    BASE_URL = "https://opendata.dwd.de/weather/radar/sites/sweep_pcp_z/"
    CAPABILITIES = RadarSourceCapabilities(
        source_id="dwd-open-data",
        native_format="ODIM HDF5 precipitation sweep",
        quantitative_reflectivity=True,
        training_allowed=True,
        visualization_allowed=True,
        raw_polar_volume=False,
        notes="Open quantitative DBZH sweep; polarimetric quality-filtered product.",
        access_mode="open",
        probe_supported=True,
        download_supported=True,
        adapter_status="active",
        archive_note="Station directories expose recent ODIM HDF5 sweeps over direct HTTP.",
    )
    TIMESTAMP_PATTERN = re.compile(r"dbzh_00-(\d{12})", re.IGNORECASE)

    def __init__(
        self,
        *,
        pipeline: Optional[RadarPipeline] = None,
        timeout_seconds: int = 30,
        session: Optional[requests.Session] = None,
        filter_name: str = "filter_polarimetric",
        base_url: Optional[str] = None,
    ):
        if filter_name not in {"filter_polarimetric", "filter_simple"}:
            raise ValueError("Unsupported DWD filter")
        self.pipeline = pipeline or RadarPipeline.canonical()
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self.filter_name = filter_name
        self.base_url = (base_url or self.BASE_URL).rstrip("/") + "/"

    def list_stations(self) -> list[str]:
        response = self.session.get(self.base_url, timeout=self.timeout_seconds)
        response.raise_for_status()
        stations = set(
            re.findall(
                r'href=["\']([a-z0-9]{3})/["\']',
                response.text,
                flags=re.IGNORECASE,
            )
        )
        return sorted(station.lower() for station in stations)

    def list_scans(self, station_code: str) -> list[DWDScanRef]:
        station = self._station_code(station_code)
        directory_url = urljoin(self.base_url, f"{station}/hdf5/{self.filter_name}/")
        response = self.session.get(directory_url, timeout=self.timeout_seconds)
        response.raise_for_status()

        scans: dict[datetime.datetime, DWDScanRef] = {}
        hrefs = re.findall(r'href=["\']([^"\']+)["\']', response.text, flags=re.IGNORECASE)
        for href in hrefs:
            filename = pathlib.PurePosixPath(href.split("?", 1)[0]).name
            match = self.TIMESTAMP_PATTERN.search(filename)
            if not match:
                continue
            scan_time = datetime.datetime.strptime(match.group(1), "%Y%m%d%H%M").replace(
                tzinfo=datetime.UTC
            )
            scans[scan_time] = DWDScanRef(
                filename=filename,
                url=urljoin(directory_url, href),
                scan_time=scan_time,
            )
        return [scans[key] for key in sorted(scans)]

    def list_files(
        self,
        *,
        station: str = "ess",
        start: Optional[datetime.datetime] = None,
        end: Optional[datetime.datetime] = None,
        limit: int = 20,
        **_: Any,
    ) -> list[RemoteRadarFile]:
        scans = self.list_scans(station)
        if start is not None:
            start = self._ensure_utc(start)
            scans = [scan for scan in scans if scan.scan_time >= start]
        if end is not None:
            end = self._ensure_utc(end)
            scans = [scan for scan in scans if scan.scan_time <= end]
        scans = scans[-max(1, min(int(limit), 1000)) :]
        return [
            RemoteRadarFile(
                source_id=self.CAPABILITIES.source_id,
                file_id=scan.filename,
                filename=scan.filename,
                url=scan.url,
                native_format="ODIM_H5",
                timestamp_utc=scan.scan_time,
                station_id=station.upper(),
                metadata={"filter": self.filter_name},
            )
            for scan in reversed(scans)
        ]

    def probe(
        self,
        download_test: bool = False,
        station: str = "ess",
        **kwargs: Any,
    ) -> SourceProbeResult:
        try:
            stations = self.list_stations()
            selected_station = station.lower() if station.lower() in stations else (stations[0] if stations else station)
            files = self.list_files(station=selected_station, limit=1, **kwargs)
            sample = files[0] if files else None
            can_download = bool(sample) and (
                verify_http_download(sample, session=self.session)
                if download_test
                else True
            )
            return SourceProbeResult(
                source_id=self.CAPABILITIES.source_id,
                status="available" if sample else "degraded",
                reachable=True,
                can_list=True,
                can_download=can_download,
                credential_state="not_required",
                message=f"DWD source is reachable; {len(stations)} station directories found.",
                sample=sample.to_metadata() if sample else None,
            )
        except Exception as exc:
            return SourceProbeResult(
                source_id=self.CAPABILITIES.source_id,
                status="unavailable",
                reachable=False,
                can_list=False,
                can_download=False,
                credential_state="not_required",
                message=str(exc),
            )

    def download(self, remote: RemoteRadarFile, output_dir: str) -> dict:
        return download_http(remote, output_dir, session=self.session)

    def get_latest_sequence(
        self,
        seq_length: int,
        station_code: str = "ess",
        end_time: Optional[datetime.datetime] = None,
    ) -> RadarSequence:
        if seq_length < 1:
            raise ValueError("seq_length must be positive")
        station = self._station_code(station_code)
        scans = self.list_scans(station)
        if end_time is not None:
            end_time = self._ensure_utc(end_time)
            scans = [scan for scan in scans if scan.scan_time <= end_time]
        selected = self._select_regular_scans(scans, seq_length)
        frames = self._download_and_process(selected, station)
        return RadarSequence(
            frames=frames,
            source="dwd-open-data",
            message=(
                f"DWD Open Data: {station.upper()} | {len(frames)} DBZH sweeps | "
                f"{self.filter_name}"
            ),
        )

    def _select_regular_scans(
        self,
        scans: list[DWDScanRef],
        seq_length: int,
    ) -> list[DWDScanRef]:
        if not scans:
            raise RadarSourceError("DWD returned no reflectivity sweeps")
        selected = [scans[-1]]
        remaining = scans[:-1]
        step = datetime.timedelta(minutes=self.pipeline.config.time_step_minutes)
        tolerance = datetime.timedelta(
            minutes=min(4, max(2, self.pipeline.config.time_step_minutes // 3))
        )

        while remaining and len(selected) < seq_length:
            target = selected[-1].scan_time - step
            candidate = min(remaining, key=lambda scan: abs(scan.scan_time - target))
            if abs(candidate.scan_time - target) > tolerance:
                break
            selected.append(candidate)
            remaining = [scan for scan in remaining if scan.scan_time < candidate.scan_time]

        if len(selected) != seq_length:
            raise RadarSourceError(
                f"Not enough regular DWD scans: found {len(selected)}, required {seq_length}"
            )
        return sorted(selected, key=lambda scan: scan.scan_time)

    def _download_and_process(
        self,
        scans: list[DWDScanRef],
        station: str,
    ):
        frames = []
        with tempfile.TemporaryDirectory() as directory:
            for scan in scans:
                path = pathlib.Path(directory) / scan.filename
                response = self.session.get(scan.url, timeout=self.timeout_seconds)
                response.raise_for_status()
                path.write_bytes(response.content)
                try:
                    radar = self._read_odim(path)
                    frames.append(
                        self.pipeline.process_radar(
                            radar,
                            timestamp_utc=scan.scan_time,
                            station=station,
                            source="dwd-open-data",
                            provenance={
                                "url": scan.url,
                                "filename": scan.filename,
                                "format": "ODIM_H5",
                                "filter": self.filter_name,
                            },
                        )
                    )
                except RadarDecodeError:
                    raise
                except Exception as exc:
                    raise RadarDecodeError(
                        f"Failed to decode DWD ODIM HDF5 {scan.filename}: {exc}"
                    ) from exc
        return frames

    @staticmethod
    def _read_odim(path: pathlib.Path):
        import pyart

        return pyart.aux_io.read_odim_h5(str(path))

    @staticmethod
    def _station_code(value: str) -> str:
        station = value.strip().lower()
        if not re.fullmatch(r"[a-z0-9]{3}", station):
            raise ValueError("DWD station must be a three-character code")
        return station

    @staticmethod
    def _ensure_utc(value: datetime.datetime) -> datetime.datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=datetime.UTC)
        return value.astimezone(datetime.UTC)
