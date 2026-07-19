"""Downloadable international radar sources and access profiles."""

from __future__ import annotations

import datetime
import os
import pathlib
from typing import Any, Dict, Iterable, Optional
from urllib.parse import quote

import requests

from radar_contract import RadarSourceCapabilities
from source_access import (
    CredentialStore,
    RemoteRadarFile,
    SourceProbeResult,
    download_http,
    download_s3_file,
    list_s3_files,
    parse_utc,
    verify_http_download,
    verify_s3_download,
)


class FmiS3RadarSource:
    """Open FMI ODIM 2.3 single-radar volumes on AWS S3."""

    BUCKET = "fmi-opendata-radar-volume-hdf5"
    REGION = "eu-west-1"
    CAPABILITIES = RadarSourceCapabilities(
        source_id="fmi-s3",
        native_format="ODIM 2.3 HDF5 volumes",
        quantitative_reflectivity=True,
        training_allowed=False,
        visualization_allowed=True,
        raw_polar_volume=True,
        notes="Open FMI single-radar volumes. Enable training only after field and station QC.",
        access_mode="open",
        probe_supported=True,
        download_supported=True,
        adapter_status="active",
        license_id="CC-BY-4.0",
        archive_note="Operational volume stream, normally updated every five minutes.",
    )

    def list_files(self, prefix: str = "", limit: int = 20, **_: Any) -> list[RemoteRadarFile]:
        return list_s3_files(
            source_id=self.CAPABILITIES.source_id,
            bucket=self.BUCKET,
            region=self.REGION,
            prefix=prefix,
            limit=limit,
            native_format="ODIM_H5",
        )

    def probe(self, download_test: bool = False, prefix: str = "", **_: Any) -> SourceProbeResult:
        try:
            files = self.list_files(prefix=prefix, limit=1)
            sample = files[0] if files else None
            can_download = bool(sample) and (
                verify_s3_download(sample) if download_test else True
            )
            return SourceProbeResult(
                source_id=self.CAPABILITIES.source_id,
                status="available" if sample else "degraded",
                reachable=True,
                can_list=True,
                can_download=can_download,
                credential_state="not_required",
                message="FMI public S3 volume bucket is reachable.",
                sample=sample.to_metadata() if sample else None,
            )
        except Exception as exc:
            return _failed_probe(self.CAPABILITIES, exc)

    @staticmethod
    def download(remote: RemoteRadarFile, output_dir: str) -> Dict[str, Any]:
        return download_s3_file(remote, output_dir)


class Wis2GlobalCacheSource:
    """Open 24-hour WIS2 core-data cache on AWS."""

    BUCKET = "wis2globalcache"
    REGION = "us-east-1"
    CAPABILITIES = RadarSourceCapabilities(
        source_id="wis2-cache",
        native_format="BUFR/GRIB/NetCDF WIS2 core data",
        quantitative_reflectivity=False,
        training_allowed=False,
        visualization_allowed=False,
        raw_polar_volume=False,
        notes="Generic 24-hour cache. Radar files require discovery metadata and format verification.",
        access_mode="open",
        probe_supported=True,
        download_supported=True,
        adapter_status="active",
        archive_note="Rolling 24-hour cache; use WIS2 notifications for operational access.",
    )

    def list_files(self, prefix: str = "", limit: int = 20, **_: Any) -> list[RemoteRadarFile]:
        return list_s3_files(
            source_id=self.CAPABILITIES.source_id,
            bucket=self.BUCKET,
            region=self.REGION,
            prefix=prefix,
            limit=limit,
            native_format="WIS2",
        )

    def probe(self, download_test: bool = False, prefix: str = "", **_: Any) -> SourceProbeResult:
        try:
            files = self.list_files(prefix=prefix, limit=1)
            sample = files[0] if files else None
            can_download = bool(sample) and (
                verify_s3_download(sample) if download_test else True
            )
            return SourceProbeResult(
                source_id=self.CAPABILITIES.source_id,
                status="available" if sample else "degraded",
                reachable=True,
                can_list=True,
                can_download=can_download,
                credential_state="not_required",
                message="WIS2 global cache is reachable; files are not assumed to be radar data.",
                sample=sample.to_metadata() if sample else None,
            )
        except Exception as exc:
            return _failed_probe(self.CAPABILITIES, exc)

    @staticmethod
    def download(remote: RemoteRadarFile, output_dir: str) -> Dict[str, Any]:
        return download_s3_file(remote, output_dir)


class OperaOrdRadarSource:
    """EUMETNET OPERA ORD API plus its anonymous 24-hour S3 cache."""

    BASE_URL = "https://api.meteogate.eu/eu-eumetnet-weather-radar"
    CACHE_BUCKET = "openradar-24h"
    CACHE_REGION = "waw3-1"
    CACHE_ENDPOINT = "https://s3.waw3-1.cloudferro.com"
    CAPABILITIES = RadarSourceCapabilities(
        source_id="opera-ord",
        native_format="ODIM HDF5/BUFR single-site volumes and composites",
        quantitative_reflectivity=True,
        training_allowed=False,
        visualization_allowed=True,
        raw_polar_volume=True,
        notes="Anonymous ORD access is supported; verify per-file licence, field and scan metadata before training.",
        access_mode="open_optional_key",
        credential_env="METEOGATE_API_KEY",
        registration_url="https://devportal.meteogate.eu/",
        registration_steps=(
            "Open the MeteoGate Developer Portal and sign in with an available identity provider.",
            "Complete the user profile if requested.",
            "Choose Create API Key and copy the key when it is shown.",
            "Run: python mrl.py sources --action configure --source opera-ord",
            "Set METEOGATE_API_KEY_HEADER only if the selected MeteoGate route documents a specific header name.",
        ),
        probe_supported=True,
        download_supported=True,
        adapter_status="active",
        license_id="CC-BY-4.0 with provider exceptions",
        archive_note="Single-site volumes currently have a rolling 24-hour cache; composites have a longer archive.",
    )

    def __init__(
        self,
        timeout_seconds: int = 30,
        session: Optional[requests.Session] = None,
        credentials: Optional[CredentialStore] = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self.credentials = credentials or CredentialStore()

    def _headers(self) -> Dict[str, str]:
        token = self.credentials.get(self.CAPABILITIES.credential_env)
        header_name = os.environ.get("METEOGATE_API_KEY_HEADER")
        return {header_name: token} if token and header_name else {}

    def list_files(self, prefix: str = "", limit: int = 20, **_: Any) -> list[RemoteRadarFile]:
        return list_s3_files(
            source_id=self.CAPABILITIES.source_id,
            bucket=self.CACHE_BUCKET,
            region=self.CACHE_REGION,
            endpoint_url=self.CACHE_ENDPOINT,
            public_base_url=f"{self.CACHE_ENDPOINT}/{self.CACHE_BUCKET}",
            prefix=prefix,
            limit=limit,
            native_format="ODIM_H5_or_BUFR",
        )

    def query(
        self,
        *,
        location_id: str,
        start: datetime.datetime,
        end: datetime.datetime,
        standard_name: str = "DBZH",
        level: str = "",
        method: str = "scan",
    ) -> list[RemoteRadarFile]:
        start_text = _utc_text(start)
        end_text = _utc_text(end)
        params = {
            "datetime": f"{start_text}/{end_text}",
            "f": "CoverageJSON",
            "standard_name": standard_name,
            "format": "ODIM",
            "method": method,
        }
        if level:
            params["level"] = level
        response = self.session.get(
            f"{self.BASE_URL}/collections/observations/locations/{quote(location_id, safe='*-')}",
            params=params,
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        if response.status_code == 204:
            return []
        response.raise_for_status()
        return self._files_from_payload(response.json(), location_id, standard_name)

    def _files_from_payload(
        self,
        payload: Any,
        location_id: str,
        standard_name: str,
    ) -> list[RemoteRadarFile]:
        files: Dict[str, RemoteRadarFile] = {}
        for link in _walk_links(payload):
            url = str(link.get("href", ""))
            filename = pathlib.PurePosixPath(url.split("?", 1)[0]).name
            if not filename or not _looks_like_radar_file(filename):
                continue
            files[url] = RemoteRadarFile(
                source_id=self.CAPABILITIES.source_id,
                file_id=url,
                filename=filename,
                url=url,
                native_format="ODIM_H5_or_BUFR",
                station_id=location_id,
                metadata={"standard_name": standard_name, "link": link},
            )
        return list(files.values())

    def probe(self, download_test: bool = False, prefix: str = "", **_: Any) -> SourceProbeResult:
        credential_state = self.credentials.state(self.CAPABILITIES)
        try:
            response = self.session.get(
                self.BASE_URL,
                headers=self._headers(),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            files = self.list_files(prefix=prefix, limit=1)
            sample = files[0] if files else None
            can_download = bool(sample) and (
                verify_s3_download(sample) if download_test else True
            )
            return SourceProbeResult(
                source_id=self.CAPABILITIES.source_id,
                status="available" if sample else "degraded",
                reachable=True,
                can_list=True,
                can_download=can_download,
                credential_state=credential_state,
                message="ORD API is reachable in anonymous mode; an API key is optional for higher limits.",
                sample=sample.to_metadata() if sample else None,
            )
        except Exception as exc:
            return _failed_probe(self.CAPABILITIES, exc, credential_state)

    @staticmethod
    def download(remote: RemoteRadarFile, output_dir: str) -> Dict[str, Any]:
        if remote.metadata.get("bucket"):
            return download_s3_file(remote, output_dir)
        return download_http(remote, output_dir)


class DmiRadarSource:
    """DMI STAC API for full radar volumes, pseudo-CAPPI and composites."""

    BASE_URL = "https://opendataapi.dmi.dk/v1/radardata"
    CAPABILITIES = RadarSourceCapabilities(
        source_id="dmi-radar",
        native_format="ODIM HDF5 volumes/composites",
        quantitative_reflectivity=True,
        training_allowed=False,
        visualization_allowed=True,
        raw_polar_volume=True,
        notes="Open DMI radar STAC API. Validate scan type and ODIM quantities before training.",
        access_mode="open",
        probe_supported=True,
        download_supported=True,
        adapter_status="active",
        archive_note="Volume, pseudoCappi and composite collections; current API is unauthenticated.",
    )

    def __init__(self, timeout_seconds: int = 30, session: Optional[requests.Session] = None):
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def list_files(
        self,
        *,
        collection: str = "volume",
        station: str = "",
        start: Optional[datetime.datetime] = None,
        end: Optional[datetime.datetime] = None,
        limit: int = 20,
        **_: Any,
    ) -> list[RemoteRadarFile]:
        if collection not in {"volume", "pseudoCappi", "composite"}:
            raise ValueError("DMI collection must be volume, pseudoCappi or composite")
        params: Dict[str, Any] = {
            "limit": max(1, min(int(limit), 1000)),
            "sortorder": "datetime,DESC",
        }
        if station and collection != "composite":
            params["stationId"] = station
        if start or end:
            params["datetime"] = f"{_utc_text(start) if start else '..'}/{_utc_text(end) if end else '..'}"
        response = self.session.get(
            f"{self.BASE_URL}/collections/{collection}/items",
            params=params,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        files = []
        for feature in response.json().get("features", []):
            asset = feature.get("asset") or feature.get("assets") or {}
            data = asset.get("data", asset) if isinstance(asset, dict) else {}
            url = str(data.get("href", ""))
            filename = str(feature.get("id") or pathlib.PurePosixPath(url).name)
            if not url or not filename:
                continue
            properties = feature.get("properties", {})
            files.append(
                RemoteRadarFile(
                    source_id=self.CAPABILITIES.source_id,
                    file_id=filename,
                    filename=filename,
                    url=url,
                    native_format="ODIM_H5",
                    timestamp_utc=parse_utc(properties.get("datetime")),
                    station_id=str(properties.get("stationId") or station),
                    metadata={"collection": collection, "properties": properties},
                )
            )
        return files

    def probe(self, download_test: bool = False, **kwargs: Any) -> SourceProbeResult:
        try:
            response = self.session.get(
                f"{self.BASE_URL}/collections",
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            files = self.list_files(limit=1, **kwargs)
            sample = files[0] if files else None
            can_download = bool(sample) and (
                verify_http_download(sample, session=self.session) if download_test else True
            )
            return SourceProbeResult(
                source_id=self.CAPABILITIES.source_id,
                status="available" if sample else "degraded",
                reachable=True,
                can_list=True,
                can_download=can_download,
                credential_state="not_required",
                message="DMI radar API is reachable and no API key is required.",
                sample=sample.to_metadata() if sample else None,
            )
        except Exception as exc:
            return _failed_probe(self.CAPABILITIES, exc)

    def download(self, remote: RemoteRadarFile, output_dir: str) -> Dict[str, Any]:
        return download_http(remote, output_dir, session=self.session)


class KnmiRadarSource:
    """KNMI file API for current and archived radar datasets."""

    BASE_URL = "https://api.dataplatform.knmi.nl/open-data/v1"
    DEFAULT_DATASET = "radar_volume_full_herwijnen"
    DEFAULT_VERSION = "1.0"
    CAPABILITIES = RadarSourceCapabilities(
        source_id="knmi-radar",
        native_format="HDF5 polarimetric volumes and composites",
        quantitative_reflectivity=True,
        training_allowed=False,
        visualization_allowed=True,
        raw_polar_volume=True,
        notes="KNMI Open Data API requires an API key. Dataset and version are configurable.",
        access_mode="api_key_required",
        credential_env="KNMI_API_KEY",
        registration_url="https://developer.dataplatform.knmi.nl/open-data-api",
        registration_steps=(
            "Send an email to opendata@knmi.nl with your name, organisation and reason for access if portal sign-up is unavailable.",
            "After the account is enabled, sign in to the KNMI Developer Portal.",
            "Open API Catalog, select Open Data API and choose Request an API key.",
            "Copy the key when displayed; it may not be shown again.",
            "Run: python mrl.py sources --action configure --source knmi-radar",
            "For a complete dataset, request a bulk key from opendata@knmi.nl and state the dataset name/version.",
        ),
        probe_supported=True,
        download_supported=True,
        adapter_status="active",
        license_id="CC-BY-4.0",
        archive_note="Current 5-minute files and daily TAR archives are separate KNMI datasets.",
    )

    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET,
        dataset_version: str = DEFAULT_VERSION,
        timeout_seconds: int = 30,
        session: Optional[requests.Session] = None,
        credentials: Optional[CredentialStore] = None,
    ):
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self.credentials = credentials or CredentialStore()

    def _headers(self) -> Dict[str, str]:
        token = self.credentials.get(self.CAPABILITIES.credential_env)
        if not token:
            raise SourceAccessError(
                "KNMI_API_KEY is missing. Run: python mrl.py sources --action configure --source knmi-radar"
            )
        return {"Authorization": token}

    def list_files(self, limit: int = 20, **_: Any) -> list[RemoteRadarFile]:
        response = self.session.get(
            f"{self.BASE_URL}/datasets/{self.dataset_name}/versions/{self.dataset_version}/files",
            params={
                "maxKeys": max(1, min(int(limit), 1000)),
                "orderBy": "created",
                "sorting": "desc",
            },
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        result = []
        for item in response.json().get("files", []):
            filename = str(item.get("filename", ""))
            if not filename:
                continue
            api_url = (
                f"{self.BASE_URL}/datasets/{self.dataset_name}/versions/"
                f"{self.dataset_version}/files/{quote(filename, safe='')}/url"
            )
            result.append(
                RemoteRadarFile(
                    source_id=self.CAPABILITIES.source_id,
                    file_id=filename,
                    filename=filename,
                    url=api_url,
                    native_format="HDF5_or_TAR",
                    timestamp_utc=parse_utc(item.get("created") or item.get("lastModified")),
                    size_bytes=item.get("size"),
                    metadata={
                        "dataset_name": self.dataset_name,
                        "dataset_version": self.dataset_version,
                        "file": item,
                    },
                )
            )
        return result

    def resolve_download(self, remote: RemoteRadarFile) -> RemoteRadarFile:
        response = self.session.get(
            remote.url,
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        download_url = response.json().get("temporaryDownloadUrl")
        if not download_url:
            raise SourceAccessError(f"KNMI did not return a download URL for {remote.filename}")
        return RemoteRadarFile(
            source_id=remote.source_id,
            file_id=remote.file_id,
            filename=remote.filename,
            url=str(download_url),
            native_format=remote.native_format,
            timestamp_utc=remote.timestamp_utc,
            station_id=remote.station_id,
            size_bytes=remote.size_bytes,
            metadata=remote.metadata,
        )

    def probe(self, download_test: bool = False, **_: Any) -> SourceProbeResult:
        credential_state = self.credentials.state(self.CAPABILITIES)
        if credential_state == "missing":
            return SourceProbeResult(
                source_id=self.CAPABILITIES.source_id,
                status="credential_required",
                reachable=False,
                can_list=False,
                can_download=False,
                credential_state=credential_state,
                message="KNMI API key is not configured.",
            )
        try:
            files = self.list_files(limit=1)
            sample = files[0] if files else None
            can_download = False
            sample_metadata = sample.to_metadata() if sample else None
            if sample:
                resolved = self.resolve_download(sample)
                can_download = (
                    verify_http_download(resolved, session=self.session)
                    if download_test
                    else True
                )
                sample_metadata = resolved.to_metadata()
            return SourceProbeResult(
                source_id=self.CAPABILITIES.source_id,
                status="available" if sample else "degraded",
                reachable=True,
                can_list=True,
                can_download=can_download,
                credential_state=credential_state,
                message=f"KNMI dataset {self.dataset_name}/{self.dataset_version} is reachable.",
                sample=sample_metadata,
            )
        except Exception as exc:
            return _failed_probe(self.CAPABILITIES, exc, credential_state)

    def download(self, remote: RemoteRadarFile, output_dir: str) -> Dict[str, Any]:
        resolved = self.resolve_download(remote)
        return download_http(resolved, output_dir, session=self.session)


class ManualAccessSource:
    """Represent a documented source whose automated downloader is not ready."""

    def __init__(
        self,
        capabilities: RadarSourceCapabilities,
        landing_url: str,
        timeout_seconds: int = 20,
        session: Optional[requests.Session] = None,
        credentials: Optional[CredentialStore] = None,
    ):
        self.CAPABILITIES = capabilities
        self.landing_url = landing_url
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self.credentials = credentials or CredentialStore()

    def probe(self, **_: Any) -> SourceProbeResult:
        credential_state = self.credentials.state(self.CAPABILITIES)
        try:
            response = self.session.get(self.landing_url, timeout=self.timeout_seconds)
            response.raise_for_status()
            return SourceProbeResult(
                source_id=self.CAPABILITIES.source_id,
                status=(
                    "credential_required"
                    if self.CAPABILITIES.credential_required and credential_state == "missing"
                    else "manual_registration"
                ),
                reachable=True,
                can_list=False,
                can_download=False,
                credential_state=credential_state,
                message="Landing or registration page is reachable; automated file download is not implemented.",
            )
        except Exception as exc:
            return _failed_probe(self.CAPABILITIES, exc, credential_state)


MANUAL_SOURCE_PROFILES = {
    "meteofrance-radar": (
        RadarSourceCapabilities(
            source_id="meteofrance-radar",
            native_format="Météo-France Package Radar",
            quantitative_reflectivity=True,
            training_allowed=False,
            access_mode="account_required",
            credential_env="METEOFRANCE_API_TOKEN",
            registration_url="https://portail-api.meteofrance.fr/",
            registration_steps=(
                "Create or sign in to an account on the Météo-France API portal.",
                "Subscribe to Données Publiques Paquet Radar.",
                "Create/copy the API token shown by the portal.",
                "Run: python mrl.py sources --action configure --source meteofrance-radar",
            ),
            probe_supported=True,
            download_supported=False,
            adapter_status="probe_only",
            license_id="Licence Ouverte 2.0",
            notes="Account access is documented; endpoint-specific downloader remains to be implemented.",
        ),
        "https://www.data.gouv.fr/dataservices/api-package-radar",
    ),
    "ceda-nimrod": (
        RadarSourceCapabilities(
            source_id="ceda-nimrod",
            native_format="NIMROD/polar radar collections",
            quantitative_reflectivity=True,
            training_allowed=False,
            access_mode="account_required",
            registration_url="https://services.ceda.ac.uk/cedasite/register/info/",
            registration_steps=(
                "Create a CEDA account.",
                "Open the required NIMROD dataset record and apply for access if requested.",
                "Accept the dataset terms and use the CEDA download mechanism documented for the collection.",
            ),
            probe_supported=True,
            download_supported=False,
            adapter_status="manual",
            notes="CEDA account and dataset permissions are handled outside MRL Forecast.",
        ),
        "https://catalogue.ceda.ac.uk/uuid/82adec1f896af6169112d09cc1174499/",
    ),
    "aura-nci": (
        RadarSourceCapabilities(
            source_id="aura-nci",
            native_format="ODIM_H5/CfRadial/NetCDF",
            quantitative_reflectivity=True,
            training_allowed=False,
            access_mode="request_required",
            registration_url="https://opus.nci.org.au/spaces/NDP/pages/399803502/Australian+Unified+Radar+Archive",
            registration_steps=(
                "Create an NCI account.",
                "Request membership in a project that grants access to the required AURA collection.",
                "Follow the NCI data collection instructions for Level 0/1/1b/2 products.",
            ),
            probe_supported=True,
            download_supported=False,
            adapter_status="manual",
            notes="NCI project access is required for AURA datasets.",
        ),
        "https://opus.nci.org.au/spaces/NDP/pages/399803502/Australian+Unified+Radar+Archive",
    ),
    "ncradar-cao": (
        RadarSourceCapabilities(
            source_id="ncradar-cao",
            native_format="NCRadar NetCDF / possible FM 94 BUFR export",
            quantitative_reflectivity=True,
            training_allowed=False,
            access_mode="request_required",
            registration_url="https://www.cao-rhms.ru/ncradar/ncradar.pdf",
            registration_steps=(
                "Prepare a formal request to the Central Aerological Observatory (ЦАО).",
                "Specify radar/WIGOS IDs, date range, complete volumes or elevation scans and required quantities.",
                "Request DBZH/TH/VRADH/WRADH and dual-pol fields with QC masks where available.",
                "Ask for NetCDF/CfRadial, ODIM_H5 or FM 94 BUFR and an explicit data-use agreement.",
            ),
            probe_supported=True,
            download_supported=False,
            adapter_status="manual",
            notes="No verified anonymous file catalogue or API is currently known.",
        ),
        "https://www.cao-rhms.ru/ncradar/ncradar.pdf",
    ),
}


def _failed_probe(
    capabilities: RadarSourceCapabilities,
    error: Exception,
    credential_state: str = "not_required",
) -> SourceProbeResult:
    return SourceProbeResult(
        source_id=capabilities.source_id,
        status="unavailable",
        reachable=False,
        can_list=False,
        can_download=False,
        credential_state=credential_state,
        message=str(error),
    )


def _utc_text(value: datetime.datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=datetime.UTC)
    return value.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _looks_like_radar_file(filename: str) -> bool:
    lowered = filename.lower()
    return lowered.endswith((".h5", ".hdf5", ".bufr", ".tif", ".tiff", ".nc"))


def _walk_links(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, dict):
        if payload.get("href"):
            yield payload
        for value in payload.values():
            yield from _walk_links(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _walk_links(value)
