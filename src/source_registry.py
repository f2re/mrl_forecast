"""Registry for radar ingestion, discovery and downloadable sources."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict

from radar_contract import RadarSourceCapabilities
from source_access import CredentialStore, SourceProbeResult


@dataclass(frozen=True)
class RegisteredRadarSource:
    factory: Callable[..., Any]
    capabilities: RadarSourceCapabilities


class RadarSourceRegistry:
    def __init__(self, credentials: CredentialStore | None = None):
        self._sources: Dict[str, RegisteredRadarSource] = {}
        self.credentials = credentials or CredentialStore()

    def register(
        self,
        source_id: str,
        factory: Callable[..., Any],
        capabilities: RadarSourceCapabilities,
    ) -> None:
        key = source_id.strip().lower()
        if not key:
            raise ValueError("source_id must not be empty")
        if key in self._sources:
            raise ValueError(f"Radar source is already registered: {key}")
        self._sources[key] = RegisteredRadarSource(factory=factory, capabilities=capabilities)

    def source_ids(self) -> list[str]:
        return sorted(self._sources)

    def capabilities(self, source_id: str) -> RadarSourceCapabilities:
        key = source_id.strip().lower()
        try:
            return self._sources[key].capabilities
        except KeyError as exc:
            raise KeyError(f"Unknown radar source: {source_id}") from exc

    def create(self, source_id: str, **kwargs: Any) -> Any:
        key = source_id.strip().lower()
        try:
            source = self._sources[key]
        except KeyError as exc:
            raise KeyError(f"Unknown radar source: {source_id}") from exc
        if "credentials" not in kwargs and source.capabilities.credential_env:
            kwargs["credentials"] = self.credentials
        return source.factory(**kwargs)

    def describe(self) -> list[dict[str, Any]]:
        result = []
        for _, item in sorted(self._sources.items(), key=lambda pair: pair[0]):
            metadata = item.capabilities.to_metadata()
            metadata["credential_state"] = self.credentials.state(item.capabilities)
            result.append(metadata)
        return result

    def probe(self, source_id: str, **kwargs: Any) -> dict[str, Any]:
        capabilities = self.capabilities(source_id)
        credential_state = self.credentials.state(capabilities)
        if not capabilities.probe_supported:
            return SourceProbeResult(
                source_id=capabilities.source_id,
                status="probe_not_supported",
                reachable=False,
                can_list=False,
                can_download=False,
                credential_state=credential_state,
                message="This adapter does not implement a network availability probe.",
            ).to_metadata()
        try:
            source = self.create(source_id)
            probe = getattr(source, "probe", None)
            if probe is None:
                raise AttributeError("probe() is not implemented")
            result = probe(**kwargs)
            return result.to_metadata() if hasattr(result, "to_metadata") else dict(result)
        except Exception as exc:
            return SourceProbeResult(
                source_id=capabilities.source_id,
                status="unavailable",
                reachable=False,
                can_list=False,
                can_download=False,
                credential_state=credential_state,
                message=str(exc),
            ).to_metadata()

    def probe_all(
        self,
        *,
        active_only: bool = False,
        max_workers: int = 6,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        source_ids = [
            source_id
            for source_id in self.source_ids()
            if self.capabilities(source_id).probe_supported
            and (
                not active_only
                or self.capabilities(source_id).adapter_status == "active"
            )
        ]
        if not source_ids:
            return []

        reports: Dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(source_ids)))) as executor:
            futures = {
                executor.submit(self.probe, source_id, **kwargs): source_id
                for source_id in source_ids
            }
            for future in as_completed(futures):
                source_id = futures[future]
                try:
                    reports[source_id] = future.result()
                except Exception as exc:
                    reports[source_id] = SourceProbeResult(
                        source_id=source_id,
                        status="unavailable",
                        reachable=False,
                        can_list=False,
                        can_download=False,
                        credential_state=self.credentials.state(
                            self.capabilities(source_id)
                        ),
                        message=str(exc),
                    ).to_metadata()
        return [reports[source_id] for source_id in source_ids]


def build_default_source_registry(
    credentials: CredentialStore | None = None,
) -> RadarSourceRegistry:
    """Create the default registry while keeping source dependencies isolated."""

    from adapters import DemoRadarAdapter, LocalDirectoryAdapter, NOAAAWSAdapter, NOAAFTPAdapter
    from dwd_source import DWDOpenDataAdapter
    from international_sources import (
        DmiRadarSource,
        FmiS3RadarSource,
        KnmiRadarSource,
        MANUAL_SOURCE_PROFILES,
        ManualAccessSource,
        OperaOrdRadarSource,
        Wis2GlobalCacheSource,
    )
    from open_sources import MeteoinfoVisualSource, RainViewerMetadataSource, Wis2RadarCatalog
    from radar_pipeline import RadarPipeline

    credential_store = credentials or CredentialStore()

    def canonical_noaa_aws(**kwargs: Any):
        kwargs.setdefault("pipeline", RadarPipeline.canonical())
        return NOAAAWSAdapter(**kwargs)

    registry = RadarSourceRegistry(credentials=credential_store)
    registry.register(
        "noaa-aws",
        NOAAAWSAdapter,
        RadarSourceCapabilities(
            source_id="noaa-aws",
            native_format="NEXRAD Level II / legacy grid",
            quantitative_reflectivity=True,
            raw_polar_volume=True,
            training_allowed=True,
            notes="Reference quantitative source using the legacy 256x256 pipeline.",
            access_mode="open",
            download_supported=True,
            license_id="US public data",
        ),
    )
    registry.register(
        "noaa-aws-canonical",
        canonical_noaa_aws,
        RadarSourceCapabilities(
            source_id="noaa-aws-canonical",
            native_format="NEXRAD Level II / canonical 512x512 grid",
            quantitative_reflectivity=True,
            raw_polar_volume=True,
            training_allowed=True,
            notes="Quantitative 1 km canonical adapter for new datasets.",
            access_mode="open",
            download_supported=True,
            license_id="US public data",
        ),
    )
    registry.register("dwd-open-data", DWDOpenDataAdapter, DWDOpenDataAdapter.CAPABILITIES)
    registry.register("fmi-s3", FmiS3RadarSource, FmiS3RadarSource.CAPABILITIES)
    registry.register("opera-ord", OperaOrdRadarSource, OperaOrdRadarSource.CAPABILITIES)
    registry.register("dmi-radar", DmiRadarSource, DmiRadarSource.CAPABILITIES)
    registry.register("knmi-radar", KnmiRadarSource, KnmiRadarSource.CAPABILITIES)
    registry.register("wis2-cache", Wis2GlobalCacheSource, Wis2GlobalCacheSource.CAPABILITIES)
    registry.register(
        "noaa-ftp",
        NOAAFTPAdapter,
        RadarSourceCapabilities(
            source_id="noaa-ftp",
            native_format="NEXRAD Level III",
            quantitative_reflectivity=True,
            training_allowed=False,
            notes="Visualization/reference only until Level III gridding is unified.",
            access_mode="open",
        ),
    )
    registry.register(
        "local",
        LocalDirectoryAdapter,
        RadarSourceCapabilities(
            source_id="local",
            native_format="BUFR/NPY/NPZ",
            quantitative_reflectivity=True,
            training_allowed=False,
            notes="Training permission is decided from file provenance, not the folder type.",
            access_mode="open",
        ),
    )
    registry.register(
        "demo",
        DemoRadarAdapter,
        RadarSourceCapabilities(
            source_id="demo",
            native_format="synthetic",
            quantitative_reflectivity=False,
            training_allowed=False,
            notes="UI checks only.",
            access_mode="open",
        ),
    )
    registry.register("wis2", Wis2RadarCatalog, Wis2RadarCatalog.CAPABILITIES)
    registry.register("meteoinfo", MeteoinfoVisualSource, MeteoinfoVisualSource.CAPABILITIES)
    registry.register("rainviewer", RainViewerMetadataSource, RainViewerMetadataSource.CAPABILITIES)

    for source_id, (capabilities, landing_url) in MANUAL_SOURCE_PROFILES.items():
        registry.register(
            source_id,
            lambda capabilities=capabilities, landing_url=landing_url, **kwargs: ManualAccessSource(
                capabilities,
                landing_url,
                credentials=kwargs.pop("credentials", credential_store),
                **kwargs,
            ),
            capabilities,
        )
    return registry
