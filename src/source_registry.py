"""Small registry for radar data and discovery sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from radar_contract import RadarSourceCapabilities


@dataclass(frozen=True)
class RegisteredRadarSource:
    factory: Callable[..., Any]
    capabilities: RadarSourceCapabilities


class RadarSourceRegistry:
    def __init__(self):
        self._sources: Dict[str, RegisteredRadarSource] = {}

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

    def create(self, source_id: str, **kwargs: Any) -> Any:
        key = source_id.strip().lower()
        try:
            source = self._sources[key]
        except KeyError as exc:
            raise KeyError(f"Unknown radar source: {source_id}") from exc
        return source.factory(**kwargs)

    def describe(self) -> list[dict[str, Any]]:
        return [
            item.capabilities.to_metadata()
            for _, item in sorted(self._sources.items(), key=lambda pair: pair[0])
        ]


def build_default_source_registry() -> RadarSourceRegistry:
    """Create the default registry while keeping optional sources isolated."""

    from adapters import DemoRadarAdapter, LocalDirectoryAdapter, NOAAAWSAdapter, NOAAFTPAdapter
    from open_sources import MeteoinfoVisualSource, RainViewerMetadataSource, Wis2RadarCatalog
    from radar_pipeline import RadarPipeline

    def canonical_noaa_aws(**kwargs: Any):
        kwargs.setdefault("pipeline", RadarPipeline.canonical())
        return NOAAAWSAdapter(**kwargs)

    registry = RadarSourceRegistry()
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
        ),
    )
    registry.register(
        "noaa-ftp",
        NOAAFTPAdapter,
        RadarSourceCapabilities(
            source_id="noaa-ftp",
            native_format="NEXRAD Level III",
            quantitative_reflectivity=True,
            training_allowed=False,
            notes="Visualization/reference only until Level III gridding is unified.",
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
        ),
    )
    registry.register("wis2", Wis2RadarCatalog, Wis2RadarCatalog.CAPABILITIES)
    registry.register("meteoinfo", MeteoinfoVisualSource, MeteoinfoVisualSource.CAPABILITIES)
    registry.register("rainviewer", RainViewerMetadataSource, RainViewerMetadataSource.CAPABILITIES)
    return registry
