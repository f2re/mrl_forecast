"""Shared radar preprocessing contracts for archive and operational data."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np

from config import FORECAST_STEP_MINUTES
from radar_contract import CanonicalGridSpec, CanonicalRadarFrame

PIPELINE_VERSION = "radar-grid-v2-15min"
CANONICAL_PIPELINE_VERSION = "radar-grid-v3-1km"
PRODUCT = "lowest_elevation_reflectivity"
CANONICAL_PRODUCT = "gridded_reflectivity"
UNITS = "dBZ"


class RadarError(RuntimeError):
    pass


class RadarSourceError(RadarError):
    pass


class RadarDecodeError(RadarError):
    pass


def ensure_utc(value: datetime.datetime) -> datetime.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.UTC)
    return value.astimezone(datetime.UTC)


@dataclass(frozen=True)
class RadarPipelineConfig:
    """Versioned Py-ART grid contract."""

    width: int = 256
    height: int = 256
    radius_km: float = 250.0
    vertical_limit_m: float = 10_000.0
    time_step_minutes: int = FORECAST_STEP_MINUTES
    weighting_function: str = "Barnes2"
    pipeline_version: str = PIPELINE_VERSION
    product: str = PRODUCT
    units: str = UNITS

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.radius_km <= 0:
            raise ValueError("Radar grid dimensions and radius must be positive")

    @classmethod
    def canonical(cls, time_step_minutes: int = FORECAST_STEP_MINUTES) -> "RadarPipelineConfig":
        return cls(
            width=512,
            height=512,
            radius_km=256.0,
            time_step_minutes=time_step_minutes,
            pipeline_version=CANONICAL_PIPELINE_VERSION,
            product=CANONICAL_PRODUCT,
        )

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return (1, self.height, self.width)

    @property
    def grid_limits(self) -> tuple[tuple[float, float], ...]:
        radius_m = self.radius_km * 1000.0
        return (
            (0.0, self.vertical_limit_m),
            (-radius_m, radius_m),
            (-radius_m, radius_m),
        )

    def to_grid_spec(self) -> CanonicalGridSpec:
        resolution_m = 2.0 * self.radius_km * 1000.0 / max(self.width, self.height)
        return CanonicalGridSpec(
            width=self.width,
            height=self.height,
            resolution_m=resolution_m,
        )

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "pipeline_version": self.pipeline_version,
            "product": self.product,
            "units": self.units,
            "time_step_minutes": self.time_step_minutes,
            "grid": self.to_grid_spec().to_metadata(),
            "weighting_function": self.weighting_function,
        }


@dataclass
class RadarFrame:
    """A radar grid with provenance and explicit quality masks."""

    data: np.ndarray
    valid_mask: np.ndarray
    timestamp_utc: datetime.datetime
    station: str
    source: str
    coverage_mask: Optional[np.ndarray] = None
    clutter_mask: Optional[np.ndarray] = None
    interpolation_weight: Optional[np.ndarray] = None
    product: str = PRODUCT
    status: str = "observed"
    qc: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.data, dtype=np.float32)
        valid = np.asarray(self.valid_mask, dtype=bool)
        if values.shape != valid.shape:
            raise ValueError("RadarFrame data and valid_mask must have the same shape")
        if values.ndim != 2:
            raise ValueError("RadarFrame expects one two-dimensional radar grid")

        coverage = valid.copy() if self.coverage_mask is None else np.asarray(
            self.coverage_mask,
            dtype=bool,
        )
        clutter = np.zeros_like(valid) if self.clutter_mask is None else np.asarray(
            self.clutter_mask,
            dtype=bool,
        )
        weights = valid.astype(np.float32) if self.interpolation_weight is None else np.asarray(
            self.interpolation_weight,
            dtype=np.float32,
        )
        for name, array in (
            ("coverage_mask", coverage),
            ("clutter_mask", clutter),
            ("interpolation_weight", weights),
        ):
            if array.shape != values.shape:
                raise ValueError(f"RadarFrame {name} must match data shape")

        effective_valid = valid & coverage & ~clutter & np.isfinite(weights) & (weights > 0.0)
        self.data = np.where(effective_valid, values, 0.0).astype(np.float32)
        self.valid_mask = effective_valid
        self.coverage_mask = coverage
        self.clutter_mask = clutter
        self.interpolation_weight = np.clip(np.nan_to_num(weights, nan=0.0), 0.0, 1.0)
        self.timestamp_utc = ensure_utc(self.timestamp_utc)
        self.station = self.station.upper()
        self.source = self.source.lower()

    def quality_summary(self) -> Dict[str, float]:
        covered = self.coverage_mask
        return {
            "valid_fraction": float(self.valid_mask.mean()),
            "coverage_fraction": float(covered.mean()),
            "clutter_fraction": float(self.clutter_mask.mean()),
            "mean_interpolation_weight": (
                float(self.interpolation_weight[covered].mean()) if np.any(covered) else 0.0
            ),
        }


@dataclass
class RadarSequence:
    frames: List[RadarFrame]
    source: str
    status: str = "observed"
    message: str = ""

    @property
    def timestamps(self) -> List[datetime.datetime]:
        return [frame.timestamp_utc for frame in self.frames]

    @property
    def observed_count(self) -> int:
        return sum(frame.status == "observed" for frame in self.frames)

    def stack(self, require_observed: bool = False) -> np.ndarray:
        if not self.frames:
            raise RadarSourceError("Radar sequence is empty")
        if require_observed:
            invalid = [frame.status for frame in self.frames if frame.status != "observed"]
            if invalid:
                raise RadarSourceError(
                    f"Operational sequence contains non-observed frames: {', '.join(invalid)}"
                )
        return np.stack([frame.data for frame in self.frames], axis=0)

    def quality_arrays(self) -> Dict[str, np.ndarray]:
        if not self.frames:
            raise RadarSourceError("Radar sequence is empty")
        return {
            "valid_mask": np.stack([frame.valid_mask for frame in self.frames], axis=0),
            "coverage_mask": np.stack([frame.coverage_mask for frame in self.frames], axis=0),
            "clutter_mask": np.stack([frame.clutter_mask for frame in self.frames], axis=0),
            "interpolation_weight": np.stack(
                [frame.interpolation_weight for frame in self.frames],
                axis=0,
            ),
        }

    def __iter__(self) -> Iterator[Any]:
        yield self.stack(require_observed=self.status == "observed")
        yield self.timestamps
        yield self.message


class RadarPipeline:
    """Decode and grid radar files using one versioned Py-ART path."""

    def __init__(
        self,
        config: Optional[RadarPipelineConfig] = None,
        radar_reader: Optional[Callable[[str], Any]] = None,
        grid_mapper: Optional[Callable[..., Any]] = None,
    ):
        self.config = config or RadarPipelineConfig()
        self._radar_reader = radar_reader
        self._grid_mapper = grid_mapper

    @classmethod
    def canonical(cls, **kwargs: Any) -> "RadarPipeline":
        return cls(config=RadarPipelineConfig.canonical(), **kwargs)

    def metadata(self) -> Dict[str, Any]:
        return self.config.to_metadata()

    def process_file(
        self,
        path: str,
        *,
        timestamp_utc: datetime.datetime,
        station: str,
        source: str,
    ) -> RadarFrame:
        try:
            radar = (self._radar_reader or self._default_reader)(path)
            return self.process_radar(
                radar,
                timestamp_utc=timestamp_utc,
                station=station,
                source=source,
                provenance={"path": path},
            )
        except RadarDecodeError:
            raise
        except Exception as exc:
            raise RadarDecodeError(f"Failed to decode {path}: {exc}") from exc

    def process_file_canonical(self, path: str, **kwargs: Any) -> CanonicalRadarFrame:
        return self.to_canonical(self.process_file(path, **kwargs))

    def process_radar(
        self,
        radar: Any,
        *,
        timestamp_utc: datetime.datetime,
        station: str,
        source: str,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> RadarFrame:
        field_name = self._reflectivity_field(radar)
        try:
            grid = (self._grid_mapper or self._default_mapper)(
                (radar,),
                grid_shape=self.config.grid_shape,
                grid_limits=self.config.grid_limits,
                fields=[field_name],
                weighting_function=self.config.weighting_function,
            )
            values = grid.fields[field_name]["data"][0]
        except Exception as exc:
            raise RadarDecodeError(f"Failed to grid radar reflectivity: {exc}") from exc
        return self.frame_from_grid(
            values,
            timestamp_utc=timestamp_utc,
            station=station,
            source=source,
            provenance=provenance,
        )

    def frame_from_grid(
        self,
        grid: np.ndarray,
        *,
        timestamp_utc: datetime.datetime,
        station: str,
        source: str,
        provenance: Optional[Dict[str, Any]] = None,
        status: str = "observed",
        coverage_mask: Optional[np.ndarray] = None,
        clutter_mask: Optional[np.ndarray] = None,
        interpolation_weight: Optional[np.ndarray] = None,
    ) -> RadarFrame:
        masked = np.ma.asarray(grid, dtype=np.float32)
        raw_values = np.asarray(masked.filled(np.nan), dtype=np.float32)
        source_valid = ~np.ma.getmaskarray(masked) & np.isfinite(raw_values)
        coverage = self._coverage_mask(raw_values.shape) if coverage_mask is None else np.asarray(
            coverage_mask,
            dtype=bool,
        )
        clutter = np.zeros_like(source_valid) if clutter_mask is None else np.asarray(
            clutter_mask,
            dtype=bool,
        )
        weights = source_valid.astype(np.float32) if interpolation_weight is None else np.asarray(
            interpolation_weight,
            dtype=np.float32,
        )
        for name, array in (
            ("coverage_mask", coverage),
            ("clutter_mask", clutter),
            ("interpolation_weight", weights),
        ):
            if array.shape != raw_values.shape:
                raise RadarDecodeError(f"{name} shape {array.shape} does not match grid {raw_values.shape}")

        valid_mask = source_valid & coverage & ~clutter & np.isfinite(weights) & (weights > 0.0)
        data = np.where(valid_mask, raw_values, 0.0).astype(np.float32)
        valid_values = data[valid_mask]
        covered_weights = np.clip(np.nan_to_num(weights, nan=0.0), 0.0, 1.0)
        qc = {
            "pipeline_version": self.config.pipeline_version,
            "masked_pixels": int(valid_mask.size - valid_mask.sum()),
            "valid_fraction": float(valid_mask.mean()),
            "coverage_fraction": float(coverage.mean()),
            "clutter_fraction": float(clutter.mean()),
            "mean_interpolation_weight": (
                float(covered_weights[coverage].mean()) if np.any(coverage) else 0.0
            ),
            "interpolation_weight_source": (
                "binary_validity" if interpolation_weight is None else "source"
            ),
            "min_dbz": float(valid_values.min()) if valid_values.size else None,
            "max_dbz": float(valid_values.max()) if valid_values.size else None,
            "mean_dbz": float(valid_values.mean()) if valid_values.size else None,
        }
        return RadarFrame(
            data=data,
            valid_mask=valid_mask,
            coverage_mask=coverage,
            clutter_mask=clutter,
            interpolation_weight=covered_weights,
            timestamp_utc=timestamp_utc,
            station=station,
            source=source,
            product=self.config.product,
            status=status,
            qc=qc,
            provenance=provenance or {},
        )

    def to_canonical(self, frame: RadarFrame) -> CanonicalRadarFrame:
        return CanonicalRadarFrame.from_radar_frame(frame, self.config.to_grid_spec())

    @staticmethod
    def _coverage_mask(shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        yy, xx = np.mgrid[0:height, 0:width]
        x = (xx - (width - 1) / 2.0) / max(width / 2.0, 1.0)
        y = (yy - (height - 1) / 2.0) / max(height / 2.0, 1.0)
        return (x * x + y * y) <= 1.0

    @staticmethod
    def _reflectivity_field(radar: Any) -> str:
        if "reflectivity" in radar.fields:
            return "reflectivity"
        for field_name in radar.fields:
            if "reflectivity" in field_name.lower():
                return field_name
        raise RadarDecodeError("Radar file does not contain a reflectivity field")

    @staticmethod
    def _default_reader(path: str) -> Any:
        import pyart
        return pyart.io.read(path)

    @staticmethod
    def _default_mapper(*args: Any, **kwargs: Any) -> Any:
        import pyart
        return pyart.map.grid_from_radars(*args, **kwargs)


class DemoRadarAdapter:
    """Generate explicit synthetic demo frames, never operational observations."""

    def __init__(self, grid_size: tuple[int, int] = (256, 256)):
        self.grid_size = grid_size

    def get_latest_sequence(self, seq_length: int) -> RadarSequence:
        now = datetime.datetime.now(datetime.UTC)
        frames = []
        for index in range(seq_length):
            timestamp = now - datetime.timedelta(
                minutes=(seq_length - index - 1) * FORECAST_STEP_MINUTES
            )
            frames.append(
                RadarFrame(
                    data=self._generate_grid(f"demo_{index}"),
                    valid_mask=np.ones(self.grid_size, dtype=bool),
                    coverage_mask=np.ones(self.grid_size, dtype=bool),
                    timestamp_utc=timestamp,
                    station="DEMO",
                    source="demo",
                    status="demo",
                    qc={"pipeline_version": PIPELINE_VERSION, "demo": True},
                )
            )
        return RadarSequence(
            frames=frames,
            source="demo",
            status="demo",
            message="DEMO: synthetic radar sequence",
        )

    def _generate_grid(self, seed_value: str) -> np.ndarray:
        generator = np.random.default_rng(abs(hash(seed_value)) % (2**32))
        grid = np.zeros(self.grid_size, dtype=np.float32)
        yy, xx = np.mgrid[0 : self.grid_size[0], 0 : self.grid_size[1]]
        for _ in range(3):
            px = generator.integers(0, self.grid_size[1])
            py = generator.integers(0, self.grid_size[0])
            scale_x = generator.integers(10, 60)
            scale_y = generator.integers(10, 60)
            intensity = generator.uniform(20.0, 55.0)
            blob = intensity * np.exp(
                -((xx - px) ** 2 / scale_x**2 + (yy - py) ** 2 / scale_y**2)
            )
            grid = np.maximum(grid, blob)
        return grid
