"""Canonical radar data contracts shared by ingestion, training and rendering."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class CanonicalGridSpec:
    """Target Cartesian grid used by new adapters."""

    width: int = 512
    height: int = 512
    resolution_m: float = 1000.0
    crs: str = "local_aeqd"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.resolution_m <= 0:
            raise ValueError("Canonical grid dimensions and resolution must be positive")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def radius_km(self) -> float:
        return max(self.width, self.height) * self.resolution_m / 2000.0

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "resolution_m": self.resolution_m,
            "radius_km": self.radius_km,
            "crs": self.crs,
        }


DEFAULT_CANONICAL_GRID = CanonicalGridSpec()


@dataclass(frozen=True)
class RadarSourceCapabilities:
    """Declare what a source can safely be used for."""

    source_id: str
    native_format: str
    quantitative_reflectivity: bool
    training_allowed: bool
    visualization_allowed: bool = True
    raw_polar_volume: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.training_allowed and not self.quantitative_reflectivity:
            raise ValueError("A training source must provide quantitative reflectivity")

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "native_format": self.native_format,
            "quantitative_reflectivity": self.quantitative_reflectivity,
            "training_allowed": self.training_allowed,
            "visualization_allowed": self.visualization_allowed,
            "raw_polar_volume": self.raw_polar_volume,
            "notes": self.notes,
        }


@dataclass
class CanonicalRadarFrame:
    """A single gridded observation with explicit validity semantics."""

    reflectivity_dbz: np.ndarray
    valid_mask: np.ndarray
    timestamp_utc: datetime.datetime
    station_id: str
    source_id: str
    product_id: str = "reflectivity"
    grid: CanonicalGridSpec = DEFAULT_CANONICAL_GRID
    coverage_mask: Optional[np.ndarray] = None
    clutter_mask: Optional[np.ndarray] = None
    interpolation_weight: Optional[np.ndarray] = None
    qc: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.reflectivity_dbz, dtype=np.float32)
        valid = np.asarray(self.valid_mask, dtype=bool)
        if values.shape != valid.shape:
            raise ValueError("reflectivity_dbz and valid_mask must have the same shape")
        if values.ndim != 2:
            raise ValueError("CanonicalRadarFrame expects a 2D grid")
        if values.shape != self.grid.shape:
            raise ValueError(f"Frame shape {values.shape} does not match grid {self.grid.shape}")

        coverage = valid.copy() if self.coverage_mask is None else np.asarray(self.coverage_mask, dtype=bool)
        clutter = np.zeros_like(valid) if self.clutter_mask is None else np.asarray(self.clutter_mask, dtype=bool)
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
                raise ValueError(f"{name} must match reflectivity shape")

        timestamp = self.timestamp_utc
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=datetime.UTC)
        else:
            timestamp = timestamp.astimezone(datetime.UTC)

        self.reflectivity_dbz = np.where(valid, values, np.nan).astype(np.float32)
        self.valid_mask = valid
        self.coverage_mask = coverage
        self.clutter_mask = clutter
        self.interpolation_weight = np.clip(weights, 0.0, 1.0)
        self.timestamp_utc = timestamp
        self.station_id = self.station_id.upper()
        self.source_id = self.source_id.lower()

    @classmethod
    def from_radar_frame(cls, frame: Any, grid: CanonicalGridSpec) -> "CanonicalRadarFrame":
        """Convert the current pipeline RadarFrame without losing provenance."""

        return cls(
            reflectivity_dbz=frame.data,
            valid_mask=frame.valid_mask,
            timestamp_utc=frame.timestamp_utc,
            station_id=frame.station,
            source_id=frame.source,
            product_id=frame.product,
            grid=grid,
            qc=dict(frame.qc),
            provenance=dict(frame.provenance),
        )

    def to_model_arrays(self, fill_value: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Return reflectivity and mask arrays suitable for model input."""

        values = np.where(self.valid_mask, self.reflectivity_dbz, fill_value).astype(np.float32)
        return values, self.valid_mask.astype(np.float32)

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "station_id": self.station_id,
            "source_id": self.source_id,
            "product_id": self.product_id,
            "grid": self.grid.to_metadata(),
            "valid_fraction": float(self.valid_mask.mean()),
            "coverage_fraction": float(self.coverage_mask.mean()),
            "qc": self.qc,
            "provenance": self.provenance,
        }
