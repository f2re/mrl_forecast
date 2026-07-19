"""Shared CPU/GPU inference runtime for registered radar models."""

from __future__ import annotations

import pathlib
from typing import Any, Dict, Optional

import numpy as np
import torch

from convlstm import ConvLSTM
from phys_evolution import MRLPhysEvolution
from radar_pipeline import (
    CANONICAL_PIPELINE_VERSION,
    PIPELINE_VERSION,
    RadarPipeline,
    RadarPipelineConfig,
)

MAX_DBZ = 70.0
SUPPORTED_PIPELINES = {PIPELINE_VERSION, CANONICAL_PIPELINE_VERSION}


class ModelRuntime:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.info: Dict[str, Any] = {}

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def input_length(self) -> int:
        return int(self.info.get("input_length", 4))

    @property
    def target_length(self) -> int:
        return int(self.info.get("target_length", 4))

    @property
    def forecast_step_minutes(self) -> int:
        return int(self.info.get("forecast_step_minutes", 15))

    @property
    def grid(self) -> Dict[str, Any]:
        return dict(self.info.get("grid", {}))

    @property
    def architecture(self) -> str:
        return str(self.info.get("model_architecture", "unknown"))

    def pipeline(self) -> RadarPipeline:
        """Return a gridding pipeline aligned with both model grid and cadence."""

        if self.info.get("pipeline_version") == CANONICAL_PIPELINE_VERSION:
            config = RadarPipelineConfig.canonical(
                time_step_minutes=self.forecast_step_minutes,
            )
        else:
            config = RadarPipelineConfig(
                time_step_minutes=self.forecast_step_minutes,
            )
        return RadarPipeline(config=config)

    def load(self, checkpoint_path: str) -> Dict[str, Any]:
        path = pathlib.Path(checkpoint_path)
        checkpoint = self._safe_load(path)
        pipeline_version = checkpoint.get("pipeline_version", PIPELINE_VERSION)
        if pipeline_version not in SUPPORTED_PIPELINES:
            raise ValueError(f"Unsupported model pipeline: {pipeline_version}")

        architecture = checkpoint.get("model_architecture", "convlstm_baseline")
        hyperparameters = checkpoint.get("hyperparameters", {})
        input_length = int(checkpoint.get("input_length", hyperparameters.get("input_length", 4)))
        target_length = int(checkpoint.get("target_length", hyperparameters.get("target_length", 4)))
        model_config = dict(checkpoint.get("model_config") or {})

        if architecture == "phys-evolution":
            model_config.setdefault("output_steps", target_length)
            loaded_model = MRLPhysEvolution(**model_config)
        else:
            if not model_config:
                hidden = checkpoint.get("hidden_channels", [32, 32])
                model_config = {
                    "input_channels": 1,
                    "hidden_channels": hidden,
                    "kernel_size": (3, 3),
                    "output_steps": target_length,
                }
            loaded_model = ConvLSTM(**model_config)

        loaded_model.load_state_dict(checkpoint["model_state_dict"])
        loaded_model.to(self.device)
        loaded_model.eval()
        self.model = loaded_model
        self.info = {
            "path": str(path),
            "model_id": path.parent.name,
            "model_architecture": architecture,
            "model_config": model_config,
            "pipeline_version": pipeline_version,
            "forecast_step_minutes": int(checkpoint.get("forecast_step_minutes", 15)),
            "grid": dict(checkpoint.get("grid") or self._default_grid(pipeline_version)),
            "input_length": input_length,
            "target_length": target_length,
            "metrics": checkpoint.get("metrics", {}),
            "quality_gate_metrics": checkpoint.get("quality_gate_metrics", {}),
        }
        return dict(self.info)

    def predict(
        self,
        reflectivity_dbz: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        coverage_mask: Optional[np.ndarray] = None,
        clutter_mask: Optional[np.ndarray] = None,
        interpolation_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("No model is loaded")
        values, effective_mask, quality = self._prepare_arrays(
            reflectivity_dbz,
            valid_mask,
            coverage_mask,
            clutter_mask,
            interpolation_weight,
        )
        x = torch.from_numpy(values).unsqueeze(0).unsqueeze(2).to(self.device)
        x_mask = torch.from_numpy(effective_mask).unsqueeze(0).unsqueeze(2).to(self.device)

        with torch.no_grad():
            if self.architecture == "phys-evolution":
                prediction, diagnostics = self.model(x, x_mask)
            else:
                prediction, states = self.model(x)
                diagnostics = {"states": states}

        forecast = prediction[0, :, 0].cpu().numpy()
        forecast_quality = {
            name: np.repeat(array[-1:, ...], self.target_length, axis=0)
            for name, array in quality.items()
        }
        result = {
            "input": x[0, :, 0].cpu().numpy(),
            "input_mask": x_mask[0, :, 0].cpu().numpy().astype(bool),
            "forecast": forecast,
            "diagnostics": self._diagnostics_to_numpy(diagnostics),
            "quality_masks": quality,
            "forecast_quality_masks": forecast_quality,
        }
        return result

    def _prepare_arrays(
        self,
        reflectivity_dbz: np.ndarray,
        valid_mask: Optional[np.ndarray],
        coverage_mask: Optional[np.ndarray],
        clutter_mask: Optional[np.ndarray],
        interpolation_weight: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        values = np.asarray(reflectivity_dbz, dtype=np.float32)
        if values.ndim == 4 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        if values.ndim != 3:
            raise ValueError(f"Expected [T,H,W], got {values.shape}")
        if values.shape[0] < self.input_length:
            raise ValueError(
                f"Insufficient history: required {self.input_length}, received {values.shape[0]}"
            )
        values = values[-self.input_length:]

        expected = self.expected_grid_shape()
        if expected and values.shape[-2:] != expected:
            raise ValueError(f"Source grid {values.shape[-2:]} is incompatible with model grid {expected}")

        valid = self._quality_array(
            "valid_mask",
            valid_mask,
            np.isfinite(values),
            values.shape,
            bool,
        )
        coverage = self._quality_array(
            "coverage_mask",
            coverage_mask,
            np.ones_like(valid),
            values.shape,
            bool,
        )
        clutter = self._quality_array(
            "clutter_mask",
            clutter_mask,
            np.zeros_like(valid),
            values.shape,
            bool,
        )
        weights = self._quality_array(
            "interpolation_weight",
            interpolation_weight,
            valid.astype(np.float32),
            values.shape,
            np.float32,
        )
        weights = np.clip(np.nan_to_num(weights, nan=0.0), 0.0, 1.0)
        effective = valid & coverage & ~clutter & (weights > 0.0)
        normalized = np.clip(np.where(effective, values, 0.0), 0.0, MAX_DBZ) / MAX_DBZ
        quality = {
            "valid_mask": effective.astype(bool),
            "coverage_mask": coverage.astype(bool),
            "clutter_mask": clutter.astype(bool),
            "interpolation_weight": weights.astype(np.float32),
        }
        return normalized.astype(np.float32), effective.astype(np.float32), quality

    def _quality_array(
        self,
        name: str,
        value: Optional[np.ndarray],
        default: np.ndarray,
        expected_shape: tuple[int, int, int],
        dtype,
    ) -> np.ndarray:
        array = np.asarray(default if value is None else value, dtype=dtype)
        if array.ndim == 4 and array.shape[-1] == 1:
            array = array.squeeze(-1)
        if array.shape[0] >= self.input_length:
            array = array[-self.input_length:]
        if array.shape != expected_shape:
            raise ValueError(f"{name} shape {array.shape} does not match selected history {expected_shape}")
        return array

    def expected_grid_shape(self) -> Optional[tuple[int, int]]:
        width = self.grid.get("width")
        height = self.grid.get("height")
        if width and height:
            return int(height), int(width)
        return None

    @staticmethod
    def _diagnostics_to_numpy(diagnostics: Dict[str, Any]) -> Dict[str, np.ndarray]:
        result: Dict[str, np.ndarray] = {}
        for name, value in diagnostics.items():
            if isinstance(value, torch.Tensor):
                result[name] = value[0].detach().cpu().numpy()
        return result

    def _safe_load(self, path: pathlib.Path):
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            return torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=self.device)

    @staticmethod
    def _default_grid(pipeline_version: str) -> Dict[str, Any]:
        pipeline = RadarPipeline.canonical() if pipeline_version == CANONICAL_PIPELINE_VERSION else RadarPipeline()
        return pipeline.metadata()["grid"]
