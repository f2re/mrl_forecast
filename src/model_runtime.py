"""Shared CPU/GPU inference runtime for registered radar models."""

from __future__ import annotations

import pathlib
from typing import Any, Dict, Optional

import numpy as np
import torch

from convlstm import ConvLSTM
from phys_evolution import MRLPhysEvolution
from radar_pipeline import CANONICAL_PIPELINE_VERSION, PIPELINE_VERSION, RadarPipeline

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
        if self.info.get("pipeline_version") == CANONICAL_PIPELINE_VERSION:
            return RadarPipeline.canonical()
        return RadarPipeline()

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
        }
        return dict(self.info)

    def predict(
        self,
        reflectivity_dbz: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("No model is loaded")
        values, masks = self._prepare_arrays(reflectivity_dbz, valid_mask)
        x = torch.from_numpy(values).unsqueeze(0).unsqueeze(2).to(self.device)
        x_mask = torch.from_numpy(masks).unsqueeze(0).unsqueeze(2).to(self.device)

        with torch.no_grad():
            if self.architecture == "phys-evolution":
                prediction, diagnostics = self.model(x, x_mask)
            else:
                prediction, states = self.model(x)
                diagnostics = {"states": states}

        forecast = prediction[0, :, 0].cpu().numpy()
        result = {
            "input": x[0, :, 0].cpu().numpy(),
            "input_mask": x_mask[0, :, 0].cpu().numpy().astype(bool),
            "forecast": forecast,
            "diagnostics": self._diagnostics_to_numpy(diagnostics),
        }
        return result

    def _prepare_arrays(
        self,
        reflectivity_dbz: np.ndarray,
        valid_mask: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
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

        if valid_mask is None:
            masks = np.isfinite(values)
        else:
            masks = np.asarray(valid_mask, dtype=bool)
            if masks.ndim == 4 and masks.shape[-1] == 1:
                masks = masks.squeeze(-1)
            if masks.shape[0] >= self.input_length:
                masks = masks[-self.input_length:]
            if masks.shape != values.shape:
                raise ValueError("valid_mask must match selected reflectivity history")

        expected = self.expected_grid_shape()
        if expected and values.shape[-2:] != expected:
            raise ValueError(f"Source grid {values.shape[-2:]} is incompatible with model grid {expected}")
        normalized = np.clip(np.where(masks, values, 0.0), 0.0, MAX_DBZ) / MAX_DBZ
        return normalized.astype(np.float32), masks.astype(np.float32)

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
