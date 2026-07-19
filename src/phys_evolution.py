"""Lightweight physics-guided radar nowcasting model.

The model separates horizontal motion from reflectivity growth and decay. It is
not a full atmospheric PINN and does not claim conservation of water mass.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_DBZ = 70.0
PROXY_Z_SCALE = 1000.0
PROXY_DENOMINATOR = math.log1p((10.0 ** (MAX_DBZ / 10.0)) / PROXY_Z_SCALE)


def dbz_norm_to_proxy(reflectivity_norm: torch.Tensor) -> torch.Tensor:
    """Convert normalized dBZ to a stable monotonic linear-Z proxy."""

    dbz = torch.clamp(reflectivity_norm, 0.0, 1.0) * MAX_DBZ
    linear_z = torch.pow(10.0, dbz / 10.0)
    return torch.log1p(linear_z / PROXY_Z_SCALE) / PROXY_DENOMINATOR


def proxy_to_dbz_norm(proxy: torch.Tensor) -> torch.Tensor:
    """Convert the model proxy back to normalized non-negative dBZ."""

    linear_z = PROXY_Z_SCALE * torch.expm1(torch.clamp(proxy, 0.0, 1.0) * PROXY_DENOMINATOR)
    dbz = 10.0 * torch.log10(torch.clamp(linear_z, min=1.0))
    return torch.clamp(dbz / MAX_DBZ, 0.0, 1.0)


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        groups = 4 if output_channels % 4 == 0 else 1
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
        )


class FrameEncoder(nn.Module):
    """Encode full-resolution frames at one quarter of the spatial resolution."""

    def __init__(self, input_channels: int = 3, base_channels: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            ConvBlock(input_channels, base_channels),
            ConvBlock(base_channels, base_channels, stride=2),
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 2, stride=2),
        )
        self.output_channels = base_channels * 2

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        total_channels = input_channels + hidden_channels
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(total_channels, hidden_channels * 2, 3, padding=1)
        self.candidate = nn.Conv2d(total_channels, hidden_channels, 3, padding=1)

    def forward(self, value: torch.Tensor, hidden: Optional[torch.Tensor]) -> torch.Tensor:
        if hidden is None:
            hidden = torch.zeros(
                value.shape[0],
                self.hidden_channels,
                value.shape[-2],
                value.shape[-1],
                device=value.device,
                dtype=value.dtype,
            )
        reset, update = torch.sigmoid(self.gates(torch.cat([value, hidden], dim=1))).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([value, reset * hidden], dim=1)))
        return (1.0 - update) * hidden + update * candidate


class DifferentiableAdvection(nn.Module):
    """Semi-Lagrangian advection using flow in full-grid pixels per model step."""

    def forward(self, field: torch.Tensor, motion_pixels: torch.Tensor) -> torch.Tensor:
        batch, _channels, height, width = field.shape
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=field.device, dtype=field.dtype),
            torch.linspace(-1.0, 1.0, width, device=field.device, dtype=field.dtype),
            indexing="ij",
        )
        base_grid = torch.stack([x, y], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        x_scale = 2.0 / max(width - 1, 1)
        y_scale = 2.0 / max(height - 1, 1)
        displacement = torch.stack(
            [motion_pixels[:, 0] * x_scale, motion_pixels[:, 1] * y_scale],
            dim=-1,
        )
        sampling_grid = base_grid - displacement
        return F.grid_sample(
            field,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )


class MRLPhysEvolution(nn.Module):
    """Forecast motion, growth, decay and uncertainty for existing radar echo."""

    def __init__(
        self,
        input_channels: int = 3,
        base_channels: int = 16,
        hidden_channels: int = 24,
        output_steps: int = 4,
        max_motion_pixels: float = 14.0,
        max_evolution_per_step: float = 0.08,
    ):
        super().__init__()
        self.output_steps = output_steps
        self.max_motion_pixels = max_motion_pixels
        self.max_evolution_per_step = max_evolution_per_step
        self.encoder = FrameEncoder(input_channels, base_channels)
        self.temporal = ConvGRUCell(self.encoder.output_channels, hidden_channels)
        self.motion_head = nn.Conv2d(hidden_channels, 2, 3, padding=1)
        self.growth_head = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        self.decay_head = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        self.uncertainty_head = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        self.advection = DifferentiableAdvection()

    @staticmethod
    def _range_norm(height: int, width: int, device, dtype) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.clamp(torch.sqrt(x * x + y * y), 0.0, 1.0).unsqueeze(0).unsqueeze(0)

    def _encode_frame(
        self,
        reflectivity: torch.Tensor,
        valid_mask: torch.Tensor,
        range_norm: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(torch.cat([reflectivity, valid_mask, range_norm], dim=1))

    def forward(
        self,
        reflectivity: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if reflectivity.ndim != 5 or reflectivity.shape[2] != 1:
            raise ValueError("reflectivity must have shape [B,T,1,H,W]")
        if valid_mask is None:
            valid_mask = torch.ones_like(reflectivity)
        if valid_mask.shape != reflectivity.shape:
            raise ValueError("valid_mask must match reflectivity shape")

        batch, time_steps, _channels, height, width = reflectivity.shape
        static_range = self._range_norm(height, width, reflectivity.device, reflectivity.dtype)
        static_range = static_range.expand(batch, -1, -1, -1)

        hidden = None
        for index in range(time_steps):
            encoded = self._encode_frame(
                reflectivity[:, index],
                valid_mask[:, index],
                static_range,
            )
            hidden = self.temporal(encoded, hidden)

        current_proxy = dbz_norm_to_proxy(reflectivity[:, -1])
        current_mask = valid_mask[:, -1]
        forecasts = []
        motion_outputs = []
        growth_outputs = []
        decay_outputs = []
        uncertainty_outputs = []
        advected_outputs = []
        advected_proxy_outputs = []

        for _ in range(self.output_steps):
            full_size = (height, width)
            motion = F.interpolate(
                torch.tanh(self.motion_head(hidden)) * self.max_motion_pixels,
                size=full_size,
                mode="bilinear",
                align_corners=False,
            )
            growth = F.interpolate(
                torch.sigmoid(self.growth_head(hidden)) * self.max_evolution_per_step,
                size=full_size,
                mode="bilinear",
                align_corners=False,
            )
            decay = F.interpolate(
                torch.sigmoid(self.decay_head(hidden)) * self.max_evolution_per_step,
                size=full_size,
                mode="bilinear",
                align_corners=False,
            )
            uncertainty = F.interpolate(
                F.softplus(self.uncertainty_head(hidden)) * 0.08 + 0.005,
                size=full_size,
                mode="bilinear",
                align_corners=False,
            )

            advected_proxy = self.advection(current_proxy, motion)
            next_proxy = torch.clamp(advected_proxy + growth - decay, 0.0, 1.0)
            next_reflectivity = proxy_to_dbz_norm(next_proxy) * current_mask

            forecasts.append(next_reflectivity)
            motion_outputs.append(motion)
            growth_outputs.append(growth)
            decay_outputs.append(decay)
            uncertainty_outputs.append(uncertainty)
            advected_outputs.append(proxy_to_dbz_norm(advected_proxy))
            advected_proxy_outputs.append(advected_proxy)

            current_proxy = next_proxy
            encoded = self._encode_frame(next_reflectivity, current_mask, static_range)
            hidden = self.temporal(encoded, hidden)

        forecast = torch.stack(forecasts, dim=1)
        diagnostics = {
            "motion": torch.stack(motion_outputs, dim=1),
            "growth": torch.stack(growth_outputs, dim=1),
            "decay": torch.stack(decay_outputs, dim=1),
            "uncertainty": torch.stack(uncertainty_outputs, dim=1),
            "advected": torch.stack(advected_outputs, dim=1),
            "advected_proxy": torch.stack(advected_proxy_outputs, dim=1),
            "valid_mask": current_mask.unsqueeze(1).expand(-1, self.output_steps, -1, -1, -1),
        }
        return forecast, diagnostics
