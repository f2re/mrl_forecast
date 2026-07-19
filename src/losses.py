"""Masked losses for radar reflectivity and explicit echo evolution."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from phys_evolution import dbz_norm_to_proxy


def _masked_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    mask = valid_mask.to(dtype=values.dtype)
    denominator = mask.sum()
    if denominator.item() == 0:
        return values.sum() * 0.0
    return (values * mask).sum() / denominator


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over valid target pixels only."""

    if prediction.shape != target.shape or target.shape != valid_mask.shape:
        raise ValueError("prediction, target and valid_mask must have identical shapes")
    return _masked_mean((prediction - target) ** 2, valid_mask)


class MaskedMSELoss(nn.Module):
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        diagnostics: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        del diagnostics
        return masked_mse(prediction, target, valid_mask)


class PhysicsEvolutionLoss(nn.Module):
    """Supervise reflectivity, advection and bounded growth/decay separately."""

    def __init__(
        self,
        gradient_weight: float = 0.05,
        evolution_weight: float = 0.20,
        flow_weight: float = 0.01,
        overlap_weight: float = 0.02,
        uncertainty_weight: float = 0.02,
    ):
        super().__init__()
        self.gradient_weight = gradient_weight
        self.evolution_weight = evolution_weight
        self.flow_weight = flow_weight
        self.overlap_weight = overlap_weight
        self.uncertainty_weight = uncertainty_weight

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        diagnostics: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if diagnostics is None:
            raise ValueError("PhysicsEvolutionLoss requires model diagnostics")
        if prediction.shape != target.shape or target.shape != valid_mask.shape:
            raise ValueError("prediction, target and valid_mask must have identical shapes")

        mask = valid_mask.to(dtype=prediction.dtype)
        heavy_weight = (
            1.0
            + 1.0 * (target >= 20.0 / 70.0).to(prediction.dtype)
            + 1.5 * (target >= 30.0 / 70.0).to(prediction.dtype)
            + 2.0 * (target >= 40.0 / 70.0).to(prediction.dtype)
        )
        pointwise = F.smooth_l1_loss(prediction, target, reduction="none") * heavy_weight
        data_loss = _masked_mean(pointwise, mask)

        gradient_loss = self._gradient_loss(prediction, target, mask)

        target_proxy = dbz_norm_to_proxy(target)
        target_evolution = target_proxy - diagnostics["advected_proxy"]
        predicted_evolution = diagnostics["growth"] - diagnostics["decay"]
        evolution_loss = _masked_mean(
            F.smooth_l1_loss(predicted_evolution, target_evolution, reduction="none"),
            mask,
        )

        motion = diagnostics["motion"]
        flow_loss = self._smoothness(motion)
        overlap_loss = torch.mean(diagnostics["growth"] * diagnostics["decay"])
        observed_error = torch.abs(prediction - target).detach()
        uncertainty_loss = _masked_mean(
            F.smooth_l1_loss(diagnostics["uncertainty"], observed_error, reduction="none"),
            mask,
        )

        return (
            data_loss
            + self.gradient_weight * gradient_loss
            + self.evolution_weight * evolution_loss
            + self.flow_weight * flow_loss
            + self.overlap_weight * overlap_loss
            + self.uncertainty_weight * uncertainty_loss
        )

    @staticmethod
    def _gradient_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
        target_x = target[..., :, 1:] - target[..., :, :-1]
        mask_x = valid_mask[..., :, 1:] * valid_mask[..., :, :-1]
        pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
        target_y = target[..., 1:, :] - target[..., :-1, :]
        mask_y = valid_mask[..., 1:, :] * valid_mask[..., :-1, :]
        return 0.5 * (
            _masked_mean(torch.abs(pred_x - target_x), mask_x)
            + _masked_mean(torch.abs(pred_y - target_y), mask_y)
        )

    @staticmethod
    def _smoothness(motion: torch.Tensor) -> torch.Tensor:
        dx = torch.abs(motion[..., :, 1:] - motion[..., :, :-1]).mean()
        dy = torch.abs(motion[..., 1:, :] - motion[..., :-1, :]).mean()
        return dx + dy
