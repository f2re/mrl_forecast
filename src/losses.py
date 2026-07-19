"""Loss functions that preserve the distinction between no echo and no data."""

from __future__ import annotations

import torch
import torch.nn as nn


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over valid target pixels only."""

    if prediction.shape != target.shape or target.shape != valid_mask.shape:
        raise ValueError("prediction, target and valid_mask must have identical shapes")
    mask = valid_mask.to(dtype=prediction.dtype)
    denominator = mask.sum()
    if denominator.item() == 0:
        return prediction.sum() * 0.0
    return (((prediction - target) ** 2) * mask).sum() / denominator


class MaskedMSELoss(nn.Module):
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return masked_mse(prediction, target, valid_mask)
