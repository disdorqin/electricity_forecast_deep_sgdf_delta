"""Loss functions for DeepSGDFDelta training.

Implements:
  - sMAPE_floor50_loss: differentiable business-aligned loss
  - delta_mae_loss: MAE on price delta
  - period_9_16_weighted_loss: extra weight on 9-16 solar-volatile hours
  - smoothness_loss: penalises hour-to-hour prediction jumps
  - CombinedLoss: configurable weighted sum
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SMAPEFloor50Loss(nn.Module):
    """Differentiable sMAPE with floor-50 capping, matching the business metric."""

    def __init__(self, floor: float = 50.0, eps: float = 1e-6):
        super().__init__()
        self.floor = floor
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        yt = torch.clamp(y_true, min=self.floor)
        yp = torch.clamp(y_pred, min=self.floor)
        denom = torch.abs(yt) + torch.abs(yp) + self.eps
        return torch.mean(200.0 * torch.abs(yp - yt) / denom)


class DeltaMAELoss(nn.Module):
    """MAE on delta (realtime - dayahead)."""

    def forward(self, delta_pred: torch.Tensor, delta_true: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(delta_pred, delta_true)


class Period916WeightedLoss(nn.Module):
    """Extra weight on 9-16 segment hours (solar-volatile period)."""

    def __init__(self, weight: float = 2.0):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        delta_pred: torch.Tensor,
        delta_true: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        # segment_id: 0=1_8, 1=9_16, 2=17_24
        w = torch.where(segment_ids == 1, self.weight, 1.0)
        loss = torch.abs(delta_pred - delta_true) * w
        return loss.mean()


class SmoothnessLoss(nn.Module):
    """Penalise hour-to-hour jumps in predicted delta within a day."""

    def forward(self, delta_pred_sequence: torch.Tensor) -> torch.Tensor:
        if delta_pred_sequence.shape[1] < 2:
            return torch.tensor(0.0, device=delta_pred_sequence.device)
        diff = delta_pred_sequence[:, 1:] - delta_pred_sequence[:, :-1]
        return torch.mean(diff ** 2)


class CombinedLoss(nn.Module):
    """Configurable weighted combination of all loss components.

    Default weights:
      0.55 * sMAPE_floor50_loss
    + 0.25 * delta_mae_loss
    + 0.10 * period_9_16_weighted_loss
    + 0.10 * smoothness_loss
    """

    def __init__(
        self,
        w_smape: float = 0.55,
        w_delta_mae: float = 0.25,
        w_period: float = 0.10,
        w_smooth: float = 0.10,
        period_916_weight: float = 2.0,
        floor: float = 50.0,
    ):
        super().__init__()
        self.w_smape = w_smape
        self.w_delta_mae = w_delta_mae
        self.w_period = w_period
        self.w_smooth = w_smooth

        self.smape_loss = SMAPEFloor50Loss(floor=floor)
        self.delta_mae_loss = DeltaMAELoss()
        self.period_loss = Period916WeightedLoss(weight=period_916_weight)
        self.smooth_loss = SmoothnessLoss()

    def forward(
        self,
        rt_pred: torch.Tensor,
        rt_true: torch.Tensor,
        delta_pred: torch.Tensor,
        delta_true: torch.Tensor,
        segment_ids: torch.Tensor | None = None,
        delta_pred_sequence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}

        losses["smape"] = self.smape_loss(rt_pred, rt_true)
        losses["delta_mae"] = self.delta_mae_loss(delta_pred, delta_true)

        if segment_ids is not None:
            losses["period"] = self.period_loss(delta_pred, delta_true, segment_ids)
        else:
            losses["period"] = torch.tensor(0.0, device=rt_pred.device)

        if delta_pred_sequence is not None:
            losses["smooth"] = self.smooth_loss(delta_pred_sequence)
        else:
            losses["smooth"] = torch.tensor(0.0, device=rt_pred.device)

        total = (
            self.w_smape * losses["smape"]
            + self.w_delta_mae * losses["delta_mae"]
            + self.w_period * losses["period"]
            + self.w_smooth * losses["smooth"]
        )
        losses["total"] = total
        return losses
