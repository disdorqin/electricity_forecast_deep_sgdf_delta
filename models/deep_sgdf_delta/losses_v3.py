"""Multi-objective loss functions for TrendKnight-X v3.

Implements:
  - SMAPEFloor50Loss:         differentiable sMAPE with floor-50 capping
  - DeltaMAELoss:             MAE on price delta
  - Period916WeightedLoss:    extra weight on 9-16 solar-volatile hours
  - SmoothnessLoss:           penalise hour-to-hour prediction jumps
  - TeacherResidualDistillLoss: MSE between student residual and teacher residual
  - ConfidenceCalibrationLoss: encourage confidence to be low when error is high
  - CombinedLossV3:           configurable weighted sum of all components

Default weights:
  0.45 * smape + 0.20 * delta_mae + 0.10 * period_916
  + 0.10 * smoothness + 0.10 * teacher_distill + 0.05 * confidence_cal
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
        """delta_pred_sequence: [B, 24]"""
        if delta_pred_sequence.shape[1] < 2:
            return torch.tensor(0.0, device=delta_pred_sequence.device)
        diff = delta_pred_sequence[:, 1:] - delta_pred_sequence[:, :-1]
        return torch.mean(diff ** 2)


class TeacherResidualDistillLoss(nn.Module):
    """MSE between student residual and teacher residual, only where teacher is available.

    student_residual = delta_true - delta_pred_student
    teacher_residual = delta_true - delta_pred_teacher

    The student is encouraged to learn whatever the teacher already captures,
    so that the ensemble (teacher + student residual) improves.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        delta_pred_student: torch.Tensor,   # [B, 24]
        delta_true: torch.Tensor,            # [B, 24]
        teacher_pred: torch.Tensor,          # [B, num_teachers, 24]
        teacher_mask: torch.Tensor | None,   # [B, num_teachers]
    ) -> torch.Tensor:
        """Compute distillation loss.

        For each available teacher, compute MSE between:
          student_residual = delta_true - delta_pred_student
          teacher_residual = delta_true - teacher_pred

        Only compute where teacher_mask == 1.
        """
        if teacher_mask is None or teacher_mask.sum() == 0:
            return torch.tensor(0.0, device=delta_pred_student.device)

        student_residual = delta_true - delta_pred_student          # [B, 24]

        total_loss = torch.tensor(0.0, device=delta_pred_student.device)
        count = 0

        for t_idx in range(teacher_pred.size(1)):
            # Mask for this teacher: [B]
            t_available = teacher_mask[:, t_idx]                    # [B]
            if t_available.sum() == 0:
                continue

            teacher_residual = delta_true - teacher_pred[:, t_idx, :]  # [B, 24]

            # MSE only where teacher is available
            diff = (student_residual - teacher_residual) ** 2       # [B, 24]
            # Mask by teacher availability (broadcast over 24 hours)
            masked_diff = diff * t_available.unsqueeze(-1)          # [B, 24]
            total_loss = total_loss + masked_diff.sum()
            count += t_available.sum().item() * diff.size(1)

        if count == 0:
            return torch.tensor(0.0, device=delta_pred_student.device)

        return total_loss / count


class ConfidenceCalibrationLoss(nn.Module):
    """Encourage confidence to be low when error is high, and vice versa.

    Uses a negative correlation objective:
      loss = -corr(confidence, 1 - normalised_error)

    Simplified version: MSE between confidence and a soft target
    derived from the prediction error.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        confidence: torch.Tensor,       # [B, 24] in [0, 1]
        delta_pred: torch.Tensor,       # [B, 24]
        delta_true: torch.Tensor,       # [B, 24]
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute calibration loss.

        Target confidence = 1 / (1 + |error|) — high when error is small.
        """
        error = torch.abs(delta_pred - delta_true)                  # [B, 24]
        target_confidence = 1.0 / (1.0 + error)                     # [B, 24]

        loss = (confidence - target_confidence.detach()) ** 2

        if valid_mask is not None:
            loss = loss * valid_mask
            denom = valid_mask.sum().clamp(min=1.0)
        else:
            denom = torch.tensor(loss.numel(), dtype=torch.float, device=loss.device)

        return loss.sum() / denom


class CombinedLossV3(nn.Module):
    """Configurable weighted combination of all V3 loss components.

    Default weights:
      0.45 * sMAPE_floor50
    + 0.20 * delta_mae
    + 0.10 * period_9_16_weighted
    + 0.10 * smoothness
    + 0.10 * teacher_distill
    + 0.05 * confidence_calibration
    """

    def __init__(
        self,
        w_smape: float = 0.45,
        w_delta_mae: float = 0.20,
        w_period: float = 0.10,
        w_smooth: float = 0.10,
        w_teacher_distill: float = 0.10,
        w_confidence_cal: float = 0.05,
        period_916_weight: float = 2.0,
        floor: float = 50.0,
    ):
        super().__init__()
        self.w_smape = w_smape
        self.w_delta_mae = w_delta_mae
        self.w_period = w_period
        self.w_smooth = w_smooth
        self.w_teacher_distill = w_teacher_distill
        self.w_confidence_cal = w_confidence_cal

        self.smape_loss = SMAPEFloor50Loss(floor=floor)
        self.delta_mae_loss = DeltaMAELoss()
        self.period_loss = Period916WeightedLoss(weight=period_916_weight)
        self.smooth_loss = SmoothnessLoss()
        self.teacher_distill_loss = TeacherResidualDistillLoss()
        self.confidence_cal_loss = ConfidenceCalibrationLoss()

    def forward(
        self,
        rt_pred_24: torch.Tensor,              # [B, 24]
        rt_true_24: torch.Tensor,              # [B, 24]
        delta_pred_24: torch.Tensor,           # [B, 24]
        delta_true_24: torch.Tensor,           # [B, 24]
        segment_ids_24: torch.Tensor,          # [B, 24]
        confidence_24: torch.Tensor,           # [B, 24]
        valid_mask: torch.Tensor | None = None, # [B, 24]
        teacher_pred: torch.Tensor | None = None,     # [B, num_teachers, 24]
        teacher_mask: torch.Tensor | None = None,     # [B, num_teachers]
    ) -> dict[str, torch.Tensor]:
        """Compute combined V3 loss on 24-hour predictions.

        Returns dict with individual losses and total.
        """
        device = rt_pred_24.device

        # Apply valid_mask
        if valid_mask is not None:
            mask_bool = valid_mask.bool()
            rt_pred_flat = rt_pred_24[mask_bool]
            rt_true_flat = rt_true_24[mask_bool]
            delta_pred_flat = delta_pred_24[mask_bool]
            delta_true_flat = delta_true_24[mask_bool]
            seg_flat = segment_ids_24[mask_bool]
        else:
            rt_pred_flat = rt_pred_24.reshape(-1)
            rt_true_flat = rt_true_24.reshape(-1)
            delta_pred_flat = delta_pred_24.reshape(-1)
            delta_true_flat = delta_true_24.reshape(-1)
            seg_flat = segment_ids_24.reshape(-1)

        losses: dict[str, torch.Tensor] = {}

        if len(rt_pred_flat) == 0:
            zero = torch.tensor(0.0, device=device)
            return {
                "smape": zero, "delta_mae": zero, "period": zero,
                "smooth": zero, "teacher_distill": zero,
                "confidence_cal": zero, "total": zero,
            }

        # 1. sMAPE
        losses["smape"] = self.smape_loss(rt_pred_flat, rt_true_flat)

        # 2. Delta MAE
        losses["delta_mae"] = self.delta_mae_loss(delta_pred_flat, delta_true_flat)

        # 3. Period 9-16 weighted
        losses["period"] = self.period_loss(delta_pred_flat, delta_true_flat, seg_flat)

        # 4. Smoothness (on full 24h sequence)
        if valid_mask is not None:
            masked_delta = delta_pred_24 * valid_mask
            losses["smooth"] = self.smooth_loss(masked_delta)
        else:
            losses["smooth"] = self.smooth_loss(delta_pred_24)

        # 5. Teacher distillation
        if teacher_pred is not None and teacher_mask is not None:
            losses["teacher_distill"] = self.teacher_distill_loss(
                delta_pred_24, delta_true_24, teacher_pred, teacher_mask,
            )
        else:
            losses["teacher_distill"] = torch.tensor(0.0, device=device)

        # 6. Confidence calibration
        losses["confidence_cal"] = self.confidence_cal_loss(
            confidence_24, delta_pred_24, delta_true_24, valid_mask,
        )

        # Total
        losses["total"] = (
            self.w_smape * losses["smape"]
            + self.w_delta_mae * losses["delta_mae"]
            + self.w_period * losses["period"]
            + self.w_smooth * losses["smooth"]
            + self.w_teacher_distill * losses["teacher_distill"]
            + self.w_confidence_cal * losses["confidence_cal"]
        )

        return losses
