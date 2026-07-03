"""Tests for V3 multi-objective loss functions (losses_v3.py)."""
from __future__ import annotations

import torch
import pytest

from models.deep_sgdf_delta.losses_v3 import (
    SMAPEFloor50Loss,
    DeltaMAELoss,
    Period916WeightedLoss,
    SmoothnessLoss,
    TeacherResidualDistillLoss,
    ConfidenceCalibrationLoss,
    CombinedLossV3,
)


# ── Test: SMAPEFloor50Loss ──────────────────────────────────────────

class TestSMAPEFloor50Loss:
    def test_perfect_prediction(self):
        loss_fn = SMAPEFloor50Loss()
        y = torch.tensor([100.0, 200.0, 300.0])
        loss = loss_fn(y, y)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_floor_clipping(self):
        """Values below floor=50 should be clipped."""
        loss_fn = SMAPEFloor50Loss(floor=50.0)
        y_pred = torch.tensor([10.0])
        y_true = torch.tensor([20.0])
        # Both clipped to 50: |50-50|/(50+50) = 0
        loss = loss_fn(y_pred, y_true)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_symmetry(self):
        loss_fn = SMAPEFloor50Loss()
        y_pred = torch.tensor([100.0])
        y_true = torch.tensor([200.0])
        loss1 = loss_fn(y_pred, y_true)
        loss2 = loss_fn(y_true, y_pred)
        assert loss1.item() == pytest.approx(loss2.item(), abs=1e-5)

    def test_gradient_flow(self):
        loss_fn = SMAPEFloor50Loss()
        y_pred = torch.tensor([100.0], requires_grad=True)
        y_true = torch.tensor([200.0])
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        assert y_pred.grad is not None


# ── Test: DeltaMAELoss ──────────────────────────────────────────────

class TestDeltaMAELoss:
    def test_zero_loss(self):
        loss_fn = DeltaMAELoss()
        d = torch.tensor([1.0, 2.0, 3.0])
        assert loss_fn(d, d).item() == pytest.approx(0.0)

    def test_known_value(self):
        loss_fn = DeltaMAELoss()
        pred = torch.tensor([1.0, 2.0])
        true = torch.tensor([3.0, 4.0])
        assert loss_fn(pred, true).item() == pytest.approx(2.0)


# ── Test: Period916WeightedLoss ─────────────────────────────────────

class TestPeriod916WeightedLoss:
    def test_higher_weight_for_segment_1(self):
        loss_fn = Period916WeightedLoss(weight=2.0)
        pred = torch.tensor([1.0, 1.0])
        true = torch.tensor([0.0, 0.0])
        seg = torch.tensor([0, 1])  # 1_8, 9_16
        loss = loss_fn(pred, true, seg)
        # seg=0 -> weight 1, |1-0|=1; seg=1 -> weight 2, |1-0|=2
        # mean = (1*1 + 2*1) / 2 = 1.5
        assert loss.item() == pytest.approx(1.5)

    def test_uniform_segments(self):
        loss_fn = Period916WeightedLoss(weight=2.0)
        pred = torch.tensor([1.0, 1.0, 1.0])
        true = torch.zeros(3)
        seg = torch.tensor([0, 0, 0])
        loss = loss_fn(pred, true, seg)
        assert loss.item() == pytest.approx(1.0)


# ── Test: SmoothnessLoss ────────────────────────────────────────────

class TestSmoothnessLoss:
    def test_smooth_sequence(self):
        loss_fn = SmoothnessLoss()
        seq = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        loss = loss_fn(seq)
        # All diffs = 1.0, mean(1^2) = 1.0
        assert loss.item() == pytest.approx(1.0)

    def test_constant_sequence(self):
        loss_fn = SmoothnessLoss()
        seq = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
        loss = loss_fn(seq)
        assert loss.item() == pytest.approx(0.0)

    def test_single_timestep(self):
        loss_fn = SmoothnessLoss()
        seq = torch.tensor([[5.0]])
        loss = loss_fn(seq)
        assert loss.item() == pytest.approx(0.0)


# ── Test: TeacherResidualDistillLoss ────────────────────────────────

class TestTeacherResidualDistillLoss:
    def test_no_teachers_returns_zero(self):
        loss_fn = TeacherResidualDistillLoss()
        sp = torch.randn(2, 24)
        dt = torch.randn(2, 24)
        loss = loss_fn(sp, dt, None, None)
        assert loss.item() == pytest.approx(0.0)

    def test_all_masked_returns_zero(self):
        loss_fn = TeacherResidualDistillLoss()
        sp = torch.randn(2, 24)
        dt = torch.randn(2, 24)
        tp = torch.randn(2, 3, 24)
        tm = torch.zeros(2, 3)
        loss = loss_fn(sp, dt, tp, tm)
        assert loss.item() == pytest.approx(0.0)

    def test_perfect_distill(self):
        """When student and teacher predict the same, residual should match."""
        loss_fn = TeacherResidualDistillLoss()
        dt = torch.randn(2, 24)
        sp = dt.clone()  # student predicts perfectly
        tp = dt.clone()  # teacher also predicts perfectly
        # student_residual = dt - sp = 0
        # teacher_residual = dt - tp = 0
        # diff = 0
        tm = torch.ones(2, 1)
        tp_3d = tp.unsqueeze(1)  # [2, 1, 24]
        loss = loss_fn(sp, dt, tp_3d, tm)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)


# ── Test: ConfidenceCalibrationLoss ─────────────────────────────────

class TestConfidenceCalibrationLoss:
    def test_perfect_confidence(self):
        loss_fn = ConfidenceCalibrationLoss()
        dp = torch.tensor([[100.0, 200.0]])
        dt = torch.tensor([[100.0, 200.0]])
        # error = 0 -> target_confidence = 1/(1+0) = 1.0
        conf = torch.tensor([[1.0, 1.0]])
        loss = loss_fn(conf, dp, dt)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_high_error_low_confidence(self):
        loss_fn = ConfidenceCalibrationLoss()
        dp = torch.tensor([[100.0]])
        dt = torch.tensor([[200.0]])
        # error = 100 -> target_confidence = 1/101 ≈ 0.0099
        conf = torch.tensor([[0.01]])
        loss = loss_fn(conf, dp, dt)
        assert loss.item() < 0.01  # very small


# ── Test: CombinedLossV3 ────────────────────────────────────────────

class TestCombinedLossV3:
    def test_returns_all_keys(self):
        loss_fn = CombinedLossV3()
        B = 2
        rt_pred = torch.randn(B, 24) + 100
        rt_true = torch.randn(B, 24) + 100
        delta_pred = torch.randn(B, 24)
        delta_true = torch.randn(B, 24)
        seg_ids = torch.zeros(B, 24, dtype=torch.long)
        conf = torch.sigmoid(torch.randn(B, 24))

        losses = loss_fn(rt_pred, rt_true, delta_pred, delta_true, seg_ids, conf)

        expected_keys = {"smape", "delta_mae", "period", "smooth",
                         "teacher_distill", "confidence_cal", "total"}
        assert set(losses.keys()) == expected_keys

    def test_total_is_weighted_sum(self):
        loss_fn = CombinedLossV3()
        B = 2
        rt_pred = torch.randn(B, 24) + 100
        rt_true = torch.randn(B, 24) + 100
        delta_pred = torch.randn(B, 24)
        delta_true = torch.randn(B, 24)
        seg_ids = torch.zeros(B, 24, dtype=torch.long)
        conf = torch.sigmoid(torch.randn(B, 24))

        losses = loss_fn(rt_pred, rt_true, delta_pred, delta_true, seg_ids, conf)

        expected_total = (
            0.45 * losses["smape"]
            + 0.20 * losses["delta_mae"]
            + 0.10 * losses["period"]
            + 0.10 * losses["smooth"]
            + 0.10 * losses["teacher_distill"]
            + 0.05 * losses["confidence_cal"]
        )
        assert losses["total"].item() == pytest.approx(expected_total.item(), abs=1e-4)

    def test_with_teacher_inputs(self):
        loss_fn = CombinedLossV3()
        B = 2
        rt_pred = torch.randn(B, 24) + 100
        rt_true = torch.randn(B, 24) + 100
        delta_pred = torch.randn(B, 24)
        delta_true = torch.randn(B, 24)
        seg_ids = torch.zeros(B, 24, dtype=torch.long)
        conf = torch.sigmoid(torch.randn(B, 24))
        tp = torch.randn(B, 3, 24)
        tm = torch.ones(B, 3)

        losses = loss_fn(rt_pred, rt_true, delta_pred, delta_true, seg_ids, conf,
                         teacher_pred=tp, teacher_mask=tm)

        assert losses["teacher_distill"].item() >= 0

    def test_empty_after_mask(self):
        loss_fn = CombinedLossV3()
        B = 2
        rt_pred = torch.randn(B, 24)
        rt_true = torch.randn(B, 24)
        delta_pred = torch.randn(B, 24)
        delta_true = torch.randn(B, 24)
        seg_ids = torch.zeros(B, 24, dtype=torch.long)
        conf = torch.sigmoid(torch.randn(B, 24))
        mask = torch.zeros(B, 24)  # all masked

        losses = loss_fn(rt_pred, rt_true, delta_pred, delta_true, seg_ids, conf,
                         valid_mask=mask)

        assert losses["total"].item() == pytest.approx(0.0)

    def test_gradient_flow(self):
        loss_fn = CombinedLossV3()
        B = 2
        rt_pred = (torch.randn(B, 24) + 100).requires_grad_(True)
        rt_true = torch.randn(B, 24) + 100
        delta_pred = torch.randn(B, 24, requires_grad=True)
        delta_true = torch.randn(B, 24)
        seg_ids = torch.zeros(B, 24, dtype=torch.long)
        conf_raw = torch.randn(B, 24, requires_grad=True)
        conf = torch.sigmoid(conf_raw)
        conf.retain_grad()

        losses = loss_fn(rt_pred, rt_true, delta_pred, delta_true, seg_ids, conf)
        losses["total"].backward()

        assert rt_pred.grad is not None
        assert delta_pred.grad is not None
        assert conf.grad is not None


# ── Phase 5 NaN Safety Tests ─────────────────────────────────────────

class TestTeacherNaNSafety:
    """Phase 5 Task B: teacher_pred 含 NaN 时 loss 不 NaN."""

    def test_nan_in_teacher_pred_returns_finite(self):
        """teacher_pred contains NaN → loss must be finite, not NaN."""
        loss_fn = TeacherResidualDistillLoss()
        dp = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        dt = torch.tensor([[1.5, 2.5, 3.5, 4.5]])
        tp = torch.tensor([[[10.0, 2.0, float('nan'), 4.0]]])  # NaN at h2
        tm = torch.tensor([[1.0]])
        result = loss_fn(dp, dt, tp, tm)
        assert torch.isfinite(result), f"Expected finite, got {result.item()}"
        assert result.item() >= 0

    def test_all_teacher_hours_nan_returns_zero(self):
        """All teacher_pred hours are NaN → no valid hours → loss = 0."""
        loss_fn = TeacherResidualDistillLoss()
        dp = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        dt = torch.tensor([[1.5, 2.5, 3.5, 4.5]])
        tp = torch.full((1, 1, 4), float('nan'))
        tm = torch.tensor([[1.0]])
        result = loss_fn(dp, dt, tp, tm)
        assert result.item() == pytest.approx(0.0)

    def test_partial_mask_only_valid_hours(self):
        """teacher_mask partially available → only compute on valid hours."""
        loss_fn = TeacherResidualDistillLoss()
        dp = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        dt = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        # Two teachers: teacher 0 available, teacher 1 all NaN pred
        tp = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                            [float('nan')]*4]])
        tm = torch.tensor([[1.0, 1.0]])
        result = loss_fn(dp, dt, tp, tm)
        assert torch.isfinite(result)
        # Teacher 0: student_res=[1,2,3,4], teacher_res=[0,0,0,0], diff=[1,4,9,16], mean=30/4=7.5
        # Teacher 1: all NaN → 0 valid hours → skip
        assert result.item() == pytest.approx(7.5, abs=1e-4)

    def test_nan_in_delta_true_returns_finite(self):
        """delta_true contains NaN → those hours excluded, loss finite."""
        loss_fn = TeacherResidualDistillLoss()
        dp = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        dt = torch.tensor([[1.5, float('nan'), 3.5, 4.5]])
        tp = torch.tensor([[[10.0, 2.0, 3.0, 4.0]]])
        tm = torch.tensor([[1.0]])
        result = loss_fn(dp, dt, tp, tm)
        assert torch.isfinite(result)

    def test_nan_in_delta_pred_student_returns_finite(self):
        """delta_pred_student contains NaN → those hours excluded, loss finite."""
        loss_fn = TeacherResidualDistillLoss()
        dp = torch.tensor([[float('nan'), 2.0, 3.0, 4.0]])
        dt = torch.tensor([[1.5, 2.5, 3.5, 4.5]])
        tp = torch.tensor([[[10.0, 2.0, 3.0, 4.0]]])
        tm = torch.tensor([[1.0]])
        result = loss_fn(dp, dt, tp, tm)
        assert torch.isfinite(result)

    def test_with_valid_mask(self):
        """valid_mask further restricts which hours are computed."""
        loss_fn = TeacherResidualDistillLoss()
        dp = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        dt = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        tp = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        tm = torch.tensor([[1.0]])
        vm = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # only first 2 hours valid
        result = loss_fn(dp, dt, tp, tm, valid_mask=vm)
        assert torch.isfinite(result)
        # Only h0, h1 valid: student_res=[1,2], teacher_res=[0,0], diff=[1,4], mean=5/2=2.5
        assert result.item() == pytest.approx(2.5, abs=1e-4)


class TestCombinedLossNaN:
    """Phase 5 Task B: total loss 永远 finite."""

    def _make_inputs(self, B=2):
        rt_pred = torch.randn(B, 24) + 100
        rt_true = torch.randn(B, 24) + 100
        dp = torch.randn(B, 24)
        dt = torch.randn(B, 24)
        seg = torch.zeros(B, 24, dtype=torch.long)
        seg[:, 8:16] = 1
        seg[:, 16:] = 2
        conf = torch.sigmoid(torch.randn(B, 24))
        return rt_pred, rt_true, dp, dt, seg, conf

    def test_nan_teacher_pred_total_finite(self):
        """teacher_pred has NaN → total loss must still be finite."""
        loss_fn = CombinedLossV3()
        rt_pred, rt_true, dp, dt, seg, conf = self._make_inputs()
        tp = torch.randn(2, 1, 24)
        tp[0, 0, :10] = float('nan')
        tm = torch.ones(2, 1)
        result = loss_fn(rt_pred, rt_true, dp, dt, seg, conf,
                         teacher_pred=tp, teacher_mask=tm)
        assert torch.isfinite(result["total"]), f"total={result['total'].item()}"
        assert torch.isfinite(result["teacher_distill"])

    def test_all_teacher_nan_total_finite(self):
        """All teacher predictions are NaN → total still finite."""
        loss_fn = CombinedLossV3()
        rt_pred, rt_true, dp, dt, seg, conf = self._make_inputs()
        tp = torch.full((2, 1, 24), float('nan'))
        tm = torch.ones(2, 1)
        result = loss_fn(rt_pred, rt_true, dp, dt, seg, conf,
                         teacher_pred=tp, teacher_mask=tm)
        assert torch.isfinite(result["total"])
        assert result["teacher_distill"].item() == pytest.approx(0.0)

    def test_inf_in_inputs_total_finite(self):
        """Inf in delta_true → total must be finite (non-finite components → 0)."""
        loss_fn = CombinedLossV3()
        rt_pred, rt_true, dp, dt, seg, conf = self._make_inputs()
        dt[0, 5] = float('inf')
        result = loss_fn(rt_pred, rt_true, dp, dt, seg, conf)
        assert torch.isfinite(result["total"])

    def test_multiple_teachers_some_nan(self):
        """Multiple teachers, some with NaN → only valid ones contribute."""
        loss_fn = CombinedLossV3()
        rt_pred, rt_true, dp, dt, seg, conf = self._make_inputs()
        tp = torch.randn(2, 3, 24)
        tp[0, 0, :] = float('nan')   # teacher 0 all NaN for batch 0
        tp[1, 2, :] = float('nan')   # teacher 2 all NaN for batch 1
        tm = torch.ones(2, 3)
        result = loss_fn(rt_pred, rt_true, dp, dt, seg, conf,
                         teacher_pred=tp, teacher_mask=tm)
        assert torch.isfinite(result["total"])
        assert result["teacher_distill"].item() > 0  # some teachers still valid
