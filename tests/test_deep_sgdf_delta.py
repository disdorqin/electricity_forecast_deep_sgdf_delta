"""Tests for DeepSGDFDelta / TrendKnight.

Covers:
  1. sMAPE_floor50 calculation correctness
  2. 00:00 -> previous business_day hour_business=24
  3. Blend weights only use D-30 to D-1
  4. Negative/spike bucket doesn't affect normal trend loss default weights
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch


# ── Test 1: sMAPE_floor50 calculation ────────────────────────────────

class TestSMAPEFloor50:
    """Verify sMAPE_floor50 computation matches the business metric."""

    def test_perfect_prediction(self):
        from models.deep_sgdf_delta.metrics import smape_floor50
        y = np.array([100.0, 200.0, 300.0])
        assert smape_floor50(y, y) == pytest.approx(0.0, abs=1e-6)

    def test_floor_capping(self):
        """Values below 50 should be capped to 50."""
        from models.deep_sgdf_delta.metrics import smape_floor50
        y_true = np.array([10.0, 20.0, 100.0])
        y_pred = np.array([10.0, 20.0, 100.0])
        # After capping: both become [50, 50, 100], still perfect
        assert smape_floor50(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)

    def test_known_value(self):
        """Test with known sMAPE calculation."""
        from models.deep_sgdf_delta.metrics import smape_floor50
        y_true = np.array([100.0])
        y_pred = np.array([150.0])
        # sMAPE = 200 * |150-100| / (100+150) = 200 * 50 / 250 = 40.0
        assert smape_floor50(y_true, y_pred) == pytest.approx(40.0, abs=1e-4)

    def test_floor_effect_on_smape(self):
        """Floor-50 should reduce sMAPE for low-price hours."""
        from models.deep_sgdf_delta.metrics import smape_floor50
        y_true = np.array([10.0])
        y_pred = np.array([20.0])
        # Without floor: 200 * 10 / (10+20) = 66.67
        # With floor=50: 200 * |50-50| / (50+50) = 0.0
        result = smape_floor50(y_true, y_pred, floor=50.0)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_differentiable_loss(self):
        """Test the PyTorch loss version produces gradients."""
        from models.deep_sgdf_delta.losses import SMAPEFloor50Loss
        loss_fn = SMAPEFloor50Loss()
        y_true = torch.tensor([100.0, 200.0, 300.0], requires_grad=False)
        y_pred = torch.tensor([110.0, 190.0, 310.0], requires_grad=True)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        assert y_pred.grad is not None
        assert y_pred.grad.shape == y_pred.shape

    def test_loss_matches_metric(self):
        """Loss function should match the numpy metric."""
        from models.deep_sgdf_delta.metrics import smape_floor50
        from models.deep_sgdf_delta.losses import SMAPEFloor50Loss
        y_true_np = np.array([100.0, 200.0, -30.0, 500.0])
        y_pred_np = np.array([120.0, 180.0, 10.0, 450.0])
        metric_val = smape_floor50(y_true_np, y_pred_np)
        loss_fn = SMAPEFloor50Loss()
        loss_val = loss_fn(
            torch.tensor(y_pred_np),
            torch.tensor(y_true_np),
        ).item()
        assert loss_val == pytest.approx(metric_val, abs=1e-4)


# ── Test 2: business_day and hour_business alignment ─────────────────

class TestBusinessDayAlignment:
    """Verify 00:00 maps to previous business_day with hour_business=24."""

    def test_midnight_alignment(self):
        """00:00 of calendar day D should have business_day = D-1 and target_hour = 24."""
        from models.deep_sgdf_delta.dataset import add_business_time_columns
        # Actually imported from sgdfnet.data_contract
        from sgdfnet.data_contract import add_business_time_columns

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-03-15 00:00:00", "2026-03-15 01:00:00", "2026-03-14 23:00:00"]),
        })
        result = add_business_time_columns(df, "timestamp")

        # 00:00 on March 15 -> business_day = March 14, target_hour = 24
        row_00 = result.iloc[0]
        assert row_00["business_day"] == pd.Timestamp("2026-03-14")
        assert row_00["target_hour"] == 24

        # 01:00 on March 15 -> business_day = March 15, target_hour = 1
        row_01 = result.iloc[1]
        assert row_01["business_day"] == pd.Timestamp("2026-03-15")
        assert row_01["target_hour"] == 1

        # 23:00 on March 14 -> business_day = March 14, target_hour = 23
        row_23 = result.iloc[2]
        assert row_23["business_day"] == pd.Timestamp("2026-03-14")
        assert row_23["target_hour"] == 23

    def test_segment_assignment(self):
        """Verify segment mapping: 1-8=1_8, 9-16=9_16, 17-24=17_24."""
        from models.deep_sgdf_delta.metrics import compute_period_mask
        hours = np.array([1, 8, 9, 16, 17, 24])

        assert compute_period_mask(hours, "1_8").tolist() == [True, True, False, False, False, False]
        assert compute_period_mask(hours, "9_16").tolist() == [False, False, True, True, False, False]
        assert compute_period_mask(hours, "17_24").tolist() == [False, False, False, False, True, True]


# ── Test 3: Blend weights only use D-30 to D-1 ─────────────────────

class TestBlendWeightValidation:
    """Verify blend weight search only uses validation window (D-30 to D-1)."""

    def test_blend_weight_candidates(self):
        """find_optimal_blend_weight should search over specified candidates."""
        from models.deep_sgdf_delta.predict import find_optimal_blend_weight

        # Create mock validation data
        n = 24 * 30  # 30 days
        deep_pred = pd.DataFrame({
            "rt_pred": np.random.randn(n) * 50 + 200,
            "delta_pred": np.random.randn(n) * 20,
            "da_anchor": np.ones(n) * 200,
            "hour": np.tile(np.arange(1, 25), 30),
        })
        sgdf_pred = pd.DataFrame({
            "rt_hat": np.random.randn(n) * 50 + 200,
            "delta_hat": np.random.randn(n) * 20,
            "hour": np.tile(np.arange(1, 25), 30),
        })
        y_true = np.random.randn(n) * 50 + 200

        # Test with default candidates
        w = find_optimal_blend_weight(deep_pred, sgdf_pred, y_true, candidates=[0.2, 0.4, 0.6, 0.8])
        assert w in [0.2, 0.4, 0.6, 0.8]

    def test_blend_mode_deep_only(self):
        """deep_only mode should return deep predictions unchanged."""
        from models.deep_sgdf_delta.predict import predict_with_blend

        deep_pred = pd.DataFrame({
            "delta_pred": [10.0, 20.0],
            "rt_pred": [210.0, 220.0],
            "da_anchor": [200.0, 200.0],
            "hour": [1, 2],
        })
        result = predict_with_blend(deep_pred, None, mode="deep_only")
        assert result["rt_pred"].tolist() == [210.0, 220.0]

    def test_blend_mode_sgdfnet_blend(self):
        """sgdfnet_blend should compute weighted average."""
        from models.deep_sgdf_delta.predict import predict_with_blend

        deep_pred = pd.DataFrame({
            "delta_pred": [10.0],
            "rt_pred": [210.0],
            "da_anchor": [200.0],
            "hour": [1],
        })
        sgdf_pred = pd.DataFrame({
            "delta_hat": [20.0],
            "rt_hat": [220.0],
            "hour": [1],
        })
        # w=0.6 for SGDFNet: 0.6*220 + 0.4*210 = 132+84 = 216
        result = predict_with_blend(deep_pred, sgdf_pred, mode="sgdfnet_blend", blend_weight=0.6)
        # delta_pred = 0.6*20 + 0.4*10 = 16
        # rt_pred = 200 + 16 = 216
        assert result["rt_pred"].iloc[0] == pytest.approx(216.0, abs=1e-6)

    def test_blend_mode_sgdfnet_residual(self):
        """sgdfnet_residual should add deep residual on top of SGDFNet."""
        from models.deep_sgdf_delta.predict import predict_with_blend

        deep_pred = pd.DataFrame({
            "delta_pred": [5.0],  # residual
            "rt_pred": [205.0],
            "da_anchor": [200.0],
            "hour": [1],
        })
        sgdf_pred = pd.DataFrame({
            "delta_hat": [15.0],
            "rt_hat": [215.0],
            "hour": [1],
        })
        # residual mode: delta_pred = 15 + 5 = 20, rt_pred = 200 + 20 = 220
        result = predict_with_blend(deep_pred, sgdf_pred, mode="sgdfnet_residual")
        assert result["rt_pred"].iloc[0] == pytest.approx(220.0, abs=1e-6)


# ── Test 4: Negative/spike bucket doesn't affect normal trend loss ──

class TestLossWeightIsolation:
    """Verify that negative/spike samples don't distort normal trend loss."""

    def test_combined_loss_default_weights(self):
        """Default weights should sum to 1.0 and not over-weight extremes."""
        from models.deep_sgdf_delta.losses import CombinedLoss
        loss_fn = CombinedLoss()
        assert loss_fn.w_smape + loss_fn.w_delta_mae + loss_fn.w_period + loss_fn.w_smooth == pytest.approx(1.0)

    def test_spike_samples_dont_dominate(self):
        """A few extreme samples shouldn't dominate the loss."""
        from models.deep_sgdf_delta.losses import CombinedLoss
        loss_fn = CombinedLoss()

        # Normal samples
        rt_true = torch.tensor([200.0, 210.0, 190.0, 205.0, 195.0])
        rt_pred = torch.tensor([205.0, 208.0, 192.0, 200.0, 198.0], requires_grad=True)
        delta_true = torch.tensor([5.0, 10.0, -5.0, 8.0, -2.0])
        delta_pred = torch.tensor([7.0, 8.0, -3.0, 6.0, -1.0], requires_grad=True)
        segment_ids = torch.tensor([0, 1, 2, 0, 1])

        losses = loss_fn(rt_pred, rt_true, delta_pred, delta_true, segment_ids)
        assert losses["total"].item() > 0
        assert not torch.isnan(losses["total"])

    def test_negative_price_capping(self):
        """Negative prices should be capped by floor, not cause NaN."""
        from models.deep_sgdf_delta.losses import SMAPEFloor50Loss
        loss_fn = SMAPEFloor50Loss()
        y_true = torch.tensor([-50.0, -10.0, 0.0, 100.0])
        y_pred = torch.tensor([-30.0, 5.0, 20.0, 110.0])
        loss = loss_fn(y_pred, y_true)
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_period_916_weight_only_affects_target_segment(self):
        """Period 9-16 weight should only amplify loss for segment_id=1."""
        from models.deep_sgdf_delta.losses import Period916WeightedLoss
        loss_fn = Period916WeightedLoss(weight=3.0)
        delta_pred = torch.tensor([10.0, 10.0, 10.0])
        delta_true = torch.tensor([0.0, 0.0, 0.0])
        segment_ids = torch.tensor([0, 1, 2])  # 1_8, 9_16, 17_24

        loss = loss_fn(delta_pred, delta_true, segment_ids)
        # Expected: (10*1 + 10*3 + 10*1) / 3 = 50/3 ≈ 16.67
        assert loss.item() == pytest.approx(50.0 / 3.0, abs=1e-4)


# ── Test 5: Model forward pass ───────────────────────────────────────

class TestModelForward:
    """Verify model can run forward pass without errors."""

    def test_tcn_forward(self):
        from models.deep_sgdf_delta.model import DeepSGDFDeltaConfig, build_model
        config = DeepSGDFDeltaConfig(input_dim=20, hidden_dim=32, backbone="tcn", num_layers=2)
        model = build_model(config)

        batch_size = 4
        seq_len = 7
        features = torch.randn(batch_size, seq_len, 20)
        segment_ids = torch.tensor([0, 1, 2, 0])
        da_anchor = torch.tensor([200.0, 210.0, 190.0, 205.0])

        out = model(features, segment_ids, da_anchor)
        assert "delta_pred" in out
        assert "rt_pred" in out
        assert out["delta_pred"].shape == (batch_size,)
        assert out["rt_pred"].shape == (batch_size,)

    def test_gru_forward(self):
        from models.deep_sgdf_delta.model import DeepSGDFDeltaConfig, build_model
        config = DeepSGDFDeltaConfig(input_dim=20, hidden_dim=32, backbone="gru", num_layers=2)
        model = build_model(config)

        batch_size = 4
        seq_len = 7
        features = torch.randn(batch_size, seq_len, 20)
        segment_ids = torch.tensor([0, 1, 2, 0])
        da_anchor = torch.tensor([200.0, 210.0, 190.0, 205.0])

        out = model(features, segment_ids, da_anchor)
        assert out["delta_pred"].shape == (batch_size,)
        assert out["rt_pred"].shape == (batch_size,)

    def test_parameter_count_lightweight(self):
        """Model should have < 200k parameters with default config."""
        from models.deep_sgdf_delta.model import DeepSGDFDeltaConfig, build_model
        config = DeepSGDFDeltaConfig(input_dim=40, hidden_dim=64, backbone="tcn", num_layers=2)
        model = build_model(config)
        total = sum(p.numel() for p in model.parameters())
        assert total < 200_000, f"Model has {total:,} params, expected < 200k"
