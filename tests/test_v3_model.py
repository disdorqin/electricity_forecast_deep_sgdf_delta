"""Tests for TrendKnight-X v3 model (model_v3.py)."""
from __future__ import annotations

import torch
import pytest

from models.deep_sgdf_delta.model_v3 import (
    TrendKnightV3Config,
    TrendKnightV3,
    build_model_v3,
    count_parameters,
    FeatureProjection,
    HourEmbedding,
    SegmentEmbedding,
    MultiscaleDecomposer,
    PeriodBranch,
    TeacherFusionGate,
    DayDecoderHead,
    ConfidenceHead,
    ShockSensitivityHead,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_inputs(B=2, input_dim=40, device="cpu"):
    """Create dummy inputs for TrendKnightV3 forward pass."""
    features = torch.randn(B, 24, input_dim, device=device)
    segment_id = torch.zeros(B, dtype=torch.long, device=device)
    for i in range(B):
        segment_id[i] = i % 3
    da_anchor = torch.randn(B, 24, device=device)
    hour_ids = torch.arange(1, 25, device=device).unsqueeze(0).expand(B, -1)
    return features, segment_id, da_anchor, hour_ids


def _make_teacher_inputs(B=2, num_teachers=3):
    teacher_features = torch.randn(B, num_teachers, 24)
    teacher_mask = torch.ones(B, num_teachers)
    # Mask out one teacher for the second sample
    teacher_mask[1, 2] = 0
    return teacher_features, teacher_mask


# ── Test: Config defaults ────────────────────────────────────────────

class TestConfig:
    def test_default_config(self):
        cfg = TrendKnightV3Config()
        assert cfg.input_dim == 40
        assert cfg.hidden_dim == 96
        assert cfg.backbone == "tcn"
        assert cfg.multiscale is True
        assert cfg.use_teacher_gate is True
        assert cfg.teacher_input_dim == 3

    def test_custom_config(self):
        cfg = TrendKnightV3Config(hidden_dim=64, backbone="gru", multiscale=False)
        assert cfg.hidden_dim == 64
        assert cfg.backbone == "gru"
        assert cfg.multiscale is False


# ── Test: Building blocks ────────────────────────────────────────────

class TestBuildingBlocks:
    def test_feature_projection_shape(self):
        fp = FeatureProjection(40, 96)
        x = torch.randn(2, 24, 40)
        out = fp(x)
        assert out.shape == (2, 24, 96)

    def test_hour_embedding_shape(self):
        he = HourEmbedding(25, 8)
        ids = torch.arange(1, 25).unsqueeze(0).expand(2, -1)
        out = he(ids)
        assert out.shape == (2, 24, 8)

    def test_segment_embedding_shape(self):
        se = SegmentEmbedding(3, 8)
        ids = torch.tensor([0, 1, 2])
        out = se(ids)
        assert out.shape == (3, 8)

    def test_multiscale_decomposer_output_keys(self):
        md = MultiscaleDecomposer(96)
        h = torch.randn(2, 24, 96)
        out = md(h)
        assert set(out.keys()) == {"trend", "seasonal", "shock", "combined"}
        for key in out:
            assert out[key].shape == (2, 24)

    def test_period_branch_shape(self):
        pb = PeriodBranch(96, 8)
        h = torch.randn(2, 24, 96)
        seg_emb = torch.randn(2, 8)
        out = pb(h, seg_emb)
        assert out.shape == (2, 24, 96)

    def test_teacher_fusion_gate_with_teachers(self):
        tg = TeacherFusionGate(96, 3)
        h = torch.randn(2, 24, 96)
        tf, tm = _make_teacher_inputs()
        out = tg(h, tf, tm)
        assert out.shape == (2, 24, 96)

    def test_teacher_fusion_gate_no_teachers(self):
        tg = TeacherFusionGate(96, 3)
        h = torch.randn(2, 24, 96)
        out = tg(h, None, None)
        assert out.shape == (2, 24, 96)
        assert torch.equal(out, h)

    def test_teacher_fusion_gate_all_masked(self):
        tg = TeacherFusionGate(96, 3)
        h = torch.randn(2, 24, 96)
        tf = torch.randn(2, 3, 24)
        tm = torch.zeros(2, 3)  # all teachers masked out
        out = tg(h, tf, tm)
        assert out.shape == (2, 24, 96)
        assert torch.equal(out, h)

    def test_day_decoder_head_shape(self):
        ddh = DayDecoderHead(96)
        h = torch.randn(2, 24, 96)
        out = ddh(h)
        assert out.shape == (2, 24)

    def test_confidence_head_range(self):
        ch = ConfidenceHead(96)
        h = torch.randn(2, 24, 96)
        out = ch(h)
        assert out.shape == (2, 24)
        assert (out >= 0).all() and (out <= 1).all()

    def test_shock_sensitivity_head_range(self):
        ssh = ShockSensitivityHead(96)
        h = torch.randn(2, 24, 96)
        out = ssh(h)
        assert out.shape == (2, 24)
        assert (out >= 0).all() and (out <= 1).all()


# ── Test: Full V3 model forward ─────────────────────────────────────

class TestTrendKnightV3Forward:
    @pytest.mark.parametrize("backbone", ["tcn", "gru", "transformer_tiny"])
    def test_forward_shapes(self, backbone):
        cfg = TrendKnightV3Config(backbone=backbone, hidden_dim=32, num_layers=1)
        model = TrendKnightV3(cfg)
        model.eval()

        features, seg_id, da_anchor, hour_ids = _make_inputs(B=2, input_dim=cfg.input_dim)
        with torch.no_grad():
            out = model(features, seg_id, da_anchor, hour_ids)

        assert out["delta_pred_24"].shape == (2, 24)
        assert out["rt_pred_24"].shape == (2, 24)
        assert out["confidence_24"].shape == (2, 24)
        assert out["shock_sensitivity_24"].shape == (2, 24)
        assert out["multiscale_trend"].shape == (2, 24)
        assert out["multiscale_seasonal"].shape == (2, 24)
        assert out["multiscale_shock"].shape == (2, 24)

    def test_rt_equals_anchor_plus_delta(self):
        cfg = TrendKnightV3Config(hidden_dim=32, num_layers=1)
        model = TrendKnightV3(cfg)
        model.eval()

        features, seg_id, da_anchor, hour_ids = _make_inputs(B=2, input_dim=cfg.input_dim)
        with torch.no_grad():
            out = model(features, seg_id, da_anchor, hour_ids)

        expected_rt = da_anchor + out["delta_pred_24"]
        torch.testing.assert_close(out["rt_pred_24"], expected_rt)

    def test_with_teacher_features(self):
        cfg = TrendKnightV3Config(hidden_dim=32, num_layers=1, teacher_input_dim=3)
        model = TrendKnightV3(cfg)
        model.eval()

        features, seg_id, da_anchor, hour_ids = _make_inputs(B=2, input_dim=cfg.input_dim)
        tf, tm = _make_teacher_inputs(B=2, num_teachers=3)

        with torch.no_grad():
            out = model(features, seg_id, da_anchor, hour_ids, tf, tm)

        assert out["delta_pred_24"].shape == (2, 24)

    def test_without_multiscale(self):
        cfg = TrendKnightV3Config(hidden_dim=32, num_layers=1, multiscale=False)
        model = TrendKnightV3(cfg)
        model.eval()

        features, seg_id, da_anchor, hour_ids = _make_inputs(B=2, input_dim=cfg.input_dim)
        with torch.no_grad():
            out = model(features, seg_id, da_anchor, hour_ids)

        # Without multiscale, trend/seasonal/shock should be zeros
        assert out["multiscale_trend"].abs().sum() == 0
        assert out["multiscale_seasonal"].abs().sum() == 0
        assert out["multiscale_shock"].abs().sum() == 0

    def test_without_teacher_gate(self):
        cfg = TrendKnightV3Config(hidden_dim=32, num_layers=1, use_teacher_gate=False)
        model = TrendKnightV3(cfg)
        model.eval()

        features, seg_id, da_anchor, hour_ids = _make_inputs(B=2, input_dim=cfg.input_dim)
        tf, tm = _make_teacher_inputs(B=2)

        with torch.no_grad():
            out = model(features, seg_id, da_anchor, hour_ids, tf, tm)

        assert out["delta_pred_24"].shape == (2, 24)

    def test_default_hour_ids(self):
        """When hour_ids is None, model should use default 1..24."""
        cfg = TrendKnightV3Config(hidden_dim=32, num_layers=1)
        model = TrendKnightV3(cfg)
        model.eval()

        features, seg_id, da_anchor, _ = _make_inputs(B=2, input_dim=cfg.input_dim)
        with torch.no_grad():
            out = model(features, seg_id, da_anchor, hour_ids=None)
        assert out["delta_pred_24"].shape == (2, 24)

    def test_confidence_in_valid_range(self):
        cfg = TrendKnightV3Config(hidden_dim=32, num_layers=1)
        model = TrendKnightV3(cfg)
        model.eval()

        features, seg_id, da_anchor, hour_ids = _make_inputs(B=4, input_dim=cfg.input_dim)
        with torch.no_grad():
            out = model(features, seg_id, da_anchor, hour_ids)

        assert (out["confidence_24"] >= 0).all()
        assert (out["confidence_24"] <= 1).all()
        assert (out["shock_sensitivity_24"] >= 0).all()
        assert (out["shock_sensitivity_24"] <= 1).all()


# ── Test: Parameter count ────────────────────────────────────────────

class TestParameterCount:
    @pytest.mark.parametrize("backbone", ["tcn", "gru", "transformer_tiny"])
    def test_parameter_count_under_500k(self, backbone):
        cfg = TrendKnightV3Config(backbone=backbone, hidden_dim=96)
        model = build_model_v3(cfg)
        n_params = count_parameters(model)
        assert n_params < 500_000, f"{backbone} has {n_params:,} params (limit 500k)"

    def test_parameter_count_returns_positive(self):
        cfg = TrendKnightV3Config(hidden_dim=32, num_layers=1)
        model = build_model_v3(cfg)
        assert count_parameters(model) > 0


# ── Test: Factory ────────────────────────────────────────────────────

class TestFactory:
    def test_build_model_v3(self):
        cfg = TrendKnightV3Config(hidden_dim=32, num_layers=1)
        model = build_model_v3(cfg)
        assert isinstance(model, TrendKnightV3)
        assert model.config is cfg

    def test_invalid_backbone_raises(self):
        cfg = TrendKnightV3Config(backbone="invalid_backbone")
        with pytest.raises(ValueError, match="Unknown backbone"):
            TrendKnightV3(cfg)
