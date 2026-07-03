"""Tests for V2 day-level decoder model shapes and forward-pass correctness.

Covers:
  - TCN, GRU, Transformer-tiny backbone forward pass
  - Input [B, 24, input_dim] -> Output [B, 24]
  - Batch size 1 works
  - Different hidden dims work
  - rt_pred = da_anchor + delta_pred identity
"""
from __future__ import annotations

import pytest
import torch

from models.deep_sgdf_delta.model_v2 import (
    DeepSGDFDeltaV2,
    DeepSGDFDeltaV2Config,
    build_model_v2,
    count_parameters,
)


# ── Test: TCN forward [4, 24, 40] -> [4, 24] ─────────────────────────


class TestTCNForward:
    """TCN backbone forward pass produces correct output shape."""

    def test_tcn_output_shape(self):
        config = DeepSGDFDeltaV2Config(
            input_dim=40, hidden_dim=64, backbone="tcn", num_layers=2,
        )
        model = build_model_v2(config)
        model.eval()

        B, T, D = 4, 24, 40
        features = torch.randn(B, T, D)
        segment_id = torch.tensor([0, 1, 2, 0])
        da_anchor = torch.randn(B, T) * 20 + 200

        with torch.no_grad():
            out = model(features, segment_id, da_anchor)

        assert out["delta_pred_24"].shape == (B, T)
        assert out["rt_pred_24"].shape == (B, T)

    def test_tcn_no_nan(self):
        config = DeepSGDFDeltaV2Config(
            input_dim=40, hidden_dim=64, backbone="tcn", num_layers=2,
        )
        model = build_model_v2(config)
        model.eval()

        features = torch.randn(4, 24, 40)
        segment_id = torch.tensor([0, 1, 2, 0])
        da_anchor = torch.ones(4, 24) * 200

        with torch.no_grad():
            out = model(features, segment_id, da_anchor)

        assert not torch.isnan(out["delta_pred_24"]).any()
        assert not torch.isnan(out["rt_pred_24"]).any()


# ── Test: GRU forward [4, 24, 40] -> [4, 24] ─────────────────────────


class TestGRUForward:
    """GRU backbone forward pass produces correct output shape."""

    def test_gru_output_shape(self):
        config = DeepSGDFDeltaV2Config(
            input_dim=40, hidden_dim=64, backbone="gru", num_layers=2,
        )
        model = build_model_v2(config)
        model.eval()

        B, T, D = 4, 24, 40
        features = torch.randn(B, T, D)
        segment_id = torch.tensor([0, 1, 2, 0])
        da_anchor = torch.randn(B, T) * 20 + 200

        with torch.no_grad():
            out = model(features, segment_id, da_anchor)

        assert out["delta_pred_24"].shape == (B, T)
        assert out["rt_pred_24"].shape == (B, T)

    def test_gru_no_nan(self):
        config = DeepSGDFDeltaV2Config(
            input_dim=40, hidden_dim=64, backbone="gru", num_layers=2,
        )
        model = build_model_v2(config)
        model.eval()

        features = torch.randn(4, 24, 40)
        segment_id = torch.tensor([0, 1, 2, 0])
        da_anchor = torch.ones(4, 24) * 200

        with torch.no_grad():
            out = model(features, segment_id, da_anchor)

        assert not torch.isnan(out["delta_pred_24"]).any()
        assert not torch.isnan(out["rt_pred_24"]).any()


# ── Test: Transformer forward [4, 24, 40] -> [4, 24] ─────────────────


class TestTransformerForward:
    """Transformer-tiny backbone forward pass produces correct output shape."""

    def test_transformer_output_shape(self):
        config = DeepSGDFDeltaV2Config(
            input_dim=40, hidden_dim=64, backbone="transformer_tiny",
            num_layers=2, transformer_nhead=4, transformer_dim_ff=128,
        )
        model = build_model_v2(config)
        model.eval()

        B, T, D = 4, 24, 40
        features = torch.randn(B, T, D)
        segment_id = torch.tensor([0, 1, 2, 0])
        da_anchor = torch.randn(B, T) * 20 + 200

        with torch.no_grad():
            out = model(features, segment_id, da_anchor)

        assert out["delta_pred_24"].shape == (B, T)
        assert out["rt_pred_24"].shape == (B, T)

    def test_transformer_no_nan(self):
        config = DeepSGDFDeltaV2Config(
            input_dim=40, hidden_dim=64, backbone="transformer_tiny",
            num_layers=2, transformer_nhead=4, transformer_dim_ff=128,
        )
        model = build_model_v2(config)
        model.eval()

        features = torch.randn(4, 24, 40)
        segment_id = torch.tensor([0, 1, 2, 0])
        da_anchor = torch.ones(4, 24) * 200

        with torch.no_grad():
            out = model(features, segment_id, da_anchor)

        assert not torch.isnan(out["delta_pred_24"]).any()
        assert not torch.isnan(out["rt_pred_24"]).any()


# ── Test: Batch size 1 works ──────────────────────────────────────────


class TestBatchSizeOne:
    """All backbones should work with batch_size=1."""

    @pytest.mark.parametrize("backbone", ["tcn", "gru", "transformer_tiny"])
    def test_batch_size_1(self, backbone):
        kwargs = {"transformer_nhead": 4, "transformer_dim_ff": 128} if backbone == "transformer_tiny" else {}
        config = DeepSGDFDeltaV2Config(
            input_dim=40, hidden_dim=64, backbone=backbone, num_layers=2, **kwargs,
        )
        model = build_model_v2(config)
        model.eval()

        features = torch.randn(1, 24, 40)
        segment_id = torch.tensor([1])
        da_anchor = torch.ones(1, 24) * 200

        with torch.no_grad():
            out = model(features, segment_id, da_anchor)

        assert out["delta_pred_24"].shape == (1, 24)
        assert out["rt_pred_24"].shape == (1, 24)


# ── Test: Different hidden dims work ──────────────────────────────────


class TestDifferentHiddenDims:
    """Model should work with various hidden_dim values."""

    @pytest.mark.parametrize("hidden_dim", [32, 64, 128])
    def test_hidden_dims(self, hidden_dim):
        config = DeepSGDFDeltaV2Config(
            input_dim=40, hidden_dim=hidden_dim, backbone="tcn", num_layers=2,
        )
        model = build_model_v2(config)
        model.eval()

        features = torch.randn(2, 24, 40)
        segment_id = torch.tensor([0, 2])
        da_anchor = torch.ones(2, 24) * 200

        with torch.no_grad():
            out = model(features, segment_id, da_anchor)

        assert out["delta_pred_24"].shape == (2, 24)
        assert out["rt_pred_24"].shape == (2, 24)


# ── Test: rt_pred = da_anchor + delta_pred ────────────────────────────


class TestRtPredIdentity:
    """rt_pred_24 must equal da_anchor_24 + delta_pred_24."""

    @pytest.mark.parametrize("backbone", ["tcn", "gru", "transformer_tiny"])
    def test_rt_equals_anchor_plus_delta(self, backbone):
        kwargs = {"transformer_nhead": 4, "transformer_dim_ff": 128} if backbone == "transformer_tiny" else {}
        config = DeepSGDFDeltaV2Config(
            input_dim=40, hidden_dim=64, backbone=backbone, num_layers=2, **kwargs,
        )
        model = build_model_v2(config)
        model.eval()

        B = 4
        features = torch.randn(B, 24, 40)
        segment_id = torch.tensor([0, 1, 2, 0])
        da_anchor = torch.randn(B, 24) * 20 + 200

        with torch.no_grad():
            out = model(features, segment_id, da_anchor)

        expected_rt = da_anchor + out["delta_pred_24"]
        assert torch.allclose(out["rt_pred_24"], expected_rt, atol=1e-5), (
            f"rt_pred_24 != da_anchor + delta_pred_24 for backbone={backbone}"
        )


# ── Test: parameter count is lightweight ──────────────────────────────


class TestParameterCount:
    """V2 model should remain lightweight (< 300k params with default config)."""

    @pytest.mark.parametrize("backbone", ["tcn", "gru", "transformer_tiny"])
    def test_parameter_count(self, backbone):
        kwargs = {"transformer_nhead": 4, "transformer_dim_ff": 128} if backbone == "transformer_tiny" else {}
        config = DeepSGDFDeltaV2Config(
            input_dim=40, hidden_dim=64, backbone=backbone, num_layers=2, **kwargs,
        )
        model = build_model_v2(config)
        total = count_parameters(model)
        assert total < 300_000, (
            f"{backbone} model has {total:,} params, expected < 300k"
        )


# ── Test: unknown backbone raises ─────────────────────────────────────


class TestUnknownBackbone:
    def test_unknown_backbone_raises(self):
        config = DeepSGDFDeltaV2Config(
            input_dim=40, hidden_dim=64, backbone="lstm", num_layers=2,
        )
        with pytest.raises(ValueError, match="Unknown backbone"):
            build_model_v2(config)
