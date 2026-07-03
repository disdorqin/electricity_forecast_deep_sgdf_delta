"""Tests for models.deep_sgdf_delta.trendknight_rt — TrendKnightRT model.

Covers:
  - Forward pass shape correctness for TCN / GRU / Transformer backbones
  - Fusion modes A, B, C
  - Confidence head output range [0, 1]
  - Graceful handling when sgdfnet_pred is zeros
  - Parameter count under 500k with default config
  - Batch size 1 edge case
"""
from __future__ import annotations

import pytest
import torch

from models.deep_sgdf_delta.trendknight_rt import (
    TrendKnightRTConfig,
    TrendKnightRT,
    build_trendknight_rt,
    count_parameters,
)

# ── Fixtures ──────────────────────────────────────────────────────────

BATCH_SIZE = 4
INPUT_DIM = 40
NUM_HOURS = 24


@pytest.fixture
def default_config():
    """Default TrendKnightRTConfig."""
    return TrendKnightRTConfig(input_dim=INPUT_DIM)


@pytest.fixture
def random_inputs():
    """Random inputs for forward pass with batch_size=4."""
    torch.manual_seed(42)
    return {
        "features_24h": torch.randn(BATCH_SIZE, NUM_HOURS, INPUT_DIM),
        "segment_id": torch.randint(0, 3, (BATCH_SIZE,)),
        "da_anchor_24": torch.randn(BATCH_SIZE, NUM_HOURS) * 50 + 300,
        "sgdfnet_pred_24": torch.randn(BATCH_SIZE, NUM_HOURS) * 50 + 300,
    }


def _make_config(**overrides) -> TrendKnightRTConfig:
    """Build a config with optional overrides."""
    cfg = TrendKnightRTConfig(input_dim=INPUT_DIM)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ── Forward shape tests ───────────────────────────────────────────────

class TestForwardShape:
    def test_forward_shape_tcn(self, default_config, random_inputs):
        """TCN backbone produces correct output shapes."""
        default_config.backbone = "tcn"
        model = build_trendknight_rt(default_config)
        model.eval()

        with torch.no_grad():
            out = model(**random_inputs)

        assert out["trend_rt_pred_24"].shape == (BATCH_SIZE, NUM_HOURS)
        assert out["delta_pred_24"].shape == (BATCH_SIZE, NUM_HOURS)
        assert out["residual_to_sgdfnet_24"].shape == (BATCH_SIZE, NUM_HOURS)
        assert out["confidence_24"].shape == (BATCH_SIZE, NUM_HOURS)
        assert out["period_bias_24"].shape == (BATCH_SIZE, NUM_HOURS)

    def test_forward_shape_gru(self, random_inputs):
        """GRU backbone produces correct output shapes."""
        cfg = _make_config(backbone="gru")
        model = build_trendknight_rt(cfg)
        model.eval()

        with torch.no_grad():
            out = model(**random_inputs)

        assert out["trend_rt_pred_24"].shape == (BATCH_SIZE, NUM_HOURS)
        assert out["delta_pred_24"].shape == (BATCH_SIZE, NUM_HOURS)
        assert out["residual_to_sgdfnet_24"].shape == (BATCH_SIZE, NUM_HOURS)

    def test_forward_shape_transformer(self, random_inputs):
        """Transformer backbone produces correct output shapes."""
        cfg = _make_config(backbone="transformer")
        model = build_trendknight_rt(cfg)
        model.eval()

        with torch.no_grad():
            out = model(**random_inputs)

        assert out["trend_rt_pred_24"].shape == (BATCH_SIZE, NUM_HOURS)
        assert out["delta_pred_24"].shape == (BATCH_SIZE, NUM_HOURS)
        assert out["residual_to_sgdfnet_24"].shape == (BATCH_SIZE, NUM_HOURS)


# ── Fusion mode tests ─────────────────────────────────────────────────

class TestFusionModes:
    def test_fusion_mode_a(self, random_inputs):
        """Mode A: rt_pred = da_anchor + delta_pred."""
        cfg = _make_config(fusion_mode="A")
        model = build_trendknight_rt(cfg)
        model.eval()

        with torch.no_grad():
            out = model(**random_inputs)

        expected = random_inputs["da_anchor_24"] + out["delta_pred_24"]
        torch.testing.assert_close(out["trend_rt_pred_24"], expected, atol=1e-5, rtol=1e-5)

    def test_fusion_mode_b(self, random_inputs):
        """Mode B: rt_pred = sgdfnet_pred + residual_to_sgdfnet."""
        cfg = _make_config(fusion_mode="B")
        model = build_trendknight_rt(cfg)
        model.eval()

        with torch.no_grad():
            out = model(**random_inputs)

        expected = random_inputs["sgdfnet_pred_24"] + out["residual_to_sgdfnet_24"]
        torch.testing.assert_close(out["trend_rt_pred_24"], expected, atol=1e-5, rtol=1e-5)

    def test_fusion_mode_c(self, random_inputs):
        """Mode C: gated blend, gate in [0, 1]."""
        cfg = _make_config(fusion_mode="C")
        model = build_trendknight_rt(cfg)
        model.eval()

        with torch.no_grad():
            out = model(**random_inputs)

        # Mode C result should be between mode A and mode B (weighted by gate)
        # We just check it's finite and has the right shape
        assert out["trend_rt_pred_24"].shape == (BATCH_SIZE, NUM_HOURS)
        assert torch.isfinite(out["trend_rt_pred_24"]).all()

        # The result should not be exactly equal to mode A or mode B
        # (unless gate happens to be exactly 0 or 1, which is unlikely)
        # We mainly verify the shape and finiteness


# ── Confidence range ──────────────────────────────────────────────────

class TestConfidenceRange:
    def test_confidence_range(self, random_inputs):
        """Confidence head output in [0, 1]."""
        cfg = _make_config(use_confidence_head=True)
        model = build_trendknight_rt(cfg)
        model.eval()

        with torch.no_grad():
            out = model(**random_inputs)

        conf = out["confidence_24"]
        assert conf.min() >= 0.0, f"Confidence min {conf.min()} < 0"
        assert conf.max() <= 1.0, f"Confidence max {conf.max()} > 1"


# ── No SGDFNet ────────────────────────────────────────────────────────

class TestNoSgdfnet:
    def test_no_sgdfnet(self):
        """sgdfnet_pred = zeros, mode B should give residual only."""
        torch.manual_seed(42)
        cfg = _make_config(fusion_mode="B")
        model = build_trendknight_rt(cfg)
        model.eval()

        inputs = {
            "features_24h": torch.randn(2, NUM_HOURS, INPUT_DIM),
            "segment_id": torch.randint(0, 3, (2,)),
            "da_anchor_24": torch.randn(2, NUM_HOURS) * 50 + 300,
            "sgdfnet_pred_24": torch.zeros(2, NUM_HOURS),  # no SGDFNet
        }

        with torch.no_grad():
            out = model(**inputs)

        # Mode B: trend_rt = sgdfnet_pred + residual = 0 + residual = residual
        torch.testing.assert_close(
            out["trend_rt_pred_24"],
            out["residual_to_sgdfnet_24"],
            atol=1e-5,
            rtol=1e-5,
        )


# ── Parameter count ───────────────────────────────────────────────────

class TestParameterCount:
    def test_parameter_count(self, default_config):
        """Under 500k with default config."""
        model = build_trendknight_rt(default_config)
        n_params = count_parameters(model)
        assert n_params < 500_000, f"Parameter count {n_params} exceeds 500k"
        assert n_params > 0


# ── Batch size 1 ──────────────────────────────────────────────────────

class TestBatchSize1:
    def test_batch_size_1(self):
        """Works with batch_size=1."""
        torch.manual_seed(42)
        cfg = _make_config()
        model = build_trendknight_rt(cfg)
        model.eval()

        inputs = {
            "features_24h": torch.randn(1, NUM_HOURS, INPUT_DIM),
            "segment_id": torch.randint(0, 3, (1,)),
            "da_anchor_24": torch.randn(1, NUM_HOURS) * 50 + 300,
            "sgdfnet_pred_24": torch.randn(1, NUM_HOURS) * 50 + 300,
        }

        with torch.no_grad():
            out = model(**inputs)

        assert out["trend_rt_pred_24"].shape == (1, NUM_HOURS)
        assert out["delta_pred_24"].shape == (1, NUM_HOURS)
        assert out["confidence_24"].shape == (1, NUM_HOURS)
        assert torch.isfinite(out["trend_rt_pred_24"]).all()
