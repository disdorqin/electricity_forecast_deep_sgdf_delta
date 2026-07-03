"""Tests for SmoothnessLoss (V2 day-level version).

SmoothnessLoss penalises hour-to-hour jumps in a predicted delta sequence.
  loss = mean((delta[t+1] - delta[t])^2)   for t = 0..T-2

Covers:
  - Constant sequence produces loss = 0
  - Linear ramp produces the expected analytic value
  - Random sequence produces loss > 0
  - Gradients flow through the loss
"""
from __future__ import annotations

import pytest
import torch

from models.deep_sgdf_delta.losses import SmoothnessLoss


# ── Test: constant sequence => loss = 0 ──────────────────────────────


class TestSmoothnessConstantSequence:
    """A constant prediction sequence has zero hour-to-hour jumps."""

    def test_constant_zeros(self):
        loss_fn = SmoothnessLoss()
        seq = torch.zeros(2, 24)  # [B, 24] all zeros
        loss = loss_fn(seq)
        assert loss.item() == pytest.approx(0.0, abs=1e-8)

    def test_constant_nonzero(self):
        loss_fn = SmoothnessLoss()
        seq = torch.full((3, 24), 42.0)  # [B, 24] all 42
        loss = loss_fn(seq)
        assert loss.item() == pytest.approx(0.0, abs=1e-8)

    def test_constant_negative(self):
        loss_fn = SmoothnessLoss()
        seq = torch.full((2, 24), -15.5)
        loss = loss_fn(seq)
        assert loss.item() == pytest.approx(0.0, abs=1e-8)


# ── Test: linear ramp => expected value ───────────────────────────────


class TestSmoothnessLinearRamp:
    """A linear ramp delta[t] = t has constant first differences of 1.

    diff = [1, 1, 1, ..., 1]  (23 values for T=24)
    loss = mean(1^2) = 1.0
    """

    def test_unit_ramp(self):
        loss_fn = SmoothnessLoss()
        # delta[t] = t for t in 0..23
        seq = torch.arange(24, dtype=torch.float32).unsqueeze(0)  # [1, 24]
        loss = loss_fn(seq)
        # All diffs are 1, so mean(1^2) = 1.0
        assert loss.item() == pytest.approx(1.0, abs=1e-6)

    def test_scaled_ramp(self):
        loss_fn = SmoothnessLoss()
        # delta[t] = 3*t for t in 0..23
        seq = (torch.arange(24, dtype=torch.float32) * 3.0).unsqueeze(0)
        loss = loss_fn(seq)
        # All diffs are 3, so mean(3^2) = 9.0
        assert loss.item() == pytest.approx(9.0, abs=1e-6)

    def test_ramp_batch(self):
        loss_fn = SmoothnessLoss()
        # Batch of 2: one unit ramp, one double ramp
        ramp1 = torch.arange(24, dtype=torch.float32)          # diff=1
        ramp2 = torch.arange(24, dtype=torch.float32) * 2.0    # diff=2
        seq = torch.stack([ramp1, ramp2])                       # [2, 24]
        loss = loss_fn(seq)
        # mean of (23 ones^2 + 23 twos^2) = mean(23 + 92) = 115 / 46 = 2.5
        assert loss.item() == pytest.approx(2.5, abs=1e-6)


# ── Test: random sequence => loss > 0 ─────────────────────────────────


class TestSmoothnessRandomSequence:
    """A random sequence should have non-zero smoothness loss."""

    def test_random_positive_loss(self):
        loss_fn = SmoothnessLoss()
        torch.manual_seed(42)
        seq = torch.randn(4, 24)
        loss = loss_fn(seq)
        assert loss.item() > 0.0

    def test_random_different_seeds_differ(self):
        loss_fn = SmoothnessLoss()
        torch.manual_seed(0)
        seq1 = torch.randn(2, 24)
        loss1 = loss_fn(seq1).item()

        torch.manual_seed(99)
        seq2 = torch.randn(2, 24)
        loss2 = loss_fn(seq2).item()

        # Very unlikely to be exactly equal
        assert loss1 != loss2


# ── Test: gradient flows through smoothness loss ──────────────────────


class TestSmoothnessGradient:
    """Verify that gradients propagate through the smoothness loss."""

    def test_gradient_exists(self):
        loss_fn = SmoothnessLoss()
        seq = torch.randn(2, 24, requires_grad=True)
        loss = loss_fn(seq)
        loss.backward()
        assert seq.grad is not None
        assert seq.grad.shape == seq.shape

    def test_gradient_nonzero_for_nonconstant(self):
        loss_fn = SmoothnessLoss()
        seq = torch.arange(24, dtype=torch.float32).unsqueeze(0)
        seq.requires_grad_(True)
        loss = loss_fn(seq)
        loss.backward()
        # Gradient should be non-zero (the ramp is non-constant)
        assert seq.grad.abs().sum().item() > 0.0

    def test_gradient_zero_for_constant(self):
        loss_fn = SmoothnessLoss()
        seq = torch.full((1, 24), 5.0, requires_grad=True)
        loss = loss_fn(seq)
        loss.backward()
        # For a constant sequence, all diffs are 0, so gradient is 0
        assert seq.grad.abs().sum().item() == pytest.approx(0.0, abs=1e-8)

    def test_gradient_analytic_ramp(self):
        """For delta = [0, 1, 2, ..., 23], d(loss)/d(delta[t]) has known values.

        loss = (1/T-1) * sum_{t=0}^{T-2} (delta[t+1] - delta[t])^2
        d(loss)/d(delta[0]) = -2*(delta[1]-delta[0]) / (T-1) = -2*1/23
        d(loss)/d(delta[23]) = 2*(delta[23]-delta[22]) / (T-1) = 2*1/23
        d(loss)/d(delta[t]) = 2*(2*delta[t] - delta[t-1] - delta[t+1]) / (T-1) for 0<t<T-1
        For a unit ramp: 2*delta[t] - delta[t-1] - delta[t+1] = 2*t - (t-1) - (t+1) = 0
        So interior gradients are 0, and boundary gradients are +/-2/23.
        """
        loss_fn = SmoothnessLoss()
        seq = torch.arange(24, dtype=torch.float32).unsqueeze(0)
        seq.requires_grad_(True)
        loss = loss_fn(seq)
        loss.backward()

        grad = seq.grad[0]
        # Interior gradients should be ~0 for a linear ramp
        for t in range(1, 23):
            assert grad[t].item() == pytest.approx(0.0, abs=1e-5), (
                f"Interior gradient at t={t} should be 0, got {grad[t].item()}"
            )
        # Boundary gradients
        assert grad[0].item() == pytest.approx(-2.0 / 23.0, abs=1e-5)
        assert grad[23].item() == pytest.approx(2.0 / 23.0, abs=1e-5)


# ── Test: single-element sequence ─────────────────────────────────────


class TestSmoothnessEdgeCases:
    """Edge cases for smoothness loss."""

    def test_single_hour_returns_zero(self):
        """A sequence of length 1 has no diffs, so loss should be 0."""
        loss_fn = SmoothnessLoss()
        seq = torch.randn(2, 1)  # [B, 1]
        loss = loss_fn(seq)
        assert loss.item() == pytest.approx(0.0, abs=1e-8)

    def test_two_hours(self):
        """A sequence of length 2 has exactly one diff."""
        loss_fn = SmoothnessLoss()
        seq = torch.tensor([[3.0, 7.0]])  # diff = 4
        loss = loss_fn(seq)
        # mean(4^2) = 16
        assert loss.item() == pytest.approx(16.0, abs=1e-6)
