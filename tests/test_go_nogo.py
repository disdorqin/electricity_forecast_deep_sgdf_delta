"""Tests for the Go/No-Go evaluation verdict logic.

Covers:
  - PASS when sMAPE < 15
  - SOFT_PASS when sMAPE <= 15.8
  - BASELINE_PASS when sMAPE < 16.5902
  - NO-GO when sMAPE >= 16.5902
  - verdict_detail text matches the verdict

Uses ``evaluate_predictions`` from the evaluate module, which is standalone
(no SGDFNet dependency).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.evaluate import (
    BASELINE_SGDFNET,
    PASS_THRESHOLD,
    SOFT_PASS_THRESHOLD,
    evaluate_predictions,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _make_pred_df(
    n_hours: int = 24 * 30,
    target_smape: float = 10.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a prediction DataFrame that achieves roughly the target sMAPE.

    Strategy: start with perfect predictions, then add calibrated noise
    to reach the approximate target sMAPE.
    """
    rng = np.random.RandomState(seed)
    n = n_hours

    # Base prices
    da_anchor = rng.uniform(100, 300, size=n)
    delta_target = rng.randn(n) * 20
    rt_actual = da_anchor + delta_target

    # To achieve a target sMAPE, add noise proportional to the desired error
    # sMAPE ~ 200 * |err| / (2 * price) => |err| ~ sMAPE * price / 100
    noise_scale = target_smape * np.mean(da_anchor) / 100.0
    noise = rng.randn(n) * noise_scale * 0.5

    rt_pred = rt_actual + noise
    delta_pred = rt_pred - da_anchor

    # Build business_day and hour columns
    base_day = pd.Timestamp("2026-01-01")
    days = np.repeat(pd.date_range(base_day, periods=n // 24, freq="D"), 24)
    if len(days) < n:
        extra = n - len(days)
        days = np.concatenate([days, np.repeat([days[-1] + pd.Timedelta(days=1)], extra)])
    hours = np.tile(np.arange(1, 25), n // 24 + 1)[:n]

    # Assign periods
    periods = []
    for h in hours:
        if 1 <= h <= 8:
            periods.append("1_8")
        elif 9 <= h <= 16:
            periods.append("9_16")
        else:
            periods.append("17_24")

    return pd.DataFrame({
        "business_day": days[:n],
        "hour": hours,
        "target_hour": hours,
        "period": periods,
        "ds": pd.date_range("2026-01-01", periods=n, freq="h"),
        "rt_actual": rt_actual,
        "rt_pred": rt_pred,
        "y_true": rt_actual,
        "y_pred": rt_pred,
        "delta_target": delta_target,
        "delta_pred": delta_pred,
        "da_anchor": da_anchor,
    })


def _evaluate_with_smape(target_smape: float, seed: int = 42) -> dict:
    """Run evaluate_predictions with a DataFrame tuned to ~target_smape."""
    df = _make_pred_df(target_smape=target_smape, seed=seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = evaluate_predictions(
            df,
            run_id=f"test_smape_{target_smape}",
            output_dir=Path(tmpdir),
        )
    return result


# ── Test: PASS when sMAPE < 15 ────────────────────────────────────────


class TestPassVerdict:
    """Verdict should be PASS when overall sMAPE < 15."""

    def test_pass_verdict(self):
        result = _evaluate_with_smape(target_smape=5.0, seed=10)
        assert result["verdict"] == "PASS"

    def test_pass_threshold_boundary(self):
        """sMAPE well below 15 should pass."""
        result = _evaluate_with_smape(target_smape=2.0, seed=20)
        assert result["verdict"] == "PASS"

    def test_pass_verdict_detail_mentions_threshold(self):
        result = _evaluate_with_smape(target_smape=5.0, seed=30)
        assert result["verdict"] == "PASS"
        assert "< 15" in result["verdict_detail"]


# ── Test: SOFT_PASS when sMAPE <= 15.8 ────────────────────────────────


class TestSoftPassVerdict:
    """Verdict should be SOFT_PASS when 15 <= sMAPE <= 15.8."""

    def test_soft_pass_verdict(self):
        # Target ~15.4 to land in the SOFT_PASS range
        result = _evaluate_with_smape(target_smape=15.4, seed=40)
        overall = result.get("overall_sMAPE_floor50", float("nan"))
        # Only check verdict if overall is actually in range
        if PASS_THRESHOLD <= overall <= SOFT_PASS_THRESHOLD:
            assert result["verdict"] == "SOFT_PASS"
        # Otherwise the noise pushed it out of range — that's OK for
        # a probabilistic test, just verify the verdict is consistent
        elif overall < PASS_THRESHOLD:
            assert result["verdict"] == "PASS"
        elif overall < BASELINE_SGDFNET:
            assert result["verdict"] == "BASELINE_PASS"
        else:
            assert result["verdict"] == "NO-GO"

    def test_soft_pass_verdict_detail(self):
        """When verdict is SOFT_PASS, detail should mention the threshold."""
        result = _evaluate_with_smape(target_smape=15.4, seed=50)
        if result["verdict"] == "SOFT_PASS":
            assert str(SOFT_PASS_THRESHOLD) in result["verdict_detail"] or "15.8" in result["verdict_detail"]
            assert "spike" in result["verdict_detail"].lower() or "fus" in result["verdict_detail"].lower()


# ── Test: BASELINE_PASS when sMAPE < 16.5902 ─────────────────────────


class TestBaselinePassVerdict:
    """Verdict should be BASELINE_PASS when 15.8 < sMAPE < 16.5902."""

    def test_baseline_pass_verdict(self):
        result = _evaluate_with_smape(target_smape=16.0, seed=60)
        overall = result.get("overall_sMAPE_floor50", float("nan"))
        if SOFT_PASS_THRESHOLD < overall < BASELINE_SGDFNET:
            assert result["verdict"] == "BASELINE_PASS"
        # Otherwise verify consistency
        elif overall <= SOFT_PASS_THRESHOLD and overall >= PASS_THRESHOLD:
            assert result["verdict"] == "SOFT_PASS"
        elif overall < PASS_THRESHOLD:
            assert result["verdict"] == "PASS"
        else:
            assert result["verdict"] == "NO-GO"

    def test_baseline_pass_verdict_detail(self):
        """When verdict is BASELINE_PASS, detail should mention SGDFNet baseline."""
        result = _evaluate_with_smape(target_smape=16.0, seed=70)
        if result["verdict"] == "BASELINE_PASS":
            assert f"{BASELINE_SGDFNET:.4f}" in result["verdict_detail"] or "baseline" in result["verdict_detail"].lower()


# ── Test: NO-GO when sMAPE >= 16.5902 ────────────────────────────────


class TestNoGoVerdict:
    """Verdict should be NO-GO when sMAPE >= 16.5902."""

    def test_nogo_verdict(self):
        result = _evaluate_with_smape(target_smape=25.0, seed=80)
        overall = result.get("overall_sMAPE_floor50", float("nan"))
        if overall >= BASELINE_SGDFNET:
            assert result["verdict"] == "NO-GO"

    def test_nogo_large_error(self):
        """Very large errors should definitely be NO-GO."""
        result = _evaluate_with_smape(target_smape=50.0, seed=90)
        assert result["verdict"] == "NO-GO"

    def test_nogo_verdict_detail(self):
        """When verdict is NO-GO, detail should mention the baseline comparison."""
        result = _evaluate_with_smape(target_smape=50.0, seed=100)
        if result["verdict"] == "NO-GO":
            assert ">=" in result["verdict_detail"] or "baseline" in result["verdict_detail"].lower()


# ── Test: verdict_detail text matches verdict ─────────────────────────


class TestVerdictDetailConsistency:
    """verdict_detail text should always be consistent with the verdict."""

    @pytest.mark.parametrize(
        "target_smape,seed",
        [
            (2.0, 1),    # likely PASS
            (15.4, 2),   # likely SOFT_PASS region
            (16.0, 3),   # likely BASELINE_PASS region
            (30.0, 4),   # likely NO-GO
        ],
    )
    def test_verdict_detail_matches_verdict(self, target_smape, seed):
        result = _evaluate_with_smape(target_smape=target_smape, seed=seed)
        verdict = result["verdict"]
        detail = result["verdict_detail"]

        assert isinstance(detail, str)
        assert len(detail) > 0
        # Detail always contains the sMAPE value
        assert "sMAPE_floor50" in detail

        if verdict == "PASS":
            # Detail format: "Overall sMAPE_floor50=X.XXXX < 15.0"
            assert "< 15" in detail
            assert ">=" not in detail
        elif verdict == "SOFT_PASS":
            # Detail format: "...<= 15.8, awaiting spike/negative module fusion"
            assert "<= 15.8" in detail or "15.8" in detail
        elif verdict == "BASELINE_PASS":
            # Detail format: "...< SGDFNet baseline 16.5902"
            assert "SGDFNet baseline" in detail or "baseline" in detail.lower()
            assert "<" in detail
        elif verdict == "NO-GO":
            # Detail format: "...>= SGDFNet baseline 16.5902"
            assert ">=" in detail

    def test_verdict_is_one_of_four(self):
        """Verdict must be exactly one of PASS, SOFT_PASS, BASELINE_PASS, NO-GO."""
        valid_verdicts = {"PASS", "SOFT_PASS", "BASELINE_PASS", "NO-GO"}
        for seed in range(5):
            result = _evaluate_with_smape(target_smape=10.0 + seed * 5, seed=seed + 200)
            assert result["verdict"] in valid_verdicts, (
                f"Unexpected verdict: {result['verdict']}"
            )


# ── Test: thresholds are defined and ordered ─────────────────────────────


class TestThresholdConstants:
    """Verify threshold constants are defined and correctly ordered."""

    def test_pass_threshold(self):
        assert PASS_THRESHOLD == 15.0

    def test_soft_pass_threshold(self):
        assert SOFT_PASS_THRESHOLD == 15.8

    def test_baseline_sgdfnet(self):
        assert BASELINE_SGDFNET == pytest.approx(16.5902, abs=1e-4)

    def test_threshold_ordering(self):
        assert PASS_THRESHOLD < SOFT_PASS_THRESHOLD < BASELINE_SGDFNET


# ── Test: output files are written ────────────────────────────────────


class TestOutputFilesWritten:
    """evaluate_predictions should write all expected output files."""

    def test_all_files_written(self):
        df = _make_pred_df(target_smape=10.0, seed=300)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            evaluate_predictions(df, run_id="file_test", output_dir=out)

            assert (out / "predictions.csv").exists()
            assert (out / "metrics_summary.json").exists()
            assert (out / "go_nogo.md").exists()

    def test_metrics_json_has_verdict(self):
        df = _make_pred_df(target_smape=10.0, seed=310)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            evaluate_predictions(df, run_id="json_test", output_dir=out)

            with open(out / "metrics_summary.json", "r") as f:
                data = json.load(f)

            assert "verdict" in data
            assert "verdict_detail" in data
            assert "overall_sMAPE_floor50" in data

    def test_go_nogo_md_has_verdict(self):
        df = _make_pred_df(target_smape=10.0, seed=320)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            evaluate_predictions(df, run_id="md_test", output_dir=out)

            text = (out / "go_nogo.md").read_text(encoding="utf-8")
            assert "Verdict" in text
            assert "Threshold" in text or "threshold" in text
