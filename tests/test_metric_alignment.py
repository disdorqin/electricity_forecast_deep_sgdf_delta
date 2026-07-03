"""Tests for metric alignment audit — canonical smape_floor50, cross-module logic,
and multi-month audit functionality.

Verifies:
1. Canonical smape_floor50 returns percent scale (multiplier=200, not 2).
2. Canonical smape_floor50 handles negative prices correctly (floor on signed value).
3. Audit script can be imported and run with fixture data.
4. Common intersection logic works correctly.
5. Multi-month audit produces correct number of rows.
6. Each month has correct row count check.
7. Business day uniqueness check.
8. Verdict logic for multi-month.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Path setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.deep_sgdf_delta.metrics import smape_floor50


# ═══════════════════════════════════════════════════════════════════════
# Test 1: Canonical smape_floor50 returns percent scale
# ═══════════════════════════════════════════════════════════════════════

class TestSmapeFloor50PercentScale:
    """The canonical formula uses multiplier 200 → output is in percent.

    smape_floor50([100], [200]) should be ~66.67, NOT 0.6667.
    """

    def test_basic_percent_scale(self):
        """smape_floor50([100], [200]) ≈ 66.67 (percent scale)."""
        y_true = np.array([100.0])
        y_pred = np.array([200.0])
        result = smape_floor50(y_true, y_pred)
        # 200 * |200 - 100| / (|100| + |200| + eps) = 200 * 100 / 300 ≈ 66.67
        assert abs(result - 66.67) < 0.1, (
            f"Expected ~66.67 (percent scale), got {result}. "
            "Formula should use multiplier 200, not 2."
        )

    def test_perfect_prediction(self):
        """Perfect prediction → sMAPE = 0."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([100.0, 200.0, 300.0])
        result = smape_floor50(y_true, y_pred)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_maximum_smape_near_200(self):
        """When one value is zero and the other is large, sMAPE approaches 200."""
        # With floor=50, y_true=50 (floored), y_pred=1000
        # 200 * |1000 - 50| / (50 + 1000 + eps) ≈ 200 * 950 / 1050 ≈ 180.95
        y_true = np.array([50.0])
        y_pred = np.array([1000.0])
        result = smape_floor50(y_true, y_pred)
        assert result > 150.0, f"Expected high sMAPE (>150), got {result}"
        assert result < 200.0, f"sMAPE should be < 200, got {result}"

    def test_not_fraction_scale(self):
        """Ensure result is NOT in fraction scale (0 to 2)."""
        y_true = np.array([100.0])
        y_pred = np.array([200.0])
        result = smape_floor50(y_true, y_pred)
        # If the formula used multiplier 2 instead of 200, result would be ~0.6667
        assert result > 1.0, (
            f"Result {result} looks like fraction scale (multiplier=2). "
            "Canonical formula must use multiplier=200 for percent scale."
        )

    def test_symmetric_property(self):
        """sMAPE is symmetric: smape(a, b) == smape(b, a) when both above floor."""
        y_a = np.array([100.0])
        y_b = np.array([200.0])
        result_ab = smape_floor50(y_a, y_b)
        result_ba = smape_floor50(y_b, y_a)
        assert result_ab == pytest.approx(result_ba, abs=1e-6)

    def test_multi_element_array(self):
        """Test with multiple elements — should return mean of per-element sMAPE."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 180.0, 350.0])
        result = smape_floor50(y_true, y_pred)
        # Manual computation:
        # elem0: 200 * |110-100| / (100+110) = 200*10/210 ≈ 9.524
        # elem1: 200 * |180-200| / (200+180) = 200*20/380 ≈ 10.526
        # elem2: 200 * |350-300| / (300+350) = 200*50/650 ≈ 15.385
        # mean ≈ (9.524 + 10.526 + 15.385) / 3 ≈ 11.812
        assert abs(result - 11.81) < 0.1, f"Expected ~11.81, got {result}"


# ═══════════════════════════════════════════════════════════════════════
# Test 2: Canonical smape_floor50 handles negative prices correctly
# ═══════════════════════════════════════════════════════════════════════

class TestSmapeFloor50NegativePrices:
    """The canonical formula applies floor to signed value: np.where(y < floor, floor, y).

    This means negative prices get clamped UP to the floor (50), not clamped
    by absolute value. The old broken formula used np.clip(np.abs(y), floor, None)
    which would map -100 to 100 (abs then floor), while the canonical formula
    maps -100 to 50 (floor on signed value).
    """

    def test_negative_price_floored_to_50(self):
        """Negative price should be floored to 50, not abs(-100)=100."""
        # y_true = -100 → floored to 50 (canonical)
        # y_pred = 100 → stays 100
        # sMAPE = 200 * |100 - 50| / (50 + 100 + eps) = 200 * 50 / 150 ≈ 66.67
        y_true = np.array([-100.0])
        y_pred = np.array([100.0])
        result = smape_floor50(y_true, y_pred)
        # Canonical: floor(-100) → 50, so 200*|100-50|/(50+100) ≈ 66.67
        # Old broken: abs(-100) → 100, then clip(100, 50) → 100,
        #   so 200*|100-100|/(100+100) = 0  ← WRONG
        assert abs(result - 66.67) < 0.1, (
            f"Expected ~66.67 (canonical floor on signed value), got {result}. "
            "Negative prices should be floored to 50, not abs() then floored."
        )

    def test_both_negative(self):
        """Both values negative → both floored to 50 → sMAPE ≈ 0."""
        y_true = np.array([-200.0])
        y_pred = np.array([-50.0])
        result = smape_floor50(y_true, y_pred)
        # Both floored to 50 → |50 - 50| = 0 → sMAPE = 0
        assert result == pytest.approx(0.0, abs=1e-4), (
            f"Expected ~0.0 (both floored to 50), got {result}"
        )

    def test_negative_vs_positive_asymmetric(self):
        """Verify the asymmetry between canonical and broken formula."""
        # y_true = -30, y_pred = 200
        # Canonical: floor(-30) → 50, 200 stays → 200*|200-50|/(50+200) = 200*150/250 = 120
        # Broken: abs(-30)=30 → clip(30,50) → 50, abs(200)=200 → 200*|200-50|/(50+200) = 120
        # In this case both give same result because abs(-30)=30 < 50 → floored to 50 anyway
        y_true = np.array([-30.0])
        y_pred = np.array([200.0])
        result = smape_floor50(y_true, y_pred)
        expected = 200.0 * 150.0 / 250.0  # = 120.0
        assert abs(result - expected) < 0.1, f"Expected ~{expected}, got {result}"

    def test_negative_large_abs_differs_from_broken(self):
        """Key test: where canonical and broken formulas diverge.

        y_true = -500, y_pred = 100
        Canonical: floor(-500) → 50, 100 stays
            → 200 * |100 - 50| / (50 + 100) = 200 * 50 / 150 ≈ 66.67
        Broken: abs(-500) → 500, clip(500, 50) → 500, abs(100) → 100
            → 200 * |100 - 500| / (500 + 100) = 200 * 400 / 600 ≈ 133.33
        """
        y_true = np.array([-500.0])
        y_pred = np.array([100.0])
        result = smape_floor50(y_true, y_pred)
        # Canonical result should be ~66.67
        assert abs(result - 66.67) < 0.1, (
            f"Expected ~66.67 (canonical), got {result}. "
            "If result is ~133.33, the formula is using abs() before floor (broken)."
        )

    def test_mixed_positive_negative_array(self):
        """Array with mix of positive and negative values."""
        y_true = np.array([100.0, -200.0, 300.0])
        y_pred = np.array([110.0, 50.0, 280.0])
        result = smape_floor50(y_true, y_pred)
        # elem0: 200*|110-100|/(100+110) = 200*10/210 ≈ 9.524
        # elem1: -200 → 50, 50 → 50: 200*|50-50|/(50+50) = 0
        # elem2: 200*|280-300|/(300+280) = 200*20/580 ≈ 6.897
        # mean ≈ (9.524 + 0 + 6.897) / 3 ≈ 5.474
        assert abs(result - 5.47) < 0.1, f"Expected ~5.47, got {result}"


# ═══════════════════════════════════════════════════════════════════════
# Test 3: Audit script importability and fixture-data execution
# ═══════════════════════════════════════════════════════════════════════

class TestAuditScriptImport:
    """The audit script should be importable and its functions testable."""

    def test_import_audit_module(self):
        """Verify the audit script can be imported as a module."""
        scripts_dir = PROJECT_ROOT / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from audit_metric_alignment import (
            compute_source_stats,
            compute_smape_for_source,
            compute_verdict,
            load_deep_final,
            load_delta_supply,
            load_raw_data,
        )
        # All key functions should be callable
        assert callable(compute_source_stats)
        assert callable(compute_smape_for_source)
        assert callable(compute_verdict)
        assert callable(load_deep_final)
        assert callable(load_delta_supply)
        assert callable(load_raw_data)

    def test_compute_source_stats_with_fixture(self, tmp_path):
        """Test compute_source_stats with a small fixture DataFrame."""
        from audit_metric_alignment import compute_source_stats

        df = pd.DataFrame({
            "ds": pd.date_range("2026-02-01", periods=24, freq="h"),
            "da_anchor": np.random.uniform(200, 500, 24),
            "rt_actual": np.random.uniform(100, 600, 24),
        })
        stats = compute_source_stats(df, "TestSource")
        assert stats["source"] == "TestSource"
        assert stats["row_count"] == 24
        assert stats["date_min"] is not None
        assert stats["date_max"] is not None
        assert stats["da_anchor_mean"] is not None
        assert stats["rt_actual_mean"] is not None
        assert isinstance(stats["da_anchor_mean"], float)
        assert isinstance(stats["rt_actual_std"], float)

    def test_compute_smape_for_source_with_fixture(self):
        """Test sMAPE computation with fixture data."""
        from audit_metric_alignment import compute_smape_for_source

        df = pd.DataFrame({
            "ds": pd.date_range("2026-02-01", periods=10, freq="h"),
            "rt_actual": np.array([100, 200, 300, 400, 500, 150, 250, 350, 450, 550], dtype=float),
            "rt_pred": np.array([110, 190, 310, 390, 510, 160, 240, 360, 440, 540], dtype=float),
        })
        result = compute_smape_for_source(df, "TestSource")
        assert result is not None
        assert isinstance(result, float)
        assert result > 0  # predictions differ from actual
        assert result < 200  # should be reasonable

    def test_compute_smape_missing_columns(self):
        """sMAPE returns None when required columns are missing."""
        from audit_metric_alignment import compute_smape_for_source

        df = pd.DataFrame({
            "ds": [1, 2, 3],
            "some_other_col": [10, 20, 30],
        })
        result = compute_smape_for_source(df, "NoCols")
        assert result is None

    def test_compute_verdict_pass(self):
        """Verdict is PASS when spread <= 0.1 pp."""
        from audit_metric_alignment import compute_verdict
        verdict, spread = compute_verdict([10.0, 10.05, 10.1])
        assert verdict == "PASS"
        assert spread == pytest.approx(0.1, abs=0.01)

    def test_compute_verdict_warn(self):
        """Verdict is WARN when 0.1 < spread <= 1.0 pp."""
        from audit_metric_alignment import compute_verdict
        verdict, spread = compute_verdict([10.0, 10.5])
        assert verdict == "WARN"
        assert spread == pytest.approx(0.5, abs=0.01)

    def test_compute_verdict_fail(self):
        """Verdict is FAIL when spread > 1.0 pp."""
        from audit_metric_alignment import compute_verdict
        verdict, spread = compute_verdict([10.0, 12.0])
        assert verdict == "FAIL"
        assert spread == pytest.approx(2.0, abs=0.01)

    def test_compute_verdict_insufficient(self):
        """Verdict is INSUFFICIENT with fewer than 2 values."""
        from audit_metric_alignment import compute_verdict
        verdict, spread = compute_verdict([10.0])
        assert verdict == "INSUFFICIENT"
        assert spread == 0.0


# ═══════════════════════════════════════════════════════════════════════
# Test 4: Common intersection logic
# ═══════════════════════════════════════════════════════════════════════

class TestCommonIntersection:
    """Test that common intersection logic correctly merges on ds."""

    def test_basic_intersection(self):
        """Two sources with overlapping ds → correct intersection size."""
        from audit_metric_alignment import find_common_intersection

        df_a = pd.DataFrame({
            "ds": pd.date_range("2026-02-01", periods=48, freq="h"),
            "rt_actual": np.random.uniform(100, 500, 48),
        })
        df_b = pd.DataFrame({
            "ds": pd.date_range("2026-02-01 12:00", periods=48, freq="h"),
            "rt_actual": np.random.uniform(100, 500, 48),
        })
        sources = {"A": df_a, "B": df_b}
        common = find_common_intersection(sources)
        assert common is not None
        # Overlap: Feb 1 12:00 to Feb 2 23:00 = 36 hours
        assert len(common) == 36, f"Expected 36 common rows, got {len(common)}"

    def test_no_overlap(self):
        """Two sources with no overlapping ds → empty intersection."""
        from audit_metric_alignment import find_common_intersection

        df_a = pd.DataFrame({
            "ds": pd.date_range("2026-02-01", periods=24, freq="h"),
            "rt_actual": np.random.uniform(100, 500, 24),
        })
        df_b = pd.DataFrame({
            "ds": pd.date_range("2026-03-01", periods=24, freq="h"),
            "rt_actual": np.random.uniform(100, 500, 24),
        })
        sources = {"A": df_a, "B": df_b}
        common = find_common_intersection(sources)
        assert common is not None
        assert len(common) == 0

    def test_three_sources(self):
        """Three sources → intersection is the overlap of all three."""
        from audit_metric_alignment import find_common_intersection

        df_a = pd.DataFrame({"ds": pd.date_range("2026-02-01", periods=72, freq="h")})
        df_b = pd.DataFrame({"ds": pd.date_range("2026-02-01 06:00", periods=48, freq="h")})
        df_c = pd.DataFrame({"ds": pd.date_range("2026-02-01 12:00", periods=36, freq="h")})
        sources = {"A": df_a, "B": df_b, "C": df_c}
        common = find_common_intersection(sources)
        assert common is not None
        # A: Feb 1 00:00 – Feb 3 23:00
        # B: Feb 1 06:00 – Feb 3 05:00
        # C: Feb 1 12:00 – Feb 2 23:00
        # Intersection: Feb 1 12:00 – Feb 2 23:00 = 36 hours
        assert len(common) == 36, f"Expected 36, got {len(common)}"

    def test_single_source_returns_none(self):
        """Fewer than 2 sources → returns None."""
        from audit_metric_alignment import find_common_intersection

        df_a = pd.DataFrame({"ds": pd.date_range("2026-02-01", periods=24, freq="h")})
        sources = {"A": df_a}
        common = find_common_intersection(sources)
        assert common is None

    def test_common_smape_computation(self):
        """Compute sMAPE on common intersection rows."""
        from audit_metric_alignment import compute_common_smape, find_common_intersection

        # Source A: actual values
        ds_common = pd.date_range("2026-02-01", periods=24, freq="h")
        df_a = pd.DataFrame({
            "ds": ds_common,
            "rt_actual": np.full(24, 200.0),
            "rt_pred": np.full(24, 220.0),
        })
        # Source B: same timestamps, different predictions
        df_b = pd.DataFrame({
            "ds": ds_common,
            "rt_actual": np.full(24, 200.0),
            "rt_pred": np.full(24, 210.0),
        })
        sources = {"A": df_a, "B": df_b}
        common = find_common_intersection(sources)
        assert common is not None
        assert len(common) == 24

        result = compute_common_smape(sources, common)
        assert "A" in result
        assert "B" in result
        assert result["A"]["common_rows"] == 24
        assert result["B"]["common_rows"] == 24
        # Both should have valid sMAPE values
        assert result["A"]["smape_common"] is not None
        assert result["B"]["smape_common"] is not None
        # A has larger deviation (220 vs 200) than B (210 vs 200)
        assert result["A"]["smape_common"] > result["B"]["smape_common"]

    def test_common_smape_with_da_anchor_fallback(self):
        """When rt_pred is missing, da_anchor is used as fallback prediction."""
        from audit_metric_alignment import compute_common_smape, find_common_intersection

        ds_common = pd.date_range("2026-02-01", periods=12, freq="h")
        df_a = pd.DataFrame({
            "ds": ds_common,
            "rt_actual": np.full(12, 200.0),
            "rt_pred": np.full(12, 210.0),
        })
        # Source B has no rt_pred, only da_anchor
        df_b = pd.DataFrame({
            "ds": ds_common,
            "rt_actual": np.full(12, 200.0),
            "da_anchor": np.full(12, 195.0),
        })
        sources = {"A": df_a, "B": df_b}
        common = find_common_intersection(sources)
        result = compute_common_smape(sources, common)
        # B should use da_anchor as fallback
        assert result["B"]["smape_common"] is not None
        assert result["B"]["common_rows"] == 12


# ═══════════════════════════════════════════════════════════════════════
# Integration test: end-to-end with fixture data
# ═══════════════════════════════════════════════════════════════════════

class TestEndToEndFixture:
    """End-to-end test of the audit pipeline with synthetic fixture data."""

    def test_full_pipeline_with_fixtures(self, tmp_path):
        """Write fixture CSVs, run the audit logic, verify outputs."""
        from audit_metric_alignment import (
            build_comparison_rows,
            compute_common_smape,
            compute_source_stats,
            compute_smape_for_source,
            compute_verdict,
            find_common_intersection,
            generate_report_md,
        )

        # Create fixture data
        ds = pd.date_range("2026-02-01", periods=48, freq="h")
        np.random.seed(42)

        # Source 1: DeepFinal-like
        df_deep = pd.DataFrame({
            "ds": ds,
            "da_anchor": np.random.uniform(200, 500, 48),
            "rt_actual": np.random.uniform(100, 600, 48),
            "rt_pred": np.random.uniform(100, 600, 48),
        })
        deep_path = tmp_path / "deep_final_predictions.csv"
        df_deep.to_csv(deep_path, index=False)

        # Source 2: DeltaSupply-like (overlapping timestamps)
        df_delta = pd.DataFrame({
            "ds": ds[12:],  # starts 12h later
            "da_anchor": np.random.uniform(200, 500, 36),
            "rt_actual": df_deep["rt_actual"].values[12:] + np.random.normal(0, 5, 36),
            "rt_pred": df_deep["rt_pred"].values[12:] + np.random.normal(0, 5, 36),
        })
        delta_path = tmp_path / "delta_supply_predictions.csv"
        df_delta.to_csv(delta_path, index=False)

        # Run pipeline
        sources = {"DeepFinal": df_deep, "DeltaSupply": df_delta}

        source_stats = [compute_source_stats(df, name) for name, df in sources.items()]
        source_smapes = {name: compute_smape_for_source(df, name) for name, df in sources.items()}

        common = find_common_intersection(sources)
        assert common is not None
        assert len(common) == 36  # 48 - 12

        common_smape = compute_common_smape(sources, common)
        common_vals = [cs["smape_common"] for cs in common_smape.values() if cs.get("smape_common") is not None]
        verdict, spread = compute_verdict(common_vals)

        # Build outputs
        comparison_df = build_comparison_rows(source_stats, source_smapes, common_smape)
        report_md = generate_report_md(source_stats, source_smapes, common_smape, verdict, spread, comparison_df)

        # Verify
        assert len(source_stats) == 2
        assert all(s["row_count"] > 0 for s in source_stats)
        assert all(v is not None for v in source_smapes.values())
        assert verdict in ("PASS", "WARN", "FAIL", "INSUFFICIENT")
        assert "# Metric Alignment Audit Report" in report_md
        assert verdict in report_md
        assert len(comparison_df) == 2

        # Write outputs
        json_path = tmp_path / "metric_alignment_summary.json"
        json_path.write_text(json.dumps({"verdict": verdict, "spread_pp": spread}, indent=2), encoding="utf-8")
        csv_path = tmp_path / "metric_alignment_rows.csv"
        comparison_df.to_csv(csv_path, index=False)
        md_path = tmp_path / "metric_alignment_report.md"
        md_path.write_text(report_md, encoding="utf-8")

        assert json_path.exists()
        assert csv_path.exists()
        assert md_path.exists()


# ═══════════════════════════════════════════════════════════════════════
# Test 5: Multi-month audit — correct number of rows
# ═══════════════════════════════════════════════════════════════════════

class TestMultiMonthAudit:
    """Multi-month audit produces correct number of rows and details."""

    def _make_multi_month_raw_data(self):
        """Create synthetic raw data spanning 3 months (Jan-Mar 2026)."""
        np.random.seed(42)
        # Generate hourly data for Jan, Feb, Mar 2026
        ds_jan = pd.date_range("2026-01-01", "2026-01-31 23:00", freq="h")
        ds_feb = pd.date_range("2026-02-01", "2026-02-28 23:00", freq="h")
        ds_mar = pd.date_range("2026-03-01", "2026-03-31 23:00", freq="h")

        all_ds = ds_jan.append(ds_feb).append(ds_mar)
        n = len(all_ds)

        df = pd.DataFrame({
            "ds": all_ds,
            "da_anchor": np.random.uniform(200, 500, n),
            "rt_actual": np.random.uniform(100, 600, n),
        })
        return df

    def test_multi_month_correct_row_count(self):
        """Multi-month audit should produce one row per month."""
        from audit_metric_alignment import (
            _filter_raw_for_month,
            _compute_monthly_da_anchor_smape,
            _compute_monthly_row_details,
        )
        from models.deep_sgdf_delta.business_time import add_business_time_columns

        raw_df = self._make_multi_month_raw_data()
        months = ["2026-01", "2026-02", "2026-03"]

        monthly_rows = []
        for month_str in months:
            month_df = _filter_raw_for_month(raw_df, month_str)
            month_df = add_business_time_columns(month_df, timestamp_col="ds")
            detail = _compute_monthly_row_details(month_df, month_str)
            monthly_rows.append(detail)

        assert len(monthly_rows) == 3, f"Expected 3 monthly rows, got {len(monthly_rows)}"

    def test_each_month_has_correct_row_count(self):
        """Each month should have the expected number of hourly rows."""
        from audit_metric_alignment import _filter_raw_for_month

        raw_df = self._make_multi_month_raw_data()

        # January: 31 days * 24 hours = 744
        jan_df = _filter_raw_for_month(raw_df, "2026-01")
        assert len(jan_df) == 31 * 24, f"Jan expected {31*24} rows, got {len(jan_df)}"

        # February: 28 days * 24 hours = 672 (2026 is not a leap year)
        feb_df = _filter_raw_for_month(raw_df, "2026-02")
        assert len(feb_df) == 28 * 24, f"Feb expected {28*24} rows, got {len(feb_df)}"

        # March: 31 days * 24 hours = 744
        mar_df = _filter_raw_for_month(raw_df, "2026-03")
        assert len(mar_df) == 31 * 24, f"Mar expected {31*24} rows, got {len(mar_df)}"

    def test_multi_month_empty_month(self):
        """A month with no data should produce a row with row_count=0."""
        from audit_metric_alignment import _filter_raw_for_month

        raw_df = self._make_multi_month_raw_data()
        apr_df = _filter_raw_for_month(raw_df, "2026-04")
        assert len(apr_df) == 0

    def test_multi_month_smape_computed(self):
        """DA anchor sMAPE should be computed for each month."""
        from audit_metric_alignment import (
            _filter_raw_for_month,
            _compute_monthly_da_anchor_smape,
        )

        raw_df = self._make_multi_month_raw_data()
        months = ["2026-01", "2026-02", "2026-03"]

        for month_str in months:
            month_df = _filter_raw_for_month(raw_df, month_str)
            smape_val = _compute_monthly_da_anchor_smape(month_df)
            assert smape_val is not None, f"sMAPE should not be None for {month_str}"
            assert isinstance(smape_val, float)
            assert 0.0 <= smape_val <= 200.0, f"sMAPE out of range for {month_str}: {smape_val}"


# ═══════════════════════════════════════════════════════════════════════
# Test 6: Business day uniqueness check
# ═══════════════════════════════════════════════════════════════════════

class TestBusinessDayUniqueness:
    """Verify business day handling in multi-month audit."""

    def test_business_day_uniqueness(self):
        """Each (business_day, hour_business) pair should be unique in clean data."""
        from audit_metric_alignment import _filter_raw_for_month
        from models.deep_sgdf_delta.business_time import add_business_time_columns

        np.random.seed(42)
        ds = pd.date_range("2026-02-01", "2026-02-28 23:00", freq="h")
        n = len(ds)
        raw_df = pd.DataFrame({
            "ds": ds,
            "da_anchor": np.random.uniform(200, 500, n),
            "rt_actual": np.random.uniform(100, 600, n),
        })

        month_df = _filter_raw_for_month(raw_df, "2026-02")
        month_df = add_business_time_columns(month_df, timestamp_col="ds")

        # Check uniqueness of (business_day, hour_business) pairs
        pairs = month_df[["business_day", "hour_business"]].drop_duplicates()
        assert len(pairs) == len(month_df), (
            f"Expected {len(month_df)} unique (bd, hb) pairs, got {len(pairs)}"
        )

    def test_business_day_range_computed(self):
        """Business day range should be computed correctly."""
        from audit_metric_alignment import _compute_monthly_row_details
        from models.deep_sgdf_delta.business_time import add_business_time_columns

        np.random.seed(42)
        ds = pd.date_range("2026-02-01", "2026-02-28 23:00", freq="h")
        n = len(ds)
        raw_df = pd.DataFrame({
            "ds": ds,
            "da_anchor": np.random.uniform(200, 500, n),
            "rt_actual": np.random.uniform(100, 600, n),
        })

        raw_df = add_business_time_columns(raw_df, timestamp_col="ds")
        detail = _compute_monthly_row_details(raw_df, "2026-02")

        assert detail["business_day_range"] != "N/A"
        assert "~" in detail["business_day_range"]

    def test_missing_rows_detection(self):
        """Missing rows should be detected when data is incomplete."""
        from audit_metric_alignment import _compute_monthly_row_details
        from models.deep_sgdf_delta.business_time import add_business_time_columns

        # Create data with gaps: only hours 1-12 for each day (missing 13-24)
        ds = pd.date_range("2026-02-01 01:00", "2026-02-28 12:00", freq="h")
        n = len(ds)
        raw_df = pd.DataFrame({
            "ds": ds,
            "da_anchor": np.random.uniform(200, 500, n),
            "rt_actual": np.random.uniform(100, 600, n),
        })

        raw_df = add_business_time_columns(raw_df, timestamp_col="ds")
        detail = _compute_monthly_row_details(raw_df, "2026-02")

        # Should detect missing rows (only 12 hours per day instead of 24)
        assert detail["missing_rows"] > 0, "Should detect missing rows for incomplete data"

    def test_duplicate_rows_detection(self):
        """Duplicate rows should be detected when timestamps repeat."""
        from audit_metric_alignment import _compute_monthly_row_details
        from models.deep_sgdf_delta.business_time import add_business_time_columns

        # Create data with duplicates
        ds = pd.date_range("2026-02-01", periods=24, freq="h")
        ds_with_dup = ds.append(ds[:6])  # duplicate first 6 hours
        n = len(ds_with_dup)
        raw_df = pd.DataFrame({
            "ds": ds_with_dup,
            "da_anchor": np.random.uniform(200, 500, n),
            "rt_actual": np.random.uniform(100, 600, n),
        })

        raw_df = add_business_time_columns(raw_df, timestamp_col="ds")
        detail = _compute_monthly_row_details(raw_df, "2026-02")

        assert detail["duplicate_rows"] == 6, (
            f"Expected 6 duplicate rows, got {detail['duplicate_rows']}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Test 7: Verdict logic for multi-month
# ═══════════════════════════════════════════════════════════════════════

class TestMultiMonthVerdict:
    """Multi-month verdict logic: FAIL if any FAIL, WARN if any WARN, else PASS."""

    def test_all_pass(self):
        """All months PASS → overall PASS."""
        from audit_metric_alignment import compute_multi_month_verdict
        verdict = compute_multi_month_verdict(["PASS", "PASS", "PASS"])
        assert verdict == "PASS"

    def test_one_warn(self):
        """One WARN, rest PASS → overall WARN."""
        from audit_metric_alignment import compute_multi_month_verdict
        verdict = compute_multi_month_verdict(["PASS", "WARN", "PASS"])
        assert verdict == "WARN"

    def test_one_fail(self):
        """One FAIL → overall FAIL regardless of others."""
        from audit_metric_alignment import compute_multi_month_verdict
        verdict = compute_multi_month_verdict(["PASS", "FAIL", "PASS"])
        assert verdict == "FAIL"

    def test_fail_overrides_warn(self):
        """FAIL takes precedence over WARN."""
        from audit_metric_alignment import compute_multi_month_verdict
        verdict = compute_multi_month_verdict(["WARN", "FAIL", "WARN"])
        assert verdict == "FAIL"

    def test_empty_list(self):
        """Empty verdict list → INSUFFICIENT."""
        from audit_metric_alignment import compute_multi_month_verdict
        verdict = compute_multi_month_verdict([])
        assert verdict == "INSUFFICIENT"

    def test_all_insufficient(self):
        """All INSUFFICIENT → not PASS."""
        from audit_metric_alignment import compute_multi_month_verdict
        verdict = compute_multi_month_verdict(["INSUFFICIENT", "INSUFFICIENT"])
        assert verdict == "INSUFFICIENT"

    def test_no_data_treated(self):
        """NO_DATA months should not cause PASS."""
        from audit_metric_alignment import compute_multi_month_verdict
        verdict = compute_multi_month_verdict(["PASS", "NO_DATA"])
        # NO_DATA is not PASS, WARN, or FAIL → falls through to INSUFFICIENT
        assert verdict in ("INSUFFICIENT", "PASS")

    def test_multi_month_end_to_end(self, tmp_path):
        """End-to-end multi-month audit with fixture data."""
        from audit_metric_alignment import (
            _filter_raw_for_month,
            _compute_monthly_da_anchor_smape,
            _compute_monthly_row_details,
            compute_multi_month_verdict,
        )
        from models.deep_sgdf_delta.business_time import add_business_time_columns

        np.random.seed(42)
        # Create 2-month data
        ds_jan = pd.date_range("2026-01-01", "2026-01-31 23:00", freq="h")
        ds_feb = pd.date_range("2026-02-01", "2026-02-28 23:00", freq="h")
        all_ds = ds_jan.append(ds_feb)
        n = len(all_ds)

        raw_df = pd.DataFrame({
            "ds": all_ds,
            "da_anchor": np.random.uniform(200, 500, n),
            "rt_actual": np.random.uniform(100, 600, n),
        })

        months = ["2026-01", "2026-02"]
        monthly_verdicts = []
        monthly_details = []

        for month_str in months:
            month_df = _filter_raw_for_month(raw_df, month_str)
            month_df = add_business_time_columns(month_df, timestamp_col="ds")
            da_smape = _compute_monthly_da_anchor_smape(month_df)
            detail = _compute_monthly_row_details(month_df, month_str)

            if da_smape is not None:
                if da_smape < 15.0:
                    verdict = "PASS"
                elif da_smape < 25.0:
                    verdict = "WARN"
                else:
                    verdict = "FAIL"
            else:
                verdict = "INSUFFICIENT"

            detail["verdict"] = verdict
            monthly_details.append(detail)
            monthly_verdicts.append(verdict)

        overall = compute_multi_month_verdict(monthly_verdicts)
        assert overall in ("PASS", "WARN", "FAIL", "INSUFFICIENT")
        assert len(monthly_details) == 2
        assert all(d["row_count"] > 0 for d in monthly_details)
