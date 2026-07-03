"""Track D tests: v1.1.0 field alignment contract verification.

Covers:
  1. Contract doc fields match exporter ONLINE_COLUMNS
  2. relative_spike_prob and relative_down_prob are in ONLINE_COLUMNS
  3. metric_alignment_warning_reason is in ONLINE_COLUMNS
  4. All probability columns should be in [0,1] range (test with mock data)
  5. risk_feature_version = "v1.1.0"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_risk_feature_pack_multimonth import (  # noqa: E402
    ONLINE_COLUMNS,
    RISK_FEATURE_VERSION,
    SPIKE_RISK_COLS,
    NEGATIVE_RISK_COLS,
    DELTA_SUPPLY_RISK_COLS,
    KEY_COLUMNS,
    build_risk_feature_pack_multimonth,
    write_manifest,
)


# -- Contract doc field lists -------------------------------------------------
# These are the fields documented in docs/RISK_FEATURE_PACK_CONTRACT.md.
# The contract doc was written at v1.0.0; v1.1.0 adds new fields.

CONTRACT_DOC_KEY_COLUMNS = [
    "business_day",
    "hour_business",
    "ds",
]

CONTRACT_DOC_DELTA_SUPPLY_COLS = [
    "deviation_up_prob",
    "deviation_down_prob",
    "deviation_large_abs_prob",
    "deviation_risk_score",
]

CONTRACT_DOC_SPIKE_RISK_COLS = [
    "spike_prob",
    "extreme_spike_prob",
    "spike_risk_score",
]

CONTRACT_DOC_NEGATIVE_RISK_COLS = [
    "negative_prob",
    "deep_negative_prob",
    "negative_risk_score",
]

CONTRACT_DOC_METADATA_COLS = [
    "risk_feature_version",
    "metric_alignment_status",
]

CONTRACT_DOC_ALL_FIELDS = (
    CONTRACT_DOC_KEY_COLUMNS
    + CONTRACT_DOC_DELTA_SUPPLY_COLS
    + CONTRACT_DOC_SPIKE_RISK_COLS
    + CONTRACT_DOC_NEGATIVE_RISK_COLS
    + CONTRACT_DOC_METADATA_COLS
)

# v1.1.0 additions (beyond the v1.0.0 contract doc):
V1_1_0_NEW_FIELDS = [
    "target_month",
    "relative_spike_prob",
    "relative_down_prob",
    "module_status_delta_supply",
    "module_status_spike",
    "module_status_negative",
    "threshold_version",
    "metric_alignment_warning_reason",
]

# Probability columns that must be in [0, 1].
PROBABILITY_COLUMNS = [
    "deviation_up_prob",
    "deviation_down_prob",
    "deviation_large_abs_prob",
    "deviation_risk_score",
    "spike_prob",
    "extreme_spike_prob",
    "relative_spike_prob",
    "spike_risk_score",
    "negative_prob",
    "deep_negative_prob",
    "relative_down_prob",
    "negative_risk_score",
]


# -- Helpers ------------------------------------------------------------------

def _make_business_hours(n_days: int = 3, base: str = "2026-01-01") -> pd.DataFrame:
    rows = []
    start = pd.Timestamp(base)
    for d in range(n_days):
        day = start + pd.Timedelta(days=d)
        for h in range(1, 25):
            rows.append({"business_day": day, "hour_business": h})
    return pd.DataFrame(rows)


def _make_prediction_df(month: str, n_days: int = 3) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    kh = _make_business_hours(n_days, base=f"{month}-01")
    n = len(kh)
    return pd.DataFrame({
        "business_day": kh["business_day"],
        "hour_business": kh["hour_business"],
        "ds": kh["business_day"] + pd.to_timedelta(kh["hour_business"], unit="h"),
        "period": [month] * n,
        "upward_deviation_prob": rng.beta(2, 5, n),
        "downward_deviation_prob": rng.beta(2, 5, n),
        "large_abs_deviation_prob": rng.beta(2, 5, n),
        "deviation_risk_score": rng.uniform(0, 1, n),
        "spike_prob": rng.beta(2, 8, n),
        "extreme_spike_prob": rng.beta(1, 15, n),
        "relative_spike_prob": rng.beta(2, 8, n),
        "spike_risk_score": rng.uniform(0, 1, n),
        "negative_prob": rng.beta(1, 10, n),
        "deep_negative_prob": rng.beta(1, 20, n),
        "relative_down_prob": rng.beta(1, 10, n),
        "negative_risk_score": rng.uniform(0, 1, n),
    })


def _write_monthly_predictions(root: Path, month: str, df: pd.DataFrame) -> None:
    root.mkdir(parents=True, exist_ok=True)
    yyyymm = month.replace("-", "_")
    path = root / f"predictions_{yyyymm}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_champion_summary(root: Path, verdicts: dict, overall: str = "ACCEPTABLE") -> None:
    root.mkdir(parents=True, exist_ok=True)
    summary = {
        "overall_verdict": overall,
        "monthly_verdicts": verdicts,
    }
    with open(root / "champion_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


@pytest.fixture()
def fixture_roots(tmp_path):
    """Create backtest roots with one month of data and GO verdicts."""
    months = ["2026-01"]
    delta_root = tmp_path / "delta_backtest"
    spike_root = tmp_path / "spike_backtest"
    negative_root = tmp_path / "negative_backtest"

    verdicts = {"2026-01": "GO"}
    for month in months:
        pred_df = _make_prediction_df(month)
        _write_monthly_predictions(delta_root, month, pred_df)
        _write_monthly_predictions(spike_root, month, pred_df)
        _write_monthly_predictions(negative_root, month, pred_df)

    _write_champion_summary(delta_root, verdicts, "ACCEPTABLE")
    _write_champion_summary(spike_root, verdicts, "ACCEPTABLE")
    _write_champion_summary(negative_root, verdicts, "GO")

    return {
        "delta_root": delta_root,
        "spike_root": spike_root,
        "negative_root": negative_root,
    }


# -- 1. Contract doc fields match exporter ONLINE_COLUMNS ---------------------

class TestContractDocFieldsMatchExporter:
    def test_all_contract_doc_fields_in_online_columns(self):
        """Every field from the v1.0.0 contract doc must be in ONLINE_COLUMNS."""
        for field in CONTRACT_DOC_ALL_FIELDS:
            assert field in ONLINE_COLUMNS, (
                f"Contract doc field '{field}' is missing from ONLINE_COLUMNS"
            )

    def test_v1_1_0_new_fields_in_online_columns(self):
        """All v1.1.0 new fields must be in ONLINE_COLUMNS."""
        for field in V1_1_0_NEW_FIELDS:
            assert field in ONLINE_COLUMNS, (
                f"v1.1.0 field '{field}' is missing from ONLINE_COLUMNS"
            )

    def test_online_columns_has_no_duplicates(self):
        """ONLINE_COLUMNS must not contain duplicate column names."""
        assert len(ONLINE_COLUMNS) == len(set(ONLINE_COLUMNS))

    def test_contract_doc_key_columns_are_first(self):
        """Key columns should appear first in ONLINE_COLUMNS."""
        for i, col in enumerate(CONTRACT_DOC_KEY_COLUMNS):
            assert ONLINE_COLUMNS[i] == col, (
                f"Expected key column '{col}' at position {i}, "
                f"but found '{ONLINE_COLUMNS[i]}'"
            )


# -- 2. relative_spike_prob and relative_down_prob are in ONLINE_COLUMNS ------

class TestV110RelativeColumns:
    def test_relative_spike_prob_in_online_columns(self):
        assert "relative_spike_prob" in ONLINE_COLUMNS

    def test_relative_down_prob_in_online_columns(self):
        assert "relative_down_prob" in ONLINE_COLUMNS

    def test_relative_spike_prob_in_spike_risk_cols(self):
        assert "relative_spike_prob" in SPIKE_RISK_COLS

    def test_relative_down_prob_in_negative_risk_cols(self):
        assert "relative_down_prob" in NEGATIVE_RISK_COLS

    def test_relative_columns_present_in_pack(self, fixture_roots):
        """The exported pack must contain relative_spike_prob and relative_down_prob."""
        pack, _, _ = build_risk_feature_pack_multimonth(
            delta_supply_root=fixture_roots["delta_root"],
            spike_root=fixture_roots["spike_root"],
            negative_root=fixture_roots["negative_root"],
            mode="online",
            metric_alignment_status="PASS",
        )
        assert "relative_spike_prob" in pack.columns
        assert "relative_down_prob" in pack.columns


# -- 3. metric_alignment_warning_reason is in ONLINE_COLUMNS ------------------

class TestMetricAlignmentWarningReason:
    def test_in_online_columns(self):
        assert "metric_alignment_warning_reason" in ONLINE_COLUMNS

    def test_present_in_pack(self, fixture_roots):
        pack, _, _ = build_risk_feature_pack_multimonth(
            delta_supply_root=fixture_roots["delta_root"],
            spike_root=fixture_roots["spike_root"],
            negative_root=fixture_roots["negative_root"],
            mode="online",
            metric_alignment_status="PASS",
        )
        assert "metric_alignment_warning_reason" in pack.columns

    def test_in_manifest(self, tmp_path, fixture_roots):
        pack, monthly_manifest, status_sources = build_risk_feature_pack_multimonth(
            delta_supply_root=fixture_roots["delta_root"],
            spike_root=fixture_roots["spike_root"],
            negative_root=fixture_roots["negative_root"],
            mode="online",
            metric_alignment_status="PASS",
        )
        out_dir = tmp_path / "pack_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(
            out_dir, pack, "online", "PASS", monthly_manifest,
            status_sources, "",
        )

        with open(out_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert "metric_alignment_warning_reason" in manifest


# -- 4. All probability columns should be in [0,1] range ----------------------

class TestProbabilityRange:
    def test_probability_columns_in_range(self, fixture_roots):
        """All probability/score columns must have values in [0, 1]."""
        pack, _, _ = build_risk_feature_pack_multimonth(
            delta_supply_root=fixture_roots["delta_root"],
            spike_root=fixture_roots["spike_root"],
            negative_root=fixture_roots["negative_root"],
            mode="online",
            metric_alignment_status="PASS",
        )

        for col in PROBABILITY_COLUMNS:
            if col not in pack.columns:
                continue
            non_null = pack[col].dropna()
            if len(non_null) == 0:
                continue
            assert non_null.min() >= 0.0, (
                f"Column '{col}' has values below 0: min={non_null.min()}"
            )
            assert non_null.max() <= 1.0, (
                f"Column '{col}' has values above 1: max={non_null.max()}"
            )

    def test_nogo_month_nan_not_in_range_check(self, tmp_path):
        """NO-GO months have NaN risk columns; NaN should not fail range checks."""
        months = ["2026-01", "2026-02"]
        delta_root = tmp_path / "delta_backtest"
        spike_root = tmp_path / "spike_backtest"
        negative_root = tmp_path / "negative_backtest"

        verdicts_delta = {"2026-01": "GO", "2026-02": "NO_GO"}
        verdicts_spike = {"2026-01": "GO", "2026-02": "GO"}
        verdicts_negative = {"2026-01": "GO", "2026-02": "GO"}

        for month in months:
            pred_df = _make_prediction_df(month)
            _write_monthly_predictions(delta_root, month, pred_df)
            _write_monthly_predictions(spike_root, month, pred_df)
            _write_monthly_predictions(negative_root, month, pred_df)

        _write_champion_summary(delta_root, verdicts_delta, "LOW_VALUE")
        _write_champion_summary(spike_root, verdicts_spike, "GO")
        _write_champion_summary(negative_root, verdicts_negative, "GO")

        pack, _, _ = build_risk_feature_pack_multimonth(
            delta_supply_root=delta_root,
            spike_root=spike_root,
            negative_root=negative_root,
            mode="online",
            metric_alignment_status="PASS",
        )

        # Check that all non-NaN probability values are in [0, 1].
        for col in PROBABILITY_COLUMNS:
            if col not in pack.columns:
                continue
            non_null = pack[col].dropna()
            if len(non_null) == 0:
                continue
            assert non_null.min() >= 0.0, f"{col} below 0"
            assert non_null.max() <= 1.0, f"{col} above 1"


# -- 5. risk_feature_version = "v1.1.0" --------------------------------------

class TestRiskFeatureVersion:
    def test_version_constant(self):
        """RISK_FEATURE_VERSION must be exactly 'v1.1.0'."""
        assert RISK_FEATURE_VERSION == "v1.1.0"

    def test_version_in_pack(self, fixture_roots):
        """Exported pack must have risk_feature_version = 'v1.1.0'."""
        pack, _, _ = build_risk_feature_pack_multimonth(
            delta_supply_root=fixture_roots["delta_root"],
            spike_root=fixture_roots["spike_root"],
            negative_root=fixture_roots["negative_root"],
            mode="online",
            metric_alignment_status="PASS",
        )
        assert (pack["risk_feature_version"] == "v1.1.0").all()

    def test_version_in_manifest(self, tmp_path, fixture_roots):
        """Manifest must record risk_feature_version = 'v1.1.0'."""
        pack, monthly_manifest, status_sources = build_risk_feature_pack_multimonth(
            delta_supply_root=fixture_roots["delta_root"],
            spike_root=fixture_roots["spike_root"],
            negative_root=fixture_roots["negative_root"],
            mode="online",
            metric_alignment_status="PASS",
        )
        out_dir = tmp_path / "pack_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(
            out_dir, pack, "online", "PASS", monthly_manifest, status_sources,
        )

        with open(out_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["risk_feature_version"] == "v1.1.0"


# -- Additional: column count and completeness --------------------------------

class TestOnlineColumnsCompleteness:
    def test_online_columns_count(self):
        """ONLINE_COLUMNS should have the expected number of columns for v1.1.0.

        Breakdown:
          - 3 key columns (business_day, hour_business, ds)
          - 1 target_month
          - 4 delta_supply risk columns
          - 4 spike risk columns (including relative_spike_prob)
          - 4 negative risk columns (including relative_down_prob)
          - 3 module_status columns
          - 4 metadata columns (threshold_version, risk_feature_version,
            metric_alignment_status, metric_alignment_warning_reason)
          Total = 23
        """
        expected_count = 23
        assert len(ONLINE_COLUMNS) == expected_count, (
            f"Expected {expected_count} columns in ONLINE_COLUMNS, "
            f"got {len(ONLINE_COLUMNS)}: {ONLINE_COLUMNS}"
        )

    def test_all_risk_cols_accounted_for(self):
        """DELTA + SPIKE + NEGATIVE risk cols should cover all probability columns."""
        all_risk = set(DELTA_SUPPLY_RISK_COLS + SPIKE_RISK_COLS + NEGATIVE_RISK_COLS)
        expected = set(PROBABILITY_COLUMNS)
        assert all_risk == expected, (
            f"Risk cols mismatch. Missing from exporter: {expected - all_risk}, "
            f"Extra in exporter: {all_risk - expected}"
        )
