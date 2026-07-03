"""Tests for export_risk_feature_pack script.

Covers:
  1. Online mode excludes rt_actual / y_true
  2. Eval mode includes y_true
  3. Rows unique by (business_day, hour_business)
  4. Manifest has required fields
  5. FAIL alignment status prevents export
  6. Script runs end-to-end with fixture data
  7. All risk columns present
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

from scripts.export_risk_feature_pack import (  # noqa: E402
    KEY_COLUMNS,
    ONLINE_COLUMNS,
    EVAL_EXTRA_COLUMNS,
    RISK_FEATURE_VERSION,
    build_risk_feature_pack,
    main,
    write_manifest,
)


# -- Fixtures -----------------------------------------------------------------

def _make_business_hours(n_days: int = 3) -> pd.DataFrame:
    """Create a DataFrame of (business_day, hour_business) for n_days x 24 hours."""
    rows = []
    base = pd.Timestamp("2026-02-01")
    for d in range(n_days):
        day = base + pd.Timedelta(days=d)
        for h in range(1, 25):
            rows.append({"business_day": day, "hour_business": h})
    return pd.DataFrame(rows)


@pytest.fixture()
def fixture_delta_supply():
    """Synthetic DeltaSupply predictions CSV content."""
    rng = np.random.RandomState(42)
    kh = _make_business_hours(3)
    n = len(kh)
    return pd.DataFrame({
        "business_day": kh["business_day"],
        "hour_business": kh["hour_business"],
        "ds": kh["business_day"] + pd.to_timedelta(kh["hour_business"], unit="h"),
        "upward_deviation_prob": rng.beta(2, 5, n),
        "downward_deviation_prob": rng.beta(2, 5, n),
        "large_abs_deviation_prob": rng.beta(2, 5, n),
        "deviation_risk_score": rng.uniform(0, 1, n),
    })


@pytest.fixture()
def fixture_spike():
    """Synthetic SpikeRisk predictions CSV content."""
    rng = np.random.RandomState(43)
    kh = _make_business_hours(3)
    n = len(kh)
    return pd.DataFrame({
        "business_day": kh["business_day"],
        "hour_business": kh["hour_business"],
        "ds": kh["business_day"] + pd.to_timedelta(kh["hour_business"], unit="h"),
        "spike_prob": rng.beta(2, 8, n),
        "extreme_spike_prob": rng.beta(1, 15, n),
        "spike_risk_score": rng.uniform(0, 1, n),
    })


@pytest.fixture()
def fixture_negative():
    """Synthetic NegativeRisk predictions CSV content."""
    rng = np.random.RandomState(44)
    kh = _make_business_hours(3)
    n = len(kh)
    return pd.DataFrame({
        "business_day": kh["business_day"],
        "hour_business": kh["hour_business"],
        "ds": kh["business_day"] + pd.to_timedelta(kh["hour_business"], unit="h"),
        "negative_prob": rng.beta(1, 10, n),
        "deep_negative_prob": rng.beta(1, 20, n),
        "negative_risk_score": rng.uniform(0, 1, n),
    })


@pytest.fixture()
def fixture_delta_supply_eval(fixture_delta_supply):
    """DeltaSupply with y_true column for eval mode."""
    df = fixture_delta_supply.copy()
    df["y_true"] = np.random.RandomState(50).uniform(0, 500, len(df))
    return df


# -- 1. Online mode excludes rt_actual / y_true --------------------------------

class TestOnlineModeExcludesYtrue:
    def test_no_y_true_column(self, fixture_delta_supply, fixture_spike, fixture_negative):
        pack = build_risk_feature_pack(
            fixture_delta_supply, fixture_spike, fixture_negative,
            mode="online", metric_alignment_status="PASS",
        )
        assert "y_true" not in pack.columns
        assert "rt_actual" not in pack.columns

    def test_online_columns_only(self, fixture_delta_supply, fixture_spike, fixture_negative):
        pack = build_risk_feature_pack(
            fixture_delta_supply, fixture_spike, fixture_negative,
            mode="online", metric_alignment_status="PASS",
        )
        for col in pack.columns:
            assert col in ONLINE_COLUMNS, f"Unexpected column in online mode: {col}"


# -- 2. Eval mode includes y_true --------------------------------------------

class TestEvalModeIncludesYtrue:
    def test_y_true_present_in_eval(
        self, fixture_delta_supply_eval, fixture_spike, fixture_negative,
    ):
        pack = build_risk_feature_pack(
            fixture_delta_supply_eval, fixture_spike, fixture_negative,
            mode="eval", metric_alignment_status="PASS",
        )
        assert "y_true" in pack.columns

    def test_eval_has_online_plus_extra(
        self, fixture_delta_supply_eval, fixture_spike, fixture_negative,
    ):
        pack = build_risk_feature_pack(
            fixture_delta_supply_eval, fixture_spike, fixture_negative,
            mode="eval", metric_alignment_status="PASS",
        )
        expected_cols = set(ONLINE_COLUMNS) | set(EVAL_EXTRA_COLUMNS)
        actual_cols = set(pack.columns)
        # All actual columns should be in the expected set
        assert actual_cols.issubset(expected_cols), (
            f"Unexpected columns: {actual_cols - expected_cols}"
        )


# -- 3. Rows unique by (business_day, hour_business) --------------------------

class TestRowUniqueness:
    def test_unique_keys(self, fixture_delta_supply, fixture_spike, fixture_negative):
        pack = build_risk_feature_pack(
            fixture_delta_supply, fixture_spike, fixture_negative,
            mode="online", metric_alignment_status="PASS",
        )
        n_rows = len(pack)
        n_unique = pack.drop_duplicates(subset=KEY_COLUMNS).shape[0]
        assert n_rows == n_unique, (
            f"Expected {n_unique} unique rows but got {n_rows}"
        )

    def test_unique_keys_with_duplicates_in_source(
        self, fixture_delta_supply, fixture_spike, fixture_negative,
    ):
        """Even if source has duplicates, output should be deduplicated."""
        # Duplicate first 5 rows in delta supply
        dup_df = pd.concat(
            [fixture_delta_supply.head(5), fixture_delta_supply],
            ignore_index=True,
        )
        pack = build_risk_feature_pack(
            dup_df, fixture_spike, fixture_negative,
            mode="online", metric_alignment_status="PASS",
        )
        n_unique = pack.drop_duplicates(subset=KEY_COLUMNS).shape[0]
        assert n_unique == len(pack)


# -- 4. Manifest has required fields ------------------------------------------

class TestManifest:
    REQUIRED_MANIFEST_FIELDS = [
        "risk_feature_version",
        "mode",
        "metric_alignment_status",
        "n_rows",
        "columns",
        "key_columns",
        "unique_keys",
    ]

    def test_manifest_has_required_fields(
        self, tmp_path, fixture_delta_supply, fixture_spike, fixture_negative,
    ):
        pack = build_risk_feature_pack(
            fixture_delta_supply, fixture_spike, fixture_negative,
            mode="online", metric_alignment_status="PASS",
        )
        write_manifest(tmp_path, pack, "online", "PASS")

        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for field in self.REQUIRED_MANIFEST_FIELDS:
            assert field in manifest, f"Missing manifest field: {field}"

    def test_manifest_row_count_matches(
        self, tmp_path, fixture_delta_supply, fixture_spike, fixture_negative,
    ):
        pack = build_risk_feature_pack(
            fixture_delta_supply, fixture_spike, fixture_negative,
            mode="online", metric_alignment_status="PASS",
        )
        write_manifest(tmp_path, pack, "online", "PASS")

        with open(tmp_path / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["n_rows"] == len(pack)
        assert manifest["unique_keys"] == len(pack)

    def test_manifest_version_matches(
        self, tmp_path, fixture_delta_supply, fixture_spike, fixture_negative,
    ):
        pack = build_risk_feature_pack(
            fixture_delta_supply, fixture_spike, fixture_negative,
            mode="online", metric_alignment_status="PASS",
        )
        write_manifest(tmp_path, pack, "online", "PASS")

        with open(tmp_path / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["risk_feature_version"] == RISK_FEATURE_VERSION


# -- 5. FAIL alignment status prevents export ---------------------------------

class TestFailAlignmentPreventsExport:
    def test_fail_status_exits(self, tmp_path, fixture_delta_supply, fixture_spike,
                               fixture_negative):
        """Script must exit(1) when metric_alignment_status is FAIL."""
        # Write fixture CSVs
        delta_path = tmp_path / "delta.csv"
        spike_path = tmp_path / "spike.csv"
        neg_path = tmp_path / "neg.csv"
        fixture_delta_supply.to_csv(delta_path, index=False)
        fixture_spike.to_csv(spike_path, index=False)
        fixture_negative.to_csv(neg_path, index=False)

        out_dir = tmp_path / "pack_output"

        old_argv = sys.argv
        sys.argv = [
            "export_risk_feature_pack.py",
            "--delta-supply-predictions", str(delta_path),
            "--spike-predictions", str(spike_path),
            "--negative-predictions", str(neg_path),
            "--metric-alignment-status", "FAIL",
            "--out-dir", str(out_dir),
            "--mode", "online",
        ]
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
        finally:
            sys.argv = old_argv

        # Output directory should NOT have been created
        assert not (out_dir / "risk_feature_pack.csv").exists()


# -- 6. Script runs end-to-end with fixture data ------------------------------

class TestEndToEnd:
    def test_full_cli_online(self, tmp_path, fixture_delta_supply, fixture_spike,
                             fixture_negative):
        """Full CLI invocation in online mode produces expected files."""
        delta_path = tmp_path / "delta.csv"
        spike_path = tmp_path / "spike.csv"
        neg_path = tmp_path / "neg.csv"
        fixture_delta_supply.to_csv(delta_path, index=False)
        fixture_spike.to_csv(spike_path, index=False)
        fixture_negative.to_csv(neg_path, index=False)

        out_dir = tmp_path / "pack_output"

        old_argv = sys.argv
        sys.argv = [
            "export_risk_feature_pack.py",
            "--delta-supply-predictions", str(delta_path),
            "--spike-predictions", str(spike_path),
            "--negative-predictions", str(neg_path),
            "--metric-alignment-status", "PASS",
            "--out-dir", str(out_dir),
            "--mode", "online",
        ]
        try:
            main()
        finally:
            sys.argv = old_argv

        # Check outputs exist
        assert (out_dir / "risk_feature_pack.csv").exists()
        assert (out_dir / "manifest.json").exists()

        # Check CSV is parseable
        pack = pd.read_csv(out_dir / "risk_feature_pack.csv")
        assert len(pack) > 0

        # Check manifest is parseable
        with open(out_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["n_rows"] == len(pack)

    def test_full_cli_eval(self, tmp_path, fixture_delta_supply_eval, fixture_spike,
                           fixture_negative):
        """Full CLI invocation in eval mode includes y_true."""
        delta_path = tmp_path / "delta.csv"
        spike_path = tmp_path / "spike.csv"
        neg_path = tmp_path / "neg.csv"
        fixture_delta_supply_eval.to_csv(delta_path, index=False)
        fixture_spike.to_csv(spike_path, index=False)
        fixture_negative.to_csv(neg_path, index=False)

        out_dir = tmp_path / "pack_output_eval"

        old_argv = sys.argv
        sys.argv = [
            "export_risk_feature_pack.py",
            "--delta-supply-predictions", str(delta_path),
            "--spike-predictions", str(spike_path),
            "--negative-predictions", str(neg_path),
            "--metric-alignment-status", "PASS",
            "--out-dir", str(out_dir),
            "--mode", "eval",
        ]
        try:
            main()
        finally:
            sys.argv = old_argv

        pack = pd.read_csv(out_dir / "risk_feature_pack.csv")
        assert "y_true" in pack.columns


# -- 7. All risk columns present -----------------------------------------------

class TestAllRiskColumnsPresent:
    EXPECTED_RISK_COLUMNS = [
        "deviation_up_prob",
        "deviation_down_prob",
        "deviation_large_abs_prob",
        "deviation_risk_score",
        "spike_prob",
        "extreme_spike_prob",
        "spike_risk_score",
        "negative_prob",
        "deep_negative_prob",
        "negative_risk_score",
    ]

    def test_all_risk_columns_present(
        self, fixture_delta_supply, fixture_spike, fixture_negative,
    ):
        pack = build_risk_feature_pack(
            fixture_delta_supply, fixture_spike, fixture_negative,
            mode="online", metric_alignment_status="PASS",
        )
        for col in self.EXPECTED_RISK_COLUMNS:
            assert col in pack.columns, f"Missing risk column: {col}"

    def test_metadata_columns_present(
        self, fixture_delta_supply, fixture_spike, fixture_negative,
    ):
        pack = build_risk_feature_pack(
            fixture_delta_supply, fixture_spike, fixture_negative,
            mode="online", metric_alignment_status="PASS",
        )
        assert "risk_feature_version" in pack.columns
        assert "metric_alignment_status" in pack.columns

    def test_key_columns_present(
        self, fixture_delta_supply, fixture_spike, fixture_negative,
    ):
        pack = build_risk_feature_pack(
            fixture_delta_supply, fixture_spike, fixture_negative,
            mode="online", metric_alignment_status="PASS",
        )
        assert "business_day" in pack.columns
        assert "hour_business" in pack.columns
        assert "ds" in pack.columns

    def test_probabilities_in_range(
        self, fixture_delta_supply, fixture_spike, fixture_negative,
    ):
        """All probability columns should be in [0, 1]."""
        pack = build_risk_feature_pack(
            fixture_delta_supply, fixture_spike, fixture_negative,
            mode="online", metric_alignment_status="PASS",
        )
        prob_cols = [
            "deviation_up_prob", "deviation_down_prob", "deviation_large_abs_prob",
            "spike_prob", "extreme_spike_prob",
            "negative_prob", "deep_negative_prob",
        ]
        for col in prob_cols:
            if col in pack.columns:
                vals = pack[col].dropna()
                assert (vals >= 0).all(), f"{col} has values < 0"
                assert (vals <= 1).all(), f"{col} has values > 1"

    def test_risk_scores_in_range(
        self, fixture_delta_supply, fixture_spike, fixture_negative,
    ):
        """Risk score columns should be in [0, 1]."""
        pack = build_risk_feature_pack(
            fixture_delta_supply, fixture_spike, fixture_negative,
            mode="online", metric_alignment_status="PASS",
        )
        score_cols = ["deviation_risk_score", "spike_risk_score", "negative_risk_score"]
        for col in score_cols:
            if col in pack.columns:
                vals = pack[col].dropna()
                assert (vals >= 0).all(), f"{col} has values < 0"
                assert (vals <= 1).all(), f"{col} has values > 1"

    def test_version_value_matches(
        self, fixture_delta_supply, fixture_spike, fixture_negative,
    ):
        pack = build_risk_feature_pack(
            fixture_delta_supply, fixture_spike, fixture_negative,
            mode="online", metric_alignment_status="PASS",
        )
        assert (pack["risk_feature_version"] == RISK_FEATURE_VERSION).all()

    def test_alignment_status_value_matches(
        self, fixture_delta_supply, fixture_spike, fixture_negative,
    ):
        pack = build_risk_feature_pack(
            fixture_delta_supply, fixture_spike, fixture_negative,
            mode="online", metric_alignment_status="PASS",
        )
        assert (pack["metric_alignment_status"] == "PASS").all()
