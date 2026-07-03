#!/usr/bin/env python3
"""Diagnostic script for verifying the SGDFNet bridge.

Usage::

    python scripts/check_sgdfnet_bridge.py
    python scripts/check_sgdfnet_bridge.py --sgdfnet-root /path/to/SGDFNet
    python scripts/check_sgdfnet_bridge.py --help

Exits with code 0 when all checks pass, 1 on any failure.
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
import traceback
from pathlib import Path


def _banner(title: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _info(msg: str) -> None:
    print(f"  [INFO] {msg}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose the SGDFNet bridge used by DeepSGDFDelta.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            The bridge looks for SGDFNet in this order:
              1. --sgdfnet-root <path>  (CLI argument)
              2. $SGDFNET_ROOT           (environment variable)
              3. ../electricity_forecast_model2.0_exp/SGDFNet  (sibling directory)
        """),
    )
    parser.add_argument(
        "--sgdfnet-root",
        type=str,
        default=None,
        help="Path to the SGDFNet project root (directory containing src/).",
    )
    args = parser.parse_args()

    failures: list[str] = []

    # ── Step 1: Resolve the SGDFNet root ──────────────────────────────
    _banner("1. Resolving SGDFNet root")

    # Make the repo root importable so we can import the bridge
    repo_root = Path(__file__).resolve().parent.parent
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    try:
        from models.deep_sgdf_delta.sgdfnet_bridge import find_sgdfnet_root
    except Exception as exc:
        _fail(f"Cannot import sgdfnet_bridge: {exc}")
        traceback.print_exc()
        return 1

    try:
        sgdfnet_root = find_sgdfnet_root(args.sgdfnet_root)
        _ok(f"SGDFNet root: {sgdfnet_root}")
    except FileNotFoundError as exc:
        _fail(str(exc))
        return 1

    # ── Step 2: Version / structure check ─────────────────────────────
    _banner("2. Structure check")

    src_dir = sgdfnet_root / "src" / "sgdfnet"
    if src_dir.is_dir():
        _ok(f"Source directory exists: {src_dir}")
    else:
        _fail(f"Source directory missing: {src_dir}")
        failures.append("source directory missing")

    init_file = src_dir / "__init__.py"
    if init_file.is_file():
        _ok(f"__init__.py found: {init_file}")
    else:
        _fail(f"__init__.py missing: {init_file}")
        failures.append("__init__.py missing")

    expected_modules = ["data_contract", "protocol_b_cutoff", "metrics", "models"]
    for mod_name in expected_modules:
        mod_file = src_dir / f"{mod_name}.py"
        if mod_file.is_file():
            _ok(f"Module present: {mod_name}.py")
        else:
            _fail(f"Module missing: {mod_name}.py")
            failures.append(f"module {mod_name}.py missing")

    # ── Step 3: Import sgdfnet_bridge and check re-exports ────────────
    _banner("3. Bridge re-exports")

    try:
        from models.deep_sgdf_delta import sgdfnet_bridge as bridge
        _ok("sgdfnet_bridge imported successfully")
    except Exception as exc:
        _fail(f"Failed to import sgdfnet_bridge: {exc}")
        traceback.print_exc()
        failures.append("bridge import failed")
        # Cannot continue if bridge is broken
        print()
        print(f"RESULT: {len(failures)} failure(s) detected.")
        return 1

    # data_contract symbols
    data_contract_symbols = [
        "load_dataset",
        "preprocess_dataframe",
        "FeatureConfig",
        "add_business_time_columns",
        "TIMESTAMP_COL",
        "DA_COL",
        "RT_COL",
    ]
    for sym in data_contract_symbols:
        if hasattr(bridge, sym):
            _ok(f"data_contract.{sym} available")
        else:
            _fail(f"data_contract.{sym} NOT available on bridge")
            failures.append(f"missing symbol: {sym}")

    # protocol_b_cutoff symbols
    protocol_symbols = [
        "run_protocol_b_cutoff_experiment",
        "_build_protocol_b_visible_frame",
        "_build_inference_frame",
    ]
    for sym in protocol_symbols:
        if hasattr(bridge, sym):
            _ok(f"protocol_b_cutoff.{sym} available")
        else:
            _fail(f"protocol_b_cutoff.{sym} NOT available on bridge")
            failures.append(f"missing symbol: {sym}")

    # metrics symbols
    metrics_symbols = [
        "build_metrics_frame",
        "capped_smape",
        "smape",
    ]
    for sym in metrics_symbols:
        if hasattr(bridge, sym):
            _ok(f"metrics.{sym} available")
        else:
            _fail(f"metrics.{sym} NOT available on bridge")
            failures.append(f"missing symbol: {sym}")

    # models symbols
    models_symbols = [
        "DeltaRegressor",
        "HGBModelConfig",
    ]
    for sym in models_symbols:
        if hasattr(bridge, sym):
            _ok(f"models.{sym} available")
        else:
            _fail(f"models.{sym} NOT available on bridge")
            failures.append(f"missing symbol: {sym}")

    # ── Step 4: Smoke-test the data contract ──────────────────────────
    _banner("4. Data contract smoke test")

    try:
        fc = bridge.FeatureConfig(
            include_forecast_columns=True,
            include_actual_history_columns=False,
            use_visible_actual_history=True,
            include_delta_history_features=True,
            include_tf_moving_average_features=False,
            include_static_group_graph_features=False,
            include_weekly_history_features=False,
            include_forecast_residual_history_features=False,
            include_segment_local_stats=False,
            include_forecast_pressure_interactions=False,
            include_calendar_features=True,
            include_engineered_forecast_features=True,
        )
        _ok(f"FeatureConfig instantiated: {len(fc.__dataclass_fields__)} fields")
    except Exception as exc:
        _fail(f"FeatureConfig instantiation failed: {exc}")
        failures.append("FeatureConfig instantiation failed")

    try:
        import pandas as pd

        sample_df = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2026-01-15 01:00:00",
                "2026-01-15 02:00:00",
                "2026-01-15 03:00:00",
            ]),
        })
        result_df = bridge.add_business_time_columns(sample_df, "timestamp")
        _ok(
            f"add_business_time_columns processed {len(result_df)} rows; "
            f"columns: {list(result_df.columns)}"
        )
        # Validate 01:00 -> business_day = same day, target_hour = 1
        row = result_df.iloc[0]
        assert row["target_hour"] == 1, f"Expected target_hour=1, got {row['target_hour']}"
        assert row["business_day"] == pd.Timestamp("2026-01-15")
        _ok("Business-day alignment verified (01:00 -> same day, hour=1)")
    except Exception as exc:
        _fail(f"Data contract smoke test failed: {exc}")
        traceback.print_exc()
        failures.append("data contract smoke test failed")

    try:
        # Verify column constants are non-empty strings
        assert isinstance(bridge.TIMESTAMP_COL, str) and len(bridge.TIMESTAMP_COL) > 0
        assert isinstance(bridge.DA_COL, str) and len(bridge.DA_COL) > 0
        assert isinstance(bridge.RT_COL, str) and len(bridge.RT_COL) > 0
        _ok(
            f"Column constants: TIMESTAMP={bridge.TIMESTAMP_COL!r}, "
            f"DA={bridge.DA_COL!r}, RT={bridge.RT_COL!r}"
        )
    except Exception as exc:
        _fail(f"Column constant check failed: {exc}")
        failures.append("column constant check failed")

    # ── Summary ────────────────────────────────────────────────────────
    _banner("Summary")
    if failures:
        _fail(f"{len(failures)} failure(s) detected:")
        for f in failures:
            print(f"         - {f}")
        return 1
    else:
        _ok("All checks passed.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
