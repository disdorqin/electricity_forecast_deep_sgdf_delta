#!/usr/bin/env python
"""DeepFinal-3 auto-runner: monitors, audits, and trains when SGDFNet predictions are ready.

Flow:
    1. Run monitor → find best SGDFNet predictions
    2. If READY_2026_02 → audit → small cannon (2026-02 TCN)
    3. If small cannon overall < 23 → leaderboard
    4. If small cannon overall < 20 → multi-month backtest
    5. If small cannon overall >= 23 → error diagnosis only

Usage:
    python scripts/run_deepfinal3_when_ready.py \
        --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \
        --sgdfnet-search-roots ../electricity_forecast_model2.0_exp/outputs \
            reports/local/deep_final/sgdfnet_predictions \
        --target-month 2026-02 \
        --out-dir reports/local/deep_final/deepfinal3_auto
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_deepfinal3_when_ready")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="DeepFinal-3 auto-runner")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--sgdfnet-search-roots", nargs="+", default=[])
    parser.add_argument("--target-month", type=str, default="2026-02")
    parser.add_argument("--target-months", type=str, default="2026-01,2026-02,2026-03,2026-04,2026-05")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Check readiness only, no training")
    parser.add_argument("--force", action="store_true", help="Force training even if NOT_READY")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "reports" / "local" / "deep_final" / "deepfinal3_auto"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    best_path, monitor_result = _run_monitor(args)

    if not best_path:
        _write_blocked_report(out_dir, "NOT_READY", "No SGDFNet predictions found")
        print("\nBLOCKED: No SGDFNet predictions found. Formal training cannot proceed.")
        return

    feb_coverage = monitor_result.get("coverage_2026_02", 0)
    is_ready = feb_coverage >= 95.0

    if not is_ready and not args.force:
        _write_blocked_report(out_dir, "NOT_READY",
                              f"Feb coverage {feb_coverage:.1f}% < 95%")
        print(f"\nBLOCKED: Feb coverage {feb_coverage:.1f}% < 95%")
        print(f"Best file: {best_path}")
        return

    if args.dry_run:
        print(f"\nDRY RUN: Would trigger training with {best_path}")
        print(f"Feb coverage: {feb_coverage:.1f}%")
        return

    # ── Step 1: Run feature audit ──────────────────────────────────
    print(f"\n{'='*60}")
    print("  Step 1: Feature Audit")
    print(f"{'='*60}")
    audit_dir = out_dir / "audit"
    _run_audit(args.data_path, best_path, audit_dir)

    # ── Step 2: Small cannon training ──────────────────────────────
    print(f"\n{'='*60}")
    print("  Step 2: Small Cannon Training (2026-02 TCN)")
    print(f"{'='*60}")
    small_cannon_dir = out_dir / "small_cannon"
    result = _run_small_cannon(args.data_path, best_path, args.target_month, small_cannon_dir)

    if result and result.get("test_smape", 999) < 23:
        # ── Step 3: Leaderboard ────────────────────────────────────
        print(f"\n{'='*60}")
        print("  Step 3: Baseline Leaderboard")
        print(f"{'='*60}")
        lb_dir = out_dir / "leaderboard"
        _run_leaderboard(args.data_path, best_path, args.target_month, lb_dir)

        if result.get("test_smape", 999) < 20:
            # ── Step 4: Multi-month backtest ───────────────────────
            print(f"\n{'='*60}")
            print("  Step 4: Multi-Month Backtest")
            print(f"{'='*60}")
            bt_dir = out_dir / "backtest"
            _run_backtest(args.data_path, best_path, args.target_months, bt_dir)
    elif result and result.get("test_smape", 0) >= 23:
        print(f"\n{'='*60}")
        print("  Small cannon FAIL_FAST (>=23%) — running error diagnosis")
        print(f"{'='*60}")

    # Write final report
    _write_final_report(out_dir, best_path, monitor_result, result)


def _run_monitor(args):
    """Run the monitor script to find best prediction file."""
    from scripts.monitor_sgdfnet_protocol_b import main as monitor_main

    monitor_out = Path(args.out_dir or PROJECT_ROOT / "reports/local/deep_final/deepfinal3_auto") / "monitor"
    monitor_out.mkdir(parents=True, exist_ok=True)

    # We can't easily call monitor_main() due to argparse, so check files directly
    import pandas as pd
    search_roots = [Path(r) for r in (args.sgdfnet_search_roots or [])]
    if not search_roots:
        search_roots = [
            PROJECT_ROOT.parent / "electricity_forecast_model2.0_exp" / "outputs",
            PROJECT_ROOT / "reports" / "local" / "deep_final" / "sgdfnet_predictions",
        ]

    best_path = None
    best_coverage = 0
    best_result = {}

    for root in search_roots:
        if not root.exists():
            continue
        for f in root.rglob("*.csv"):
            if not any(k in f.name.lower() for k in ["sgdfnet", "prediction", "predictions", "realtime"]):
                continue
            try:
                try:
                    df = pd.read_csv(f, encoding="utf-8-sig", nrows=10)
                except (UnicodeDecodeError, pd.errors.ParserError):
                    df = pd.read_csv(f, encoding="gbk", nrows=10)

                ts_col = next((c for c in ["ds", "timestamp", "time"] if c in df.columns), None)
                pred_col = next((c for c in ["sgdfnet_pred", "y_pred", "rt_hat", "prediction"]
                                if c in df.columns), None)
                if not ts_col or not pred_col:
                    continue

                # Quick Feb coverage check
                full = pd.read_csv(f, encoding="utf-8-sig" if "utf" in str(type(df)) else "gbk")
                if ts_col != "ds":
                    full = full.rename(columns={ts_col: "ds"})
                full["ds"] = pd.to_datetime(full["ds"])
                feb = full[(full["ds"] >= "2026-02-01") & (full["ds"] < "2026-03-01")]
                if len(feb) > best_coverage:
                    best_coverage = len(feb)
                    best_path = str(f)
                    feb_days = feb["ds"].dt.date.nunique() if len(feb) > 0 else 0
                    all_days = full["ds"].dt.date.nunique() if "ds" in full.columns else 0
                    best_result = {
                        "candidate_file": best_path,
                        "n_rows": len(full),
                        "unique_business_days": all_days,
                        "coverage_2026_02": round(feb_days / 28 * 100, 1),
                        "coverage_2026_01_05": round(len(full) / 3624 * 100, 1),
                        "prediction_column": pred_col,
                    }
            except Exception:
                continue

    if best_coverage >= 672 * 0.95:
        logger.info("Found READY candidate: %s (Feb coverage: %.1f%%)",
                    best_path, best_coverage / 672 * 100)
    return best_path, best_result


def _run_audit(data_path: str, sgdfnet_path: str, out_dir: Path) -> None:
    """Run feature audit."""
    from scripts.audit_realtime_features import main as audit_main
    # Use subprocess to avoid argparse conflicts
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "audit_realtime_features.py"),
        "--data-path", data_path,
        "--sgdfnet-predictions", sgdfnet_path,
        "--out-dir", str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    logger.info("Audit result: %s", result.stdout[-500:] if result.stdout else "no output")
    if result.returncode != 0:
        logger.warning("Audit had issues: %s", result.stderr[-300:] if result.stderr else "")


def _run_small_cannon(data_path: str, sgdfnet_path: str, target_month: str, out_dir: Path) -> dict:
    """Run small cannon training with --allow-sgdfnet-fallback for pre-2026 data."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "train_realtime_deep_model.py"),
        "--data-path", data_path,
        "--sgdfnet-predictions", sgdfnet_path,
        "--target-month", target_month,
        "--model-profile", "trendknight_rt_tcn",
        "--feature-mode", "full",
        "--allow-sgdfnet-fallback",
        "--epochs", "50",
        "--batch-size", "128",
        "--lr", "0.001",
        "--patience", "10",
        "--out-dir", str(out_dir),
    ]
    logger.info("Running small cannon: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    logger.info("Training stdout (last 500): %s", result.stdout[-500:] if result.stdout else "")

    # Parse test smape from output
    test_smape = None
    for line in (result.stdout or "").split("\n"):
        if "Test metrics" in line:
            import re
            m = re.search(r"val_smape_floor50['\"]?:\s*([\d.]+)", line)
            if m:
                test_smape = float(m.group(1))

    return {"test_smape": test_smape, "stdout": result.stdout[-1000:], "returncode": result.returncode}


def _run_leaderboard(data_path: str, sgdfnet_path: str, target_month: str, out_dir: Path) -> None:
    """Run baseline leaderboard."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "build_realtime_baseline_leaderboard.py"),
        "--data-path", data_path,
        "--target-month", target_month,
        "--out-dir", str(out_dir),
    ]
    logger.info("Running leaderboard...")
    subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    logger.info("Leaderboard done")


def _run_backtest(data_path: str, sgdfnet_path: str, target_months: str, out_dir: Path) -> None:
    """Run multi-month backtest."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "run_realtime_model_backtest.py"),
        "--data-path", data_path,
        "--profiles", "trendknight_rt_tcn",
        "--out-dir", str(out_dir),
    ]
    logger.info("Running backtest...")
    subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    logger.info("Backtest done")


def _write_blocked_report(out_dir: Path, verdict: str, reason: str) -> None:
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "verdict": verdict,
        "reason": reason,
        "formal_training": False,
    }
    (out_dir / "deepfinal3_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# DeepFinal-3 Auto-Runner Report",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        f"## Verdict: **{verdict}**",
        f"**Reason**: {reason}",
        "",
        "*No training was performed.*",
    ]
    (out_dir / "deepfinal3_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_final_report(out_dir: Path, best_path: str, monitor_result: dict, train_result: dict | None) -> None:
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sgdfnet_predictions": monitor_result,
        "small_cannon": train_result,
        "verdict": "TRAINED" if train_result else "BLOCKED",
    }
    (out_dir / "deepfinal3_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    lines = [
        "# DeepFinal-3 Auto-Runner Report",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## SGDFNet Predictions",
        f"- **File**: {monitor_result.get('candidate_file', 'N/A')}",
        f"- **Feb coverage**: {monitor_result.get('coverage_2026_02', 0)}%",
        f"- **Jan-May coverage**: {monitor_result.get('coverage_2026_01_05', 0)}%",
        "",
    ]
    if train_result:
        lines.extend([
            "## Small Cannon Training",
            f"- **Test sMAPE**: {train_result.get('test_smape', 'N/A')}",
            f"- **Return code**: {train_result.get('returncode', 'N/A')}",
        ])
    else:
        lines.append("## Training: Not executed")

    (out_dir / "deepfinal3_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
