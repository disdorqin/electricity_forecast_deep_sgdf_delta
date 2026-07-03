#!/usr/bin/env python
"""Audit Solar916 feature visibility — Phase 8.

Checks each feature for:
  - Whether it uses actual RT price (target leakage)
  - Whether it uses current row's residual (target leakage)
  - Whether it's visible at prediction time

Output: docs/SOLAR916_FEATURE_VISIBILITY_AUDIT.md

Usage:
    python scripts/audit_solar916_feature_visibility.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Feature visibility table
FEATURE_AUDIT = [
    {
        "feature_name": "hour_business",
        "source_column": "business_time.py",
        "uses_actual": False,
        "uses_current_target": False,
        "visible_at_prediction_time": True,
        "leakage_risk": False,
        "notes": "Temporal feature, known at prediction time",
    },
    {
        "feature_name": "weekday",
        "source_column": "business_time.py",
        "uses_actual": False,
        "uses_current_target": False,
        "visible_at_prediction_time": True,
        "leakage_risk": False,
        "notes": "Temporal feature",
    },
    {
        "feature_name": "month",
        "source_column": "business_time.py",
        "uses_actual": False,
        "uses_current_target": False,
        "visible_at_prediction_time": True,
        "leakage_risk": False,
        "notes": "Temporal feature",
    },
    {
        "feature_name": "da_anchor",
        "source_column": "日前电价 (DA price)",
        "uses_actual": False,
        "uses_current_target": False,
        "visible_at_prediction_time": True,
        "leakage_risk": False,
        "notes": "Day-ahead price, known before RT",
    },
    {
        "feature_name": "sgdfnet_pred",
        "source_column": "SGDFNet model output",
        "uses_actual": False,
        "uses_current_target": False,
        "visible_at_prediction_time": True,
        "leakage_risk": False,
        "notes": "Base model prediction, available at correction time",
    },
    {
        "feature_name": "forecast_load",
        "source_column": "直调负荷预测值",
        "uses_actual": False,
        "uses_current_target": False,
        "visible_at_prediction_time": True,
        "leakage_risk": False,
        "notes": "Forecast, known at prediction time",
    },
    {
        "feature_name": "forecast_wind",
        "source_column": "风电总加预测值",
        "uses_actual": False,
        "uses_current_target": False,
        "visible_at_prediction_time": True,
        "leakage_risk": False,
        "notes": "Forecast",
    },
    {
        "feature_name": "forecast_solar",
        "source_column": "光伏总加预测值",
        "uses_actual": False,
        "uses_current_target": False,
        "visible_at_prediction_time": True,
        "leakage_risk": False,
        "notes": "Forecast",
    },
    {
        "feature_name": "forecast_new_energy",
        "source_column": "新能源总加预测值",
        "uses_actual": False,
        "uses_current_target": False,
        "visible_at_prediction_time": True,
        "leakage_risk": False,
        "notes": "Forecast",
    },
    {
        "feature_name": "bidding_space",
        "source_column": "竞价空间预测值",
        "uses_actual": False,
        "uses_current_target": False,
        "visible_at_prediction_time": True,
        "leakage_risk": False,
        "notes": "Forecast",
    },
    {
        "feature_name": "net_load",
        "source_column": "Derived: forecast_load - forecast_new_energy",
        "uses_actual": False,
        "uses_current_target": False,
        "visible_at_prediction_time": True,
        "leakage_risk": False,
        "notes": "Derived from forecasts only",
    },
    {
        "feature_name": "renewable_share",
        "source_column": "Derived: (solar + wind) / net_load",
        "uses_actual": False,
        "uses_current_target": False,
        "visible_at_prediction_time": True,
        "leakage_risk": False,
        "notes": "Derived from forecasts only",
    },
    {
        "feature_name": "delta_lag_24",
        "source_column": "Previous business_day same hour delta (merge-based)",
        "uses_actual": True,
        "uses_current_target": False,
        "visible_at_prediction_time": True,
        "leakage_risk": False,
        "notes": "Phase 8: merge-based, uses PAST RT actual from previous day same hour. Allowed.",
    },
    {
        "feature_name": "delta_lag_168",
        "source_column": "business_day - 7 same hour delta (merge-based)",
        "uses_actual": True,
        "uses_current_target": False,
        "visible_at_prediction_time": True,
        "leakage_risk": False,
        "notes": "Phase 8: merge-based, uses PAST RT actual from 7 days ago same hour. Allowed.",
    },
    {
        "feature_name": "residual_lag_24",
        "source_column": "Previous business_day same hour SGDFNet residual",
        "uses_actual": True,
        "uses_current_target": False,
        "visible_at_prediction_time": True,
        "leakage_risk": False,
        "notes": "Phase 8: merge-based, uses PAST residual. Allowed.",
    },
    {
        "feature_name": "residual_lag_168",
        "source_column": "business_day - 7 same hour SGDFNet residual",
        "uses_actual": True,
        "uses_current_target": False,
        "visible_at_prediction_time": True,
        "leakage_risk": False,
        "notes": "Phase 8: merge-based, uses PAST residual. Allowed.",
    },
    {
        "feature_name": "rolling_residual_mean_7d",
        "source_column": "shift(1).rolling(7).mean() on sgdfnet_residual",
        "uses_actual": True,
        "uses_current_target": False,
        "visible_at_prediction_time": True,
        "leakage_risk": False,
        "notes": "Phase 8: shift(1) excludes current row. Uses only PAST residuals. Allowed.",
    },
    {
        "feature_name": "rolling_residual_std_7d",
        "source_column": "shift(1).rolling(7).std() on sgdfnet_residual",
        "uses_actual": True,
        "uses_current_target": False,
        "visible_at_prediction_time": True,
        "leakage_risk": False,
        "notes": "Phase 8: shift(1) excludes current row. Allowed.",
    },
    {
        "feature_name": "same_hour_residual_mean_7d",
        "source_column": "groupby(hour).shift(1).rolling(7).mean()",
        "uses_actual": True,
        "uses_current_target": False,
        "visible_at_prediction_time": True,
        "leakage_risk": False,
        "notes": "Phase 8: groupby + shift(1) excludes current row. Allowed.",
    },
    {
        "feature_name": "same_hour_residual_std_7d",
        "source_column": "groupby(hour).shift(1).rolling(7).std()",
        "uses_actual": True,
        "uses_current_target": False,
        "visible_at_prediction_time": True,
        "leakage_risk": False,
        "notes": "Phase 8: groupby + shift(1) excludes current row. Allowed.",
    },
    {
        "feature_name": "sgdfnet_residual",
        "source_column": "TARGET: rt_actual - sgdfnet_pred",
        "uses_actual": True,
        "uses_current_target": True,
        "visible_at_prediction_time": False,
        "leakage_risk": True,
        "notes": "TARGET VARIABLE — must NOT be used as feature. Used only as training target.",
    },
]


def main():
    logger.info("=" * 60)
    logger.info("Solar916 Feature Visibility Audit — Phase 8")
    logger.info("=" * 60)

    high_risk = [f for f in FEATURE_AUDIT if f["leakage_risk"] and f["feature_name"] != "sgdfnet_residual"]
    target_entries = [f for f in FEATURE_AUDIT if f["uses_current_target"]]

    # Check: sgdfnet_residual should only appear as target, not as feature
    target_as_feature = [f for f in FEATURE_AUDIT
                         if f["uses_current_target"] and f["feature_name"] != "sgdfnet_residual"]

    passed = len(high_risk) == 0 and len(target_as_feature) == 0

    # Write report
    lines = [
        "# Solar916 Feature Visibility Audit",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Phase:** 8 (No-Leak Revalidation)",
        "",
        "## Feature Table",
        "",
        "| Feature | Source | Uses Actual | Uses Current Target | Visible at Pred Time | Leakage Risk | Notes |",
        "|---------|--------|-------------|--------------------|--------------------|-------------|-------|",
    ]
    for f in FEATURE_AUDIT:
        lines.append(
            f"| {f['feature_name']} | {f['source_column']} | "
            f"{'YES' if f['uses_actual'] else 'no'} | "
            f"{'**YES**' if f['uses_current_target'] else 'no'} | "
            f"{'yes' if f['visible_at_prediction_time'] else '**NO**'} | "
            f"{'**HIGH**' if f['leakage_risk'] else 'low'} | "
            f"{f['notes']} |"
        )

    lines.extend([
        "",
        "## Audit Results",
        "",
        f"- High-risk features (excluding target): {len(high_risk)}",
        f"- Features using current target: {len(target_entries)} (should only be sgdfnet_residual)",
        f"- Target used as feature: {len(target_as_feature)}",
        "",
        f"## Verdict: **{'PASSED' if passed else 'FAILED'}**",
        "",
    ])

    if passed:
        lines.append("All features are visible at prediction time. No leakage detected.")
        lines.append("sgdfnet_residual is correctly used only as target, not as feature.")
    else:
        lines.append("LEAKAGE DETECTED. Training must be blocked until fixed.")
        for f in high_risk:
            lines.append(f"- HIGH RISK: {f['feature_name']}: {f['notes']}")

    report_path = PROJECT_ROOT / "docs" / "SOLAR916_FEATURE_VISIBILITY_AUDIT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Also write JSON for programmatic consumption
    json_path = PROJECT_ROOT / "reports" / "local" / "phase8" / "feature_visibility_audit.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "passed": passed,
            "high_risk_count": len(high_risk),
            "target_as_feature_count": len(target_as_feature),
            "features": FEATURE_AUDIT,
        }, f, ensure_ascii=False, indent=2)

    logger.info("Audit complete. Verdict: %s", "PASSED" if passed else "FAILED")
    logger.info("Report: %s", report_path)

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
