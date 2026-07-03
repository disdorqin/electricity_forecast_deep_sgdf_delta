#!/usr/bin/env python
"""Realtime feature audit script (Phase DeepFinal-2).

Audits the feature pipeline for TrendKnightRT by:
1. Loading raw Shandong PMOS data.
2. Building full features via ``realtime_feature_builder``.
3. Running the feature contract audit.
4. Generating audit reports (JSON, MD, CSV).

Usage:
    python scripts/audit_realtime_features.py \\
        --data-path ../data/shandong_pmos_hourly.csv \\
        --sgdfnet-predictions ../path/to/sgdfnet_preds.csv \\
        --out-dir reports/local/deep_final/features

    # Audit-only (no predictions, no training):
    python scripts/audit_realtime_features.py \\
        --data-path ../data/shandong_pmos_hourly.csv \\
        --allow-sgdfnet-fallback \\
        --out-dir reports/local/deep_final/features
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.deep_sgdf_delta.realtime_feature_builder import (
    audit_feature_coverage,
    build_realtime_features,
)
from models.deep_sgdf_delta.realtime_feature_contract import (
    ALL_FEATURES,
    OPTIONAL_FEATURES,
    REQUIRED_FEATURES,
)
from models.deep_sgdf_delta.realtime_column_mapping import (
    audit_chinese_column_mapping,
    rename_chinese_columns,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("audit_realtime_features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime feature audit for TrendKnightRT"
    )
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to hourly CSV data file")
    parser.add_argument("--sgdfnet-predictions", type=str, default=None,
                        help="Path to CSV with SGDFNet predictions")
    parser.add_argument("--allow-sgdfnet-fallback", action="store_true",
                        help="Allow sgdfnet_pred fallback to da_anchor")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory for audit reports")
    return parser.parse_args()


def generate_audit_report(audit: dict, cn_audit: dict, data_shape: tuple) -> str:
    """Generate a Markdown audit report from the audit dict."""
    lines = []
    lines.append("# Realtime Feature Audit Report")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")
    lines.append(f"## Verdict: **{audit['verdict']}**")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Data shape**: {data_shape[0]} rows x {data_shape[1]} cols")
    lines.append(f"- **Total features (contract)**: {audit['n_features']}")
    lines.append(f"- **Required features present**: {len(audit['required_present'])}")
    lines.append(f"- **Required features missing**: {len(audit['required_missing'])}")
    lines.append(f"- **Optional features present**: {audit['n_optional_present']}")
    lines.append("")
    lines.append("## SGDFNet Coverage")
    lines.append("")
    lines.append(f"- **Real coverage**: {audit['sgdfnet_real_coverage']:.1f}%")
    lines.append(f"- **Effective coverage**: {audit['sgdfnet_effective_coverage']:.1f}%")
    lines.append(f"- **Missing rows**: {audit['sgdfnet_missing_rows']}")
    lines.append(f"- **Fallback used**: {audit['sgdfnet_fallback_used']}")
    lines.append(f"- **Fallback count**: {audit.get('sgdfnet_fallback_count', 0)}")
    lines.append(f"- **Source**: {audit.get('sgdfnet_source', 'N/A')}")
    lines.append("")
    lines.append("## Calendar Features")
    lines.append("")
    lines.append(f"- **Generated**: {audit['calendar_feature_generated']}")
    lines.append(f"- **Present**: {audit.get('calendar_features_present', [])}")
    lines.append("")
    lines.append("## Lag Features")
    lines.append("")
    lines.append(f"- **Coverage**: {audit['lag_feature_coverage']:.0%}")
    lines.append(f"- **Present**: {audit.get('lag_features_present', [])}")
    lines.append("")
    lines.append("## Required Missing")
    lines.append("")
    if audit["required_missing"]:
        lines.append("The following required features are MISSING:")
        for feat in audit["required_missing"]:
            lines.append(f"- `{feat}`")
    else:
        lines.append("All required features are present. ✓")
    lines.append("")
    lines.append("## Chinese Column Mapping")
    lines.append("")
    lines.append(f"- **Mapped**: {cn_audit.get('n_mapped', 'N/A')}")
    lines.append(f"- **Unmapped**: {cn_audit.get('n_unmapped', 'N/A')}")
    if cn_audit.get("unmapped_cn_columns"):
        lines.append(f"- **Unmapped columns**: {cn_audit['unmapped_cn_columns']}")
    lines.append("")
    lines.append("## Leakage Check")
    lines.append("")
    lines.append(f"- **Leakage OK**: {audit['leakage_checked']}")
    lines.append("")
    lines.append("## Formal Training Readiness")
    lines.append("")
    if audit["formal_train_ready"]:
        lines.append("✓ **Ready for formal training.**")
    elif audit["verdict"] == "PARTIAL_READY":
        lines.append("⚠ **Partial readiness — requires attention before formal training.**")
    else:
        lines.append("✗ **Not ready for formal training.**")
    lines.append("")
    lines.append("## Feature Version")
    lines.append(f"- **{audit.get('feature_version', 'N/A')}**")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    # Resolve output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = PROJECT_ROOT / "reports" / "local" / "deep_final" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data from %s", args.data_path)
    # Try utf-8-sig first, fall back to gbk
    try:
        df = pd.read_csv(args.data_path, encoding="utf-8-sig")
    except (UnicodeDecodeError, pd.errors.ParserError):
        logger.info("utf-8-sig failed, retrying with gbk encoding")
        df = pd.read_csv(args.data_path, encoding="gbk")
    logger.info("Raw data: %d rows x %d columns", len(df), len(df.columns))

    # --- Chinese column audit ---
    cn_audit = audit_chinese_column_mapping(df)
    logger.info(
        "Chinese columns: %d mapped, %d unmapped",
        cn_audit["n_mapped"], cn_audit["n_unmapped"],
    )

    # --- Feature building ---
    logger.info("Building full features...")
    try:
        sgd_preds = None
        if args.sgdfnet_predictions:
            sgd_preds = pd.read_csv(args.sgdfnet_predictions)
            logger.info("Loaded SGDFNet predictions: %d rows", len(sgd_preds))

        feature_df = build_realtime_features(
            df,
            sgdfnet_pred_df=sgd_preds,
            mode="FULL_DAY",
            allow_sgdfnet_fallback=args.allow_sgdfnet_fallback,
        )
        logger.info("Feature building complete: %d rows x %d columns",
                     len(feature_df), len(feature_df.columns))
    except Exception as e:
        logger.error("Feature building failed: %s", e)
        # Generate a minimal report
        audit = {
            "n_features": 0,
            "required_present": [],
            "required_missing": REQUIRED_FEATURES,
            "n_required_missing": len(REQUIRED_FEATURES),
            "optional_present": [],
            "n_optional_present": 0,
            "sgdfnet_coverage": 0.0,
            "sgdfnet_missing_rows": 0,
            "sgdfnet_fallback_used": False,
            "calendar_feature_generated": False,
            "calendar_features_present": [],
            "lag_feature_coverage": 0.0,
            "lag_features_present": [],
            "leakage_checked": False,
            "formal_train_ready": False,
            "verdict": "FEATURE_BUILD_FAILED",
            "error": str(e),
        }
        # Still generate the report
        report_md = generate_audit_report(audit, cn_audit, df.shape)

        with open(out_dir / "realtime_feature_audit.json", "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, default=str)
        with open(out_dir / "realtime_feature_audit.md", "w", encoding="utf-8") as f:
            f.write(report_md)

        logger.error("Feature audit failed. Check %s for details.", out_dir)
        sys.exit(1)

    # --- Audit ---
    audit = audit_feature_coverage(feature_df)

    # --- Coverage CSV ---
    coverage_rows = []
    for col in ALL_FEATURES:
        present = col in feature_df.columns
        coverage_rows.append({
            "feature": col,
            "present": present,
            "required": col in REQUIRED_FEATURES,
            "optional": col in OPTIONAL_FEATURES,
            "non_null_count": int(feature_df[col].notna().sum()) if present else 0,
            "non_null_pct": round(
                feature_df[col].notna().mean() * 100, 1
            ) if present else 0.0,
            "mean": round(float(feature_df[col].mean()), 2) if present else None,
            "std": round(float(feature_df[col].std()), 2) if present else None,
        })

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(out_dir / "feature_coverage.csv", index=False, encoding="utf-8-sig")

    # --- Reports ---
    report_md = generate_audit_report(audit, cn_audit, df.shape)

    with open(out_dir / "realtime_feature_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)
    with open(out_dir / "realtime_feature_audit.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    # --- Summary ---
    print()
    print("=" * 60)
    print("  Feature Audit Complete")
    print("=" * 60)
    print(f"  Verdict:           {audit['verdict']}")
    print(f"  n_features:        {audit['n_features']}")
    print(f"  Required missing:  {len(audit['required_missing'])}")
    print(f"  SGDFNet real cov:  {audit['sgdfnet_real_coverage']:.1f}%")
    print(f"  SGDFNet eff cov:   {audit['sgdfnet_effective_coverage']:.1f}%")
    print(f"  Fallback used:     {audit['sgdfnet_fallback_used']}")
    print(f"  Calendar OK:       {audit['calendar_feature_generated']}")
    print(f"  Lag coverage:      {audit['lag_feature_coverage']:.0%}")
    print(f"  Leakage OK:        {audit['leakage_checked']}")
    print(f"  Formal ready:      {audit['formal_train_ready']}")
    print(f"  Reports:           {out_dir}")
    print("=" * 60)

    if audit["verdict"] in ("NOT_READY", "FALLBACK_READY"):
        logger.warning(
            "Feature pipeline %s for formal training (verdict=%s). "
            "Missing %d required features. %s",
            "NOT READY" if audit["verdict"] == "NOT_READY" else "FALLBACK ONLY",
            audit["verdict"],
            len(audit["required_missing"]),
            f"SGDFNet real coverage: {audit['sgdfnet_real_coverage']:.1f}% "
            f"(effective: {audit['sgdfnet_effective_coverage']:.1f}%)",
        )


if __name__ == "__main__":
    main()
