#!/usr/bin/env python
"""Export deep trend predictions as a fusion pack for the mainline system.

Loads TrendKnight-X v3 predictions, maps them to the fusion schema defined
in ``models.deep_sgdf_delta.fusion_export``, validates, and writes:
  - fusion_pack.csv       (the fusion pack in canonical column order)
  - manifest.json         (metadata, row counts, validation results)

Usage:
    python scripts/export_deep_trend_for_fusion.py \\
        --predictions reports/local/phase3/month_2026_03/champion_predictions.csv \\
        --model-name trendknight_x \\
        --out-dir reports/local/phase3/fusion_export

    python scripts/export_deep_trend_for_fusion.py \\
        --predictions reports/local/phase3/v3_predictions.csv \\
        --model-name trendknight_x \\
        --teacher-status-json outputs/teacher_status.json \\
        --runtime-profile v3_teacher_residual \\
        --include-eval \\
        --out-dir reports/local/phase3/fusion_export

    python scripts/export_deep_trend_for_fusion.py --help
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

# ── Path setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.deep_sgdf_delta.fusion_export import (  # noqa: E402
    FUSION_COLUMNS,
    validate_fusion_pack,
    convert_predictions_to_fusion,
    add_eval_columns,
    strip_eval_columns,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_deep_trend_for_fusion")


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export deep trend predictions as a fusion pack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Output files:\n"
            "  fusion_pack.csv     Fusion pack in canonical column order\n"
            "  manifest.json       Metadata and validation results\n"
        ),
    )
    parser.add_argument(
        "--predictions", type=str, required=True,
        help="Path to v3 predictions CSV (from predict_v3 or champion output)",
    )
    parser.add_argument(
        "--model-name", type=str, default="trendknight_x",
        help="Model name to write into fusion pack (default: trendknight_x)",
    )
    parser.add_argument(
        "--teacher-status-json", type=str, default=None,
        help="Path to teacher status JSON (maps teacher name -> availability)",
    )
    parser.add_argument(
        "--runtime-profile", type=str, default="v3_teacher_residual",
        help="Runtime profile name (default: v3_teacher_residual)",
    )
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="Output directory for fusion_pack.csv + manifest.json",
    )
    parser.add_argument(
        "--include-eval", action="store_true",
        help="Include eval columns (y_true, residuals) in the fusion pack",
    )
    return parser.parse_args()


# ── Teacher status loading ───────────────────────────────────────────

def load_teacher_status(path: str | None) -> dict | None:
    """Load teacher status from a JSON file.

    Expected format::

        {
            "sgdfnet": {"availability": "available", "n_predictions": 720},
            "rt916": {"availability": "unavailable"},
            "timemixer": {"availability": "available", "n_predictions": 720}
        }
    """
    if path is None:
        return None

    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p

    if not p.exists():
        logger.warning("Teacher status JSON not found: %s", p)
        return None

    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    logger.info("Loaded teacher status from %s: %s", p, list(data.keys()))
    return data


# ── Predictions loading ──────────────────────────────────────────────

def load_predictions(path: str) -> pd.DataFrame:
    """Load predictions CSV, trying multiple path resolutions."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p

    if not p.exists():
        raise FileNotFoundError(f"Predictions file not found: {path} (resolved to {p})")

    logger.info("Loading predictions from %s", p)
    ext = p.suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    elif ext == ".csv":
        df = pd.read_csv(p, encoding="utf-8-sig")
    else:
        df = pd.read_csv(p)

    logger.info("Predictions: %d rows, columns: %s", len(df), list(df.columns))
    return df


# ── Manifest builder ─────────────────────────────────────────────────

def build_manifest(
    fusion_df: pd.DataFrame,
    validation_ok: bool,
    validation_errors: list[str],
    model_name: str,
    runtime_profile: str,
    include_eval: bool,
    teacher_status: dict | None,
    source_path: str,
) -> dict:
    """Build the manifest.json content."""
    manifest: dict = {
        "timestamp": datetime.now().isoformat(),
        "source_predictions": source_path,
        "model_name": model_name,
        "runtime_profile": runtime_profile,
        "include_eval": include_eval,
        "n_rows": len(fusion_df),
        "fusion_columns": list(fusion_df.columns),
        "validation": {
            "is_valid": validation_ok,
            "errors": validation_errors,
        },
        "teacher_status": teacher_status,
        "summary": {},
    }

    # Summary statistics
    if "trend_pred" in fusion_df.columns:
        tp = pd.to_numeric(fusion_df["trend_pred"], errors="coerce")
        manifest["summary"]["trend_pred"] = {
            "mean": round(float(tp.mean()), 4) if tp.notna().any() else None,
            "min": round(float(tp.min()), 4) if tp.notna().any() else None,
            "max": round(float(tp.max()), 4) if tp.notna().any() else None,
            "std": round(float(tp.std()), 4) if tp.notna().any() else None,
        }

    if "trend_confidence" in fusion_df.columns:
        tc = pd.to_numeric(fusion_df["trend_confidence"], errors="coerce")
        manifest["summary"]["trend_confidence"] = {
            "mean": round(float(tc.mean()), 4) if tc.notna().any() else None,
            "min": round(float(tc.min()), 4) if tc.notna().any() else None,
            "max": round(float(tc.max()), 4) if tc.notna().any() else None,
        }

    if "shock_sensitivity" in fusion_df.columns:
        ss = pd.to_numeric(fusion_df["shock_sensitivity"], errors="coerce")
        manifest["summary"]["shock_sensitivity"] = {
            "mean": round(float(ss.mean()), 4) if ss.notna().any() else None,
            "min": round(float(ss.min()), 4) if ss.notna().any() else None,
            "max": round(float(ss.max()), 4) if ss.notna().any() else None,
        }

    # Date range
    if "business_day" in fusion_df.columns:
        bd = pd.to_datetime(fusion_df["business_day"])
        manifest["summary"]["date_range"] = {
            "start": str(bd.min().date()) if bd.notna().any() else None,
            "end": str(bd.max().date()) if bd.notna().any() else None,
            "n_days": int(bd.nunique()),
        }

    # Teacher availability summary
    if "teacher_used" in fusion_df.columns:
        manifest["summary"]["teacher_usage"] = (
            fusion_df["teacher_used"].value_counts().to_dict()
        )

    # Period distribution
    if "period" in fusion_df.columns:
        manifest["summary"]["period_distribution"] = (
            fusion_df["period"].value_counts().to_dict()
        )

    return manifest


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Export Deep Trend for Fusion")
    logger.info("=" * 60)
    logger.info("  Predictions  : %s", args.predictions)
    logger.info("  Model name   : %s", args.model_name)
    logger.info("  Profile      : %s", args.runtime_profile)
    logger.info("  Include eval : %s", args.include_eval)
    logger.info("  Output       : %s", out_dir)

    # Load teacher status
    teacher_status = load_teacher_status(args.teacher_status_json)

    # Load predictions
    pred_df = load_predictions(args.predictions)
    if pred_df.empty:
        logger.error("Predictions file is empty. Exiting.")
        sys.exit(1)

    # Convert to fusion format
    logger.info("Converting predictions to fusion format ...")
    fusion_df = convert_predictions_to_fusion(
        pred_df,
        model_name=args.model_name,
        runtime_profile=args.runtime_profile,
        teacher_status=teacher_status,
    )
    logger.info("Fusion pack: %d rows, %d columns", len(fusion_df), len(fusion_df.columns))

    # Optionally add eval columns
    if args.include_eval:
        if "y_true" in fusion_df.columns or "y_true" in pred_df.columns:
            # Carry y_true from predictions if present
            if "y_true" not in fusion_df.columns and "y_true" in pred_df.columns:
                fusion_df["y_true"] = pred_df["y_true"].values[:len(fusion_df)]
            try:
                fusion_df = add_eval_columns(fusion_df)
                logger.info("Added eval columns (y_true, residual_for_spike, residual_for_negative)")
            except ValueError as exc:
                logger.warning("Could not add eval columns: %s", exc)
        else:
            logger.warning("--include-eval specified but y_true not found in predictions")

    # Validate
    is_valid, errors = validate_fusion_pack(fusion_df)
    if is_valid:
        logger.info("Validation: PASSED (%d rows)", len(fusion_df))
    else:
        logger.warning("Validation: FAILED with %d errors:", len(errors))
        for err in errors:
            logger.warning("  - %s", err)

    # If eval columns were added, validation should ignore them (they're not in FUSION_COLUMNS)
    # Strip eval columns for the online pack, keep a separate eval version
    if args.include_eval:
        eval_df = fusion_df.copy()
        fusion_online = strip_eval_columns(fusion_df)
        # Write online pack
        fusion_online.to_csv(out_dir / "fusion_pack.csv", index=False, encoding="utf-8-sig")
        logger.info("fusion_pack.csv (online) -> %s", out_dir / "fusion_pack.csv")
        # Write eval pack
        eval_df.to_csv(out_dir / "fusion_pack_eval.csv", index=False, encoding="utf-8-sig")
        logger.info("fusion_pack_eval.csv (eval) -> %s", out_dir / "fusion_pack_eval.csv")
    else:
        fusion_df.to_csv(out_dir / "fusion_pack.csv", index=False, encoding="utf-8-sig")
        logger.info("fusion_pack.csv -> %s", out_dir / "fusion_pack.csv")

    # Build and write manifest
    manifest = build_manifest(
        fusion_df,
        validation_ok=is_valid,
        validation_errors=errors,
        model_name=args.model_name,
        runtime_profile=args.runtime_profile,
        include_eval=args.include_eval,
        teacher_status=teacher_status,
        source_path=args.predictions,
    )

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, default=str)
    logger.info("manifest.json -> %s", manifest_path)

    # Summary
    logger.info("=" * 60)
    logger.info("Export complete.")
    logger.info("  Rows       : %d", len(fusion_df))
    logger.info("  Valid      : %s", is_valid)
    logger.info("  Date range : %s", manifest.get("summary", {}).get("date_range", {}))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
