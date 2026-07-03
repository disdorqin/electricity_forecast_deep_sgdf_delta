"""SGDFNet prediction file loader for TrendKnightRT.

Loads and validates SGDFNet prediction CSV files, auto-detecting
column names and ensuring proper business-day alignment.

Supported column name variants:

    Timestamp:  ``ds``, ``timestamp``, ``time``, ``时刻``
    Prediction: ``sgdfnet_pred``, ``pred``, ``prediction``, ``y_pred``, ``rt_pred``

Output:
    A normalized DataFrame with columns ``ds``, ``business_day``,
    ``hour_business``, ``sgdfnet_pred``, and a coverage report.

Usage::

    loader = SGDFNetPredictionLoader()
    pred_df, report = loader.load("sgdfnet_predictions.csv")
    print(report["coverage_pct"])
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .business_time import add_business_time_columns

logger = logging.getLogger(__name__)

# ── Column aliases ─────────────────────────────────────────────────────

TIMESTAMP_ALIASES = ["ds", "timestamp", "time", "时刻"]
PREDICTION_ALIASES = ["sgdfnet_pred", "pred", "prediction", "y_pred", "rt_pred"]


@dataclass
class CoverageReport:
    """Coverage report for SGDFNet predictions."""

    total_rows: int = 0
    matched_rows: int = 0
    unmatched_rows: int = 0
    coverage_pct: float = 0.0
    n_unique_days: int = 0
    date_range: tuple[str, str] | None = None
    missing_dates: list[str] = field(default_factory=list)
    has_duplicates: bool = False
    fallback_required: bool = False
    source_file: str = ""


class SGDFNetPredictionLoader:
    """Load, validate, and align SGDFNet predictions from CSV.

    Typical usage::

        loader = SGDFNetPredictionLoader()
        pred_df, report = loader.load("sgdfnet_predictions.csv")
        if report.coverage_pct < 95.0:
            raise ValueError(
                f"SGDFNet coverage too low: {report.coverage_pct:.1f}%"
            )
    """

    def __init__(self, require_coverage: float = 95.0):
        self.require_coverage = require_coverage

    def load(
        self,
        filepath: str | Path,
        expected_dates: list[pd.Timestamp] | None = None,
    ) -> tuple[pd.DataFrame, CoverageReport]:
        """Load and validate an SGDFNet prediction CSV file.

        Args:
            filepath: Path to the CSV file.
            expected_dates: Optional list of expected business days.
                If provided, the report includes missing dates.

        Returns:
            ``(predictions_df, report)`` where *predictions_df* has columns
            ``ds``, ``business_day``, ``hour_business``, ``sgdfnet_pred``.

        Raises:
            FileNotFoundError: If *filepath* does not exist.
            ValueError: If required columns cannot be identified.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"SGDFNet predictions file not found: {path}")

        # Try UTF-8-sig first, fall back to GBK
        try:
            raw = pd.read_csv(path, encoding="utf-8-sig")
        except (UnicodeDecodeError, pd.errors.ParserError):
            raw = pd.read_csv(path, encoding="gbk")

        logger.info("Loaded SGDFNet predictions: %d rows from %s", len(raw), path)
        return self._process(raw, path, expected_dates)

    def _process(
        self,
        raw: pd.DataFrame,
        source: Path | str = "",
        expected_dates: list[pd.Timestamp] | None = None,
    ) -> tuple[pd.DataFrame, CoverageReport]:
        """Process a raw DataFrame through the prediction pipeline."""
        df = raw.copy()
        report = CoverageReport(source_file=str(source))

        # ── Step 1: Identify timestamp column ──────────────────────
        ts_col = None
        for alias in TIMESTAMP_ALIASES:
            if alias in df.columns:
                ts_col = alias
                break
        if ts_col is None:
            # Try the first column as a guess
            ts_col = df.columns[0]
            logger.warning("No recognized timestamp column — using '%s'", ts_col)

        # Rename to canonical "ds"
        if ts_col != "ds":
            df = df.rename(columns={ts_col: "ds"})

        # Parse timestamp
        df["ds"] = pd.to_datetime(df["ds"])
        report.total_rows = len(df)

        # ── Step 2: Identify prediction column ─────────────────────
        pred_col = None
        for alias in PREDICTION_ALIASES:
            if alias in df.columns:
                pred_col = alias
                break
        if pred_col is None:
            raise ValueError(
                f"Cannot identify SGDFNet prediction column. "
                f"Expected one of {PREDICTION_ALIASES}. "
                f"Available columns: {list(df.columns)}"
            )

        # Rename to canonical
        if pred_col != "sgdfnet_pred":
            df = df.rename(columns={pred_col: "sgdfnet_pred"})
        df["sgdfnet_pred"] = pd.to_numeric(df["sgdfnet_pred"], errors="coerce")

        # ── Step 3: Drop rows with NaN predictions ─────────────────
        before = len(df)
        df = df.dropna(subset=["sgdfnet_pred"])
        report.matched_rows = len(df)
        report.unmatched_rows = before - len(df)

        # ── Step 4: Add business-day alignment ─────────────────────
        if "business_day" not in df.columns or "hour_business" not in df.columns:
            df = add_business_time_columns(df, timestamp_col="ds")

        # ── Step 5: Deduplicate ─────────────────────────────────────
        dup_before = len(df)
        df = df.drop_duplicates(subset=["business_day", "hour_business"])
        report.has_duplicates = dup_before > len(df)
        if report.has_duplicates:
            logger.warning("Removed %d duplicate (business_day, hour_business) rows",
                           dup_before - len(df))

        # ── Step 6: Coverage computation ───────────────────────────
        # Coverage is matched_rows / total_rows * 100 (baseline: against input rows)
        if expected_dates:
            expected_hours = len(expected_dates) * 24
        else:
            expected_hours = report.total_rows

        report.coverage_pct = (
            report.matched_rows / expected_hours * 100 if expected_hours > 0 else 0.0
        )
        report.n_unique_days = int(df["business_day"].nunique())

        # Date range
        if not df.empty:
            report.date_range = (
                str(df["ds"].min().date()),
                str(df["ds"].max().date()),
            )

        # Missing dates (if expected dates given)
        if expected_dates:
            present_days = set(df["business_day"].unique())
            missing_days = sorted(
                set(pd.DatetimeIndex(expected_dates).normalize()) - present_days
            )
            report.missing_dates = [str(d.date()) for d in missing_days[:20]]

        # Fallback flag
        report.fallback_required = report.coverage_pct < 100.0

        # ── Step 7: Validate coverage ──────────────────────────────
        if report.coverage_pct < self.require_coverage:
            logger.error(
                "SGDFNet coverage %.1f%% < required %.1f%%. "
                "Formal training not allowed.",
                report.coverage_pct, self.require_coverage,
            )

        # Select canonical columns
        output_cols = ["ds", "business_day", "hour_business", "sgdfnet_pred"]
        df = df[[c for c in output_cols if c in df.columns]]

        logger.info(
            "SGDFNet predictions processed: %d rows (%d unique days), "
            "coverage=%.1f%%",
            len(df), report.n_unique_days, report.coverage_pct,
        )

        return df, report


def save_coverage_report(
    report: CoverageReport,
    output_dir: str | Path,
    prefix: str = "sgdfnet_prediction_coverage",
) -> None:
    """Save coverage report as JSON and Markdown."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON
    report_dict = {
        "total_rows": report.total_rows,
        "matched_rows": report.matched_rows,
        "unmatched_rows": report.unmatched_rows,
        "coverage_pct": round(report.coverage_pct, 1),
        "n_unique_days": report.n_unique_days,
        "date_range": report.date_range,
        "missing_dates": report.missing_dates[:20],
        "has_duplicates": report.has_duplicates,
        "fallback_required": report.fallback_required,
        "source_file": report.source_file,
        "formal_training_allowed": report.coverage_pct >= 95.0,
    }
    with open(out / f"{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    # Markdown
    md_lines = [
        f"# SGDFNet Prediction Coverage Report",
        f"",
        f"- **Source file**: `{report.source_file}`",
        f"- **Total rows**: {report.total_rows}",
        f"- **Matched rows**: {report.matched_rows}",
        f"- **Unmatched/NaN rows**: {report.unmatched_rows}",
        f"- **Coverage**: **{report.coverage_pct:.1f}%**",
        f"- **Unique days**: {report.n_unique_days}",
        f"- **Date range**: {report.date_range}",
        f"- **Duplicates removed**: {report.has_duplicates}",
        f"- **Fallback required**: {report.fallback_required}",
        f"",
        f"## Formal Training",
        f"",
    ]
    if report.coverage_pct >= 95.0:
        md_lines.append("✅ **Coverage >= 95% — formal training ALLOWED.**")
    else:
        md_lines.append(
            f"❌ **Coverage {report.coverage_pct:.1f}% < 95% — "
            f"formal training BLOCKED.**"
        )
    md_lines.append("")
    if report.missing_dates:
        md_lines.append(f"### Missing Dates ({len(report.missing_dates)})")
        for d in report.missing_dates[:20]:
            md_lines.append(f"- {d}")

    with open(out / f"{prefix}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info("Coverage report saved to %s", out)
