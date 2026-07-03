"""Canonical Chinese → English column mapping for Shandong spot market data.

This module is the SINGLE SOURCE OF TRUTH for mapping Chinese column names
found in raw Shandong PMOS hourly CSV files to the canonical English feature
names expected by the feature contract and dataset builders.

All scripts MUST use :func:`rename_chinese_columns` instead of defining
hard-coded mapping dictionaries in training / prediction scripts.

Mapping conventions:

- ``CN_CORE``: Core columns (timestamp, day-ahead price, realtime price).
- ``CN_FORECAST_MAP``: Forecast-side columns prefixed with ``"forecast_"``
  in English (e.g. ``wind_forecast``).
- ``CN_ACTUAL_MAP``: Actual-side columns prefixed with the actual variable
  name (e.g. ``wind_actual``).  These are NOT used as features unless
  explicitly allowed via a ``intraday`` mode.
- ``CN_BID_MAP``: Bidding / market-side columns.
- ``CN_LOAD_MAP``: Load / demand columns.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ── Core columns ───────────────────────────────────────────────────────
# These are the minimum required columns for any TrendKnightRT run.

CN_CORE: dict[str, str] = {
    "时刻": "ds",              # timestamp (parsed to datetime)
    "日前电价": "da_anchor",    # day-ahead price → anchor
    "日前价格": "da_anchor",    # alternative Chinese name
    "实时电价": "rt_actual",    # realtime price → target
    "实时价格": "rt_actual",    # alternative Chinese name
}

# ── Forecast-side columns ──────────────────────────────────────────────
# These are DA forecast values available before the realtime market clears.

CN_FORECAST_MAP: dict[str, str] = {
    "日前负荷预测": "load_forecast",
    "日前负荷预测值": "load_forecast",
    "统调负荷预测值": "load_forecast",
    "日前新能源预测": "renewable_forecast",
    "日前新能源预测值": "renewable_forecast",
    "新能源总加预测值": "renewable_forecast",
    "日前风电预测": "wind_forecast",
    "日前风电预测值": "wind_forecast",
    "风电总加预测值": "wind_forecast",
    "日前光伏预测": "solar_forecast",
    "日前光伏预测值": "solar_forecast",
    "光伏总加预测值": "solar_forecast",
    "联络线受电负荷预测值": "tie_line_forecast",
    "联络线": "tie_line_forecast",  # abbreviated
    "地方电厂总加预测值": "local_plant_forecast",
    "核电总加预测值": "nuclear_forecast",
    "自备机组总加预测值": "self_supply_forecast",
    "试验机组总加预测值": "test_unit_forecast",
    "直调负荷预测值": "dispatched_load_forecast",
    "竞价空间预测值": "bidding_space_forecast",
    "竞价空间": "bidding_space_forecast",
    "省内负荷": "provincial_load_forecast",
    "省内负荷预测值": "provincial_load_forecast",
}

# ── Actual-side columns ────────────────────────────────────────────────
# These columns represent *realised* values.  They MUST NOT be used as
# features in FULL_DAY mode.  In INTRADAY mode they can be used for
# same-day lag features (hour < cutoff_hour).

CN_ACTUAL_MAP: dict[str, str] = {
    "地方电厂总加实际值": "local_plant_actual",
    "联络线受电负荷实际值": "tie_line_actual",
    "风电总加实际值": "wind_actual",
    "光伏总加实际值": "solar_actual",
    "核电总加实际值": "nuclear_actual",
    "自备机组总加实际值": "self_supply_actual",
    "试验机组总加实际值": "test_unit_actual",
    "直调负荷实际值": "dispatched_load_actual",
    "竞价空间实际值": "bidding_space_actual",
    "新能源总加实际值": "renewable_actual",
    "统调负荷": "system_load_actual",
    "新能源": "renewable_actual",
    "风电": "wind_actual",
    "光伏": "solar_actual",
}

# ── Bidding / market-side columns ─────────────────────────────────────

CN_BID_MAP: dict[str, str] = {
    "竞价空间": "bidding_space_forecast",
}

# ── Load columns ───────────────────────────────────────────────────────

CN_LOAD_MAP: dict[str, str] = {
    "统调负荷": "system_load_actual",
    "省内负荷": "provincial_load_forecast",
}

# ── Combined mapping for lookup ────────────────────────────────────────

_ALL_CN_MAPS: list[dict[str, str]] = [
    CN_CORE,
    CN_FORECAST_MAP,
    CN_ACTUAL_MAP,
    CN_BID_MAP,
    CN_LOAD_MAP,
]

# Invert: English name → list of possible Chinese variants
EN_TO_CN: dict[str, list[str]] = {}
for _m in _ALL_CN_MAPS:
    for cn, en in _m.items():
        EN_TO_CN.setdefault(en, []).append(cn)


def _build_rename_map(df: pd.DataFrame) -> dict[str, str]:
    """Build a safe rename map handling duplicate English targets.

    When multiple Chinese names map to the same English name (e.g. both
    "日前电价" and "日前价格" → "da_anchor"), the first occurrence is
    renamed normally and subsequent occurrences are re-mapped to the next
    most appropriate alias (e.g. "forecast_price").

    Priority: core > forecast > actual > bid > load (first match wins).
    """
    rename_map: dict[str, str] = {}
    used_en_names: set[str] = set()

    def _safe_rename(cn: str, preferred_en: str, fallback_en: str | None = None) -> None:
        """Add rename entry, falling back to an alias if *preferred_en* is taken."""
        en = preferred_en if preferred_en not in used_en_names else (fallback_en or preferred_en)
        if en not in used_en_names or en == preferred_en:
            rename_map[cn] = en
            used_en_names.add(en)

    all_maps: list[dict[str, str]] = [CN_CORE, CN_FORECAST_MAP, CN_ACTUAL_MAP, CN_BID_MAP, CN_LOAD_MAP]
    for _m in all_maps:
        for cn_name, en_name in _m.items():
            if cn_name in df.columns and cn_name not in rename_map:
                # If en_name is already taken, try an alias
                if en_name in used_en_names:
                    # Map the second "da_anchor" to "forecast_price"
                    if en_name == "da_anchor":
                        rename_map[cn_name] = "forecast_price"
                        used_en_names.add("forecast_price")
                        continue
                rename_map[cn_name] = en_name
                used_en_names.add(en_name)

    return rename_map


def rename_chinese_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and rename Chinese column names to canonical English names.

    Scans the DataFrame columns against all known Chinese name mappings.
    Reports which columns were renamed and which known Chinese names were
    *not* found (as a diagnostic aid).

    Args:
        df: Raw DataFrame read from a Shandong PMOS CSV / XLSX file.

    Returns:
        A new DataFrame with renamed columns.  Columns that do not match
        any mapping are left unchanged.

    Example::

        df = rename_chinese_columns(raw_df)
        # df now has English column names (ds, da_anchor, rt_actual, …)
    """
    df = df.copy()
    rename_map = _build_rename_map(df)

    # Apply renaming
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(
            "Renamed %d Chinese columns to English: %s",
            len(rename_map), rename_map,
        )

    # Report unmapped Chinese-named columns
    unmapped = []
    for col in df.columns:
        if any("\u4e00" <= c <= "\u9fff" for c in str(col)):
            unmapped.append(col)
    if unmapped:
        logger.warning(
            "Unmapped Chinese columns (not renamed): %s", unmapped,
        )

    return df


def audit_chinese_column_mapping(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Audit which Chinese column mappings were found or missing.

    Args:
        df: DataFrame (before renaming) to audit.

    Returns:
        A dict with keys:

        - ``total_cn_columns``: number of Chinese-named columns found.
        - ``mapped``: list of (cn_name → en_name) tuples that were
          successfully renamed.
        - ``unmapped``: list of Chinese column names that have no mapping.
        - ``known_english_expected``: English names our contract expects
          but no Chinese variant was found in the data.
    """
    # Collect all known Chinese names
    all_cn_names: set[str] = set()
    for _m in _ALL_CN_MAPS:
        all_cn_names.update(_m.keys())

    found_cn = {c for c in df.columns if c in all_cn_names}
    mapped = {c: _resolve_en(c) for c in found_cn}

    unmapped = [
        c for c in df.columns
        if any("\u4e00" <= ch <= "\u9fff" for ch in str(c))
        and c not in all_cn_names
    ]

    # English names we have mappings for but none found in data
    all_en_wanted: set[str] = set()
    for _m in _ALL_CN_MAPS:
        all_en_wanted.update(_m.values())
    found_en = set(mapped.values())
    missing_en = sorted(all_en_wanted - found_en)

    return {
        "total_cn_columns_in_data": len(found_cn) + len(unmapped),
        "mapped": list(mapped.items()),
        "unmapped_cn_columns": sorted(unmapped),
        "known_english_not_found": missing_en,
        "n_mapped": len(mapped),
        "n_unmapped": len(unmapped),
        "n_missing_english": len(missing_en),
    }


def _resolve_en(cn_name: str) -> str | None:
    """Resolve a Chinese column name to its English equivalent."""
    for _m in _ALL_CN_MAPS:
        if cn_name in _m:
            return _m[cn_name]
    return None
