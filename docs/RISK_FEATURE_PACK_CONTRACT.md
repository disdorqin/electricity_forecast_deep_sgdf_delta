# Risk Feature Pack Contract

> Unified risk feature pack combining DeltaSupply + Spike + Negative predictions.
> **Schema version: v1.1.0**

## Overview

The risk feature pack merges predictions from three independent risk modules into a single CSV with one row per `(business_day, hour_business, target_month)`. This pack is consumed by downstream fusion and decision modules to assess hourly price risk across multiple dimensions.

### Source Modules

| Module | Source Predictions | Purpose |
|--------|-------------------|---------|
| DeltaSupply | `reports/local/risk_modules/delta_supply_risk_backtest_*/predictions_*.csv` | Deviation direction risk (up/down/large) |
| SpikeRisk | `reports/local/risk_modules/spike_risk_backtest_*/predictions_*.csv` | Price spike probability |
| NegativeRisk | `reports/local/risk_modules/negative_risk_backtest_*/predictions_*.csv` | Negative price probability |

### Output Location

```
reports/local/risk_modules/risk_feature_pack_YYYY_MM_DD/
    risk_feature_pack.csv
    manifest.json
    monthly_manifest.csv
```

---

## Column Definitions

### Key Columns

| Column | Type | Description |
|--------|------|-------------|
| `business_day` | datetime | Business day (date only, 00:00). Business day D corresponds to calendar day D-1 01:00 through D 00:00. |
| `hour_business` | int | Hour of the business day (1--24). Hour 1 = 01:00, hour 24 = 00:00 next calendar day. |
| `ds` | datetime | Full timestamp = `business_day + hour_business` hours. |
| `target_month` | string | Target month in YYYY-MM format. |

### DeltaSupply Deviation Risk

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `deviation_up_prob` | float | [0, 1] | Probability that realtime price deviates upward from day-ahead (rt - da > threshold). |
| `deviation_down_prob` | float | [0, 1] | Probability that realtime price deviates downward from day-ahead (rt - da < -threshold). |
| `deviation_large_abs_prob` | float | [0, 1] | Probability that |rt - da| exceeds a large absolute threshold. |
| `deviation_risk_score` | float | [0, 1] | Aggregate deviation risk score combining all three directions. |

### Spike Risk

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `spike_prob` | float | [0, 1] | Probability of a price spike event (rt exceeds upper threshold). |
| `extreme_spike_prob` | float | [0, 1] | Probability of an extreme price spike (rt exceeds extreme upper threshold). |
| `relative_spike_prob` | float | [0, 1] | Probability of a relative price spike (rt exceeds relative threshold). |
| `spike_risk_score` | float | [0, 1] | Aggregate spike risk score. |

### Negative Risk

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `negative_prob` | float | [0, 1] | Probability of a negative realtime price (rt < 0). |
| `deep_negative_prob` | float | [0, 1] | Probability of a deep negative price (rt < deep_threshold, e.g. -100 CNY/MWh). |
| `relative_down_prob` | float | [0, 1] | Probability of a relative downward price movement (rt exceeds relative down threshold). |
| `negative_risk_score` | float | [0, 1] | Aggregate negative price risk score. |

### Module Status Columns

| Column | Type | Description |
|--------|------|-------------|
| `module_status_delta_supply` | string | Per-month module status for DeltaSupply: GO, LOW_VALUE, NO-GO, INSUFFICIENT, or UNKNOWN. |
| `module_status_spike` | string | Per-month module status for SpikeRisk. |
| `module_status_negative` | string | Per-month module status for NegativeRisk. |

### Metadata Columns

| Column | Type | Description |
|--------|------|-------------|
| `threshold_version` | string | Version of the threshold calibration (e.g. `v1.0.0`). |
| `risk_feature_version` | string | Semantic version of the risk feature pack schema. Current: `v1.1.0`. |
| `metric_alignment_status` | string | Metric alignment audit result: `PASS`, `WARN`, or `FAIL`. |
| `metric_alignment_warning_reason` | string | Warning reason when status is `WARN`. Empty for `PASS`. |

### Eval-Only Columns (mode=eval)

| Column | Type | Description |
|--------|------|-------------|
| `y_true` | float | Actual realtime price (CNY/MWh). Present only in eval mode. |

---

## Online vs Eval Mode

| Aspect | Online Mode | Eval Mode |
|--------|-------------|-----------|
| CLI flag | `--mode online` | `--mode eval` |
| `y_true` column | Not included | Included (from source predictions) |
| Use case | Production inference, downstream fusion | Backtesting, metric computation |
| Safe for deployment | Yes | No (contains ground truth) |

**Online mode** is the default and produces a pack safe for production use. No actual/ground-truth columns are included.

**Eval mode** includes `y_true` for backtesting and metric computation. This pack must NOT be deployed to production.

---

## Version Field

The `risk_feature_version` column records the schema version of the risk feature pack. Current version: `v1.1.0`.

Version history:
- **v1.0.0**: Initial single-month pack (15 columns).
- **v1.1.0**: Multi-month pack with `target_month`, `relative_spike_prob`, `relative_down_prob`, `module_status_*`, `threshold_version`, `metric_alignment_warning_reason` (23 columns online).

Versioning follows semantic versioning:
- **Major**: breaking schema change (columns removed or semantics changed)
- **Minor**: new columns added (backward compatible)
- **Patch**: no schema change (re-export with updated data)

Downstream modules should check this version before consuming the pack and reject packs with an incompatible major version.

---

## Metric Alignment Requirement

The `--metric-alignment-status` flag gates pack production:

| Status | Behavior |
|--------|----------|
| `PASS` | Pack is produced normally. Exact alignment, no warnings. |
| `WARN` | Pack is produced. Computational alignment passed, but data completeness warning exists. Manifest records the warning reason. |
| `FAIL` | Script exits with error code 1. No pack is produced. |

**Rationale**: The metric alignment audit verifies that the risk module predictions are consistent with the mainline metric computation pipeline. If alignment fails, the risk probabilities may not be calibrated correctly, and the pack should not be used for production decisions.

CLI flags:
```
--metric-alignment-status PASS|WARN|FAIL
--metric-alignment-warning-reason "..."   # required when status=WARN
```

To resolve a FAIL status:
1. Run `scripts/audit_metric_alignment.py` to identify the discrepancy.
2. Fix the underlying issue (e.g., data preprocessing mismatch, column naming inconsistency).
3. Re-run the audit and confirm PASS or WARN.
4. Re-run the export with `--metric-alignment-status PASS` or `WARN`.

---

## Module Status Derivation

Module status columns are derived from per-month verdicts in the backtest results. The mapping is:

| Verdict Pattern | Module Status |
|----------------|---------------|
| `*_CHAMPION`, `*_STRONG`, `*_ACCEPTABLE`, `*_GO` | `GO` |
| `*_LOW_VALUE` | `LOW_VALUE` |
| `*_NO_GO` | `NO-GO` |
| `INSUFFICIENT_*` | `INSUFFICIENT` |

When monthly verdicts are available (from `champion_summary.json`), they are used directly. Otherwise, the overall verdict from `verdict.json` is applied uniformly to all months. The `status_sources` field in `manifest.json` records which source was used.

For `NO-GO` months, the corresponding risk columns are set to NaN.

---

## Usage Guidelines for Downstream Modules

### Reading the Pack

```python
import pandas as pd

pack = pd.read_csv("reports/local/risk_modules/risk_feature_pack_2026_01_05/risk_feature_pack.csv")
pack["business_day"] = pd.to_datetime(pack["business_day"])
```

### Checking Compatibility

```python
import json

with open("reports/local/risk_modules/risk_feature_pack_2026_01_05/manifest.json") as f:
    manifest = json.load(f)

assert manifest["risk_feature_version"].startswith("v1."), "Incompatible major version"
assert manifest["metric_alignment_status"] in ("PASS", "WARN"), "Alignment not verified"
```

### Recommended Consumption Patterns

1. **Fusion layer**: Use `deviation_risk_score`, `spike_risk_score`, and `negative_risk_score` as risk-aware weights for blending trend predictions with correction modules.

2. **Alert system**: Flag hours where any risk score exceeds a calibrated threshold (e.g., `spike_risk_score > 0.7` triggers a spike alert).

3. **Decision support**: Combine `deviation_up_prob` and `deviation_down_prob` to estimate the expected direction and magnitude of deviation from day-ahead prices. Use `relative_spike_prob` and `relative_down_prob` for relative risk assessment.

4. **Backtesting**: Use eval mode packs to compute precision/recall of risk alerts against actual price outcomes.

### Constraints

- Each `(business_day, hour_business, target_month)` triple appears exactly once. No duplicates.
- All probability and score columns are in [0, 1]. Missing values are NaN (not 0).
- The pack is sorted by `(target_month, business_day, hour_business)`.
- Do NOT modify the pack in place; treat it as an immutable artifact.

---

## CLI Reference

```bash
python scripts/export_risk_feature_pack_multimonth.py \
    --delta-supply-root reports/local/risk_modules/delta_supply_risk_backtest_2026_01_05 \
    --spike-root reports/local/risk_modules/spike_risk_backtest_2026_01_05 \
    --negative-root reports/local/risk_modules/negative_risk_backtest_2026_01_05 \
    --metric-alignment-status WARN \
    --metric-alignment-warning-reason "24 missing rows per month from business day alignment" \
    --out-dir reports/local/risk_modules/risk_feature_pack_2026_01_05 \
    --mode online
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--delta-supply-root` | Yes | Root directory of DeltaSupply backtest |
| `--spike-root` | Yes | Root directory of SpikeRisk backtest |
| `--negative-root` | Yes | Root directory of NegativeRisk backtest |
| `--metric-alignment-status` | Yes | `PASS`, `WARN`, or `FAIL`. FAIL prevents export. |
| `--metric-alignment-warning-reason` | No | Warning reason for WARN status. |
| `--out-dir` | Yes | Output directory for the pack |
| `--mode` | No (default: `online`) | `online` or `eval` |

---

## Manifest Schema

```json
{
  "timestamp": "2026-07-04T00:00:00",
  "risk_feature_version": "v1.1.0",
  "threshold_version": "v1.0.0",
  "mode": "online",
  "metric_alignment_status": "WARN",
  "metric_alignment_warning_reason": "24 missing rows per month from business day alignment",
  "n_rows": 3624,
  "n_months": 5,
  "columns": ["business_day", "hour_business", "ds", "target_month", ...],
  "column_types": {"business_day": "datetime64[ns]", ...},
  "key_columns": ["business_day", "hour_business", "target_month"],
  "unique_keys": 3624,
  "missing_values": {"deviation_up_prob": 0, ...},
  "date_range": {"start": "2025-12-31", "end": "2026-05-31"},
  "target_months": ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05"],
  "module_nogo_months": {"delta_supply": [], "spike": [], "negative": []},
  "status_sources": {"delta_supply": "monthly_verdicts", "spike": "monthly_verdicts", "negative": "monthly_verdicts"}
}
```
