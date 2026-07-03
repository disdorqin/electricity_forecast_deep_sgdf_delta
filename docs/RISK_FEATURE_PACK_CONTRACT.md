# Risk Feature Pack Contract

> Unified risk feature pack combining DeltaSupply + Spike + Negative predictions.

## Overview

The risk feature pack merges predictions from three independent risk modules into a single CSV with one row per `(business_day, hour_business)`. This pack is consumed by downstream fusion and decision modules to assess hourly price risk across multiple dimensions.

### Source Modules

| Module | Source Predictions | Purpose |
|--------|-------------------|---------|
| DeltaSupply | `artifacts/delta_supply/exp_YYYY_MM/predictions.csv` | Deviation direction risk (up/down/large) |
| SpikeRisk | `artifacts/spike_risk/exp_YYYY_MM/predictions.csv` | Price spike probability |
| NegativeRisk | `artifacts/negative_risk/exp_YYYY_MM/predictions.csv` | Negative price probability |

### Output Location

```
reports/local/risk_modules/risk_feature_pack_YYYY_MM/
    risk_feature_pack.csv
    manifest.json
```

---

## Column Definitions

### Key Columns

| Column | Type | Description |
|--------|------|-------------|
| `business_day` | datetime | Business day (date only, 00:00). Business day D corresponds to calendar day D-1 01:00 through D 00:00. |
| `hour_business` | int | Hour of the business day (1--24). Hour 1 = 01:00, hour 24 = 00:00 next calendar day. |
| `ds` | datetime | Full timestamp = `business_day + hour_business` hours. |

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
| `spike_risk_score` | float | [0, 1] | Aggregate spike risk score. |

### Negative Risk

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `negative_prob` | float | [0, 1] | Probability of a negative realtime price (rt < 0). |
| `deep_negative_prob` | float | [0, 1] | Probability of a deep negative price (rt < deep_threshold, e.g. -100 CNY/MWh). |
| `negative_risk_score` | float | [0, 1] | Aggregate negative price risk score. |

### Metadata Columns

| Column | Type | Description |
|--------|------|-------------|
| `risk_feature_version` | string | Semantic version of the risk feature pack schema (e.g. `v1.0.0`). |
| `metric_alignment_status` | string | Metric alignment audit result. Always `PASS` in a valid pack. |

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

The `risk_feature_version` column records the schema version of the risk feature pack. Current version: `v1.0.0`.

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
| `PASS` | Pack is produced normally. |
| `FAIL` | Script exits with error code 1. No pack is produced. |

**Rationale**: The metric alignment audit verifies that the risk module predictions are consistent with the mainline metric computation pipeline. If alignment fails, the risk probabilities may not be calibrated correctly, and the pack should not be used for production decisions.

To resolve a FAIL status:
1. Run `scripts/audit_metric_alignment.py` to identify the discrepancy.
2. Fix the underlying issue (e.g., data preprocessing mismatch, column naming inconsistency).
3. Re-run the audit and confirm PASS.
4. Re-run the export with `--metric-alignment-status PASS`.

---

## Usage Guidelines for Downstream Modules

### Reading the Pack

```python
import pandas as pd

pack = pd.read_csv("reports/local/risk_modules/risk_feature_pack_2026_02/risk_feature_pack.csv")
pack["business_day"] = pd.to_datetime(pack["business_day"])
```

### Checking Compatibility

```python
import json

with open("reports/local/risk_modules/risk_feature_pack_2026_02/manifest.json") as f:
    manifest = json.load(f)

assert manifest["risk_feature_version"].startswith("v1."), "Incompatible major version"
assert manifest["metric_alignment_status"] == "PASS", "Alignment not verified"
```

### Recommended Consumption Patterns

1. **Fusion layer**: Use `deviation_risk_score`, `spike_risk_score`, and `negative_risk_score` as risk-aware weights for blending trend predictions with correction modules.

2. **Alert system**: Flag hours where any risk score exceeds a calibrated threshold (e.g., `spike_risk_score > 0.7` triggers a spike alert).

3. **Decision support**: Combine `deviation_up_prob` and `deviation_down_prob` to estimate the expected direction and magnitude of deviation from day-ahead prices.

4. **Backtesting**: Use eval mode packs to compute precision/recall of risk alerts against actual price outcomes.

### Constraints

- Each `(business_day, hour_business)` pair appears exactly once. No duplicates.
- All probability and score columns are in [0, 1]. Missing values are NaN (not 0).
- The pack is sorted by `(business_day, hour_business)`.
- Do NOT modify the pack in place; treat it as an immutable artifact.

---

## CLI Reference

```bash
python scripts/export_risk_feature_pack.py \
    --delta-supply-predictions artifacts/delta_supply/exp_2026_02/predictions.csv \
    --spike-predictions artifacts/spike_risk/exp_2026_02/predictions.csv \
    --negative-predictions artifacts/negative_risk/exp_2026_02/predictions.csv \
    --metric-alignment-status PASS \
    --out-dir reports/local/risk_modules/risk_feature_pack_2026_02 \
    --mode online
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--delta-supply-predictions` | Yes | Path to DeltaSupply predictions CSV |
| `--spike-predictions` | Yes | Path to SpikeRisk predictions CSV |
| `--negative-predictions` | Yes | Path to NegativeRisk predictions CSV |
| `--metric-alignment-status` | Yes | `PASS` or `FAIL`. FAIL prevents export. |
| `--out-dir` | Yes | Output directory for the pack |
| `--mode` | No (default: `online`) | `online` or `eval` |

---

## Manifest Schema

```json
{
  "timestamp": "2026-03-01T12:00:00",
  "risk_feature_version": "v1.0.0",
  "mode": "online",
  "metric_alignment_status": "PASS",
  "n_rows": 672,
  "columns": ["business_day", "hour_business", "ds", ...],
  "column_types": {"business_day": "datetime64[ns]", ...},
  "key_columns": ["business_day", "hour_business"],
  "unique_keys": 672,
  "missing_values": {"deviation_up_prob": 0, ...},
  "date_range": {"start": "2026-02-01", "end": "2026-02-28"}
}
```
