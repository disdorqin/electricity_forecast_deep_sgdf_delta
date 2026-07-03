# Mainline Integration Guide

## Overview

DeepSGDFDelta / TrendKnight is the realtime normal trend prediction model for the Shandong electricity spot market forecasting system. This guide describes how to integrate its output into the mainline pipeline.

## Architecture Position

```
                    ┌──────────────────────┐
                    │   DA Price (日前电价)   │
                    └──────────┬───────────┘
                               │ da_anchor
                               ▼
┌──────────────────────────────────────────────────────┐
│              TrendKnight (本模型)                      │
│  SGDFNet Protocol B / D15 cutoff / walk-forward      │
│  delta_pred = f(forecast_features, history_lag24)     │
│  trend_pred = da_anchor + delta_pred                  │
└──────────────────────┬───────────────────────────────┘
                       │ trend_pred + flags
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌────────────┐ ┌──────────┐ ┌──────────────┐
   │ Spike      │ │ Negative │ │ Ledger       │
   │ Module     │ │ Module   │ │ Fusion       │
   │ (尖峰模块)  │ │ (负价模块)│ │ (融合模块)    │
   └─────┬──────┘ └────┬─────┘ └──────┬───────┘
         │              │              │
         └──────────────┼──────────────┘
                        ▼
                 ┌──────────────┐
                 │ Final Report │
                 │ (最终交付)     │
                 └──────────────┘
```

## Output Files

### Online Prediction Pack (`trend_prediction_pack.csv`)

This is the primary output for production use. It contains:

| Column | Type | Description |
|--------|------|-------------|
| business_day | datetime | Business day (date) |
| hour_business | int | Hour of business day (1-24) |
| period | str | Segment: "1_8", "9_16", "17_24" |
| ds | datetime | Full timestamp |
| trend_pred | float | Main trend prediction (CNY/MWh) |
| trend_model_name | str | Model identifier |
| trend_confidence | float | Confidence score (0-1) |
| deep_rt_pred | float | Deep model RT prediction |
| sgdfnet_pred | float | SGDFNet baseline prediction |
| blend_pred | float | Blended prediction |
| da_anchor | float | Day-ahead anchor price |
| normal_trend_flag | int | 1=normal, 0=outlier |
| high_price_bucket_flag | int | 1 if |pred| > 500 |
| negative_bucket_flag | int | 1 if pred < 0 |

**Important**: This file does NOT contain `y_true` or eval-only residuals. It is safe for online/production use.

### Eval Pack (`trend_prediction_pack_eval.csv`)

Additional columns for evaluation only:

| Column | Type | Description |
|--------|------|-------------|
| y_true | float | Actual realtime price |
| residual_for_spike_module | float | y_true - trend_pred |
| residual_for_negative_module | float | y_true - trend_pred |

**Warning**: This file contains ground truth. Do NOT use in production.

### Manifest (`trend_prediction_manifest.json`)

Machine-readable metadata including date range, column info, model performance metrics, and blend weights.

## Integration Points

### 1. Spike Module (尖峰模块)

Input: `trend_pred`, `normal_trend_flag`, `high_price_bucket_flag`

Behavior:
- When `normal_trend_flag=1`: spike module may optionally refine
- When `normal_trend_flag=0`: spike module should override `trend_pred`
- When `high_price_bucket_flag=1`: spike module activates special handling

### 2. Negative Price Module (负价模块)

Input: `trend_pred`, `negative_bucket_flag`

Behavior:
- When `negative_bucket_flag=1`: negative module activates
- Uses `da_anchor` and forecast features to predict negative price magnitude

### 3. Ledger Fusion (融合模块)

Input: `trend_pred` (after spike/negative corrections)

Behavior:
- Uses `trend_pred` as the base trend
- Applies spike/negative corrections
- Produces final `rt_final` prediction

## How to Export

```bash
# From champion search results
python scripts/export_trend_prediction_pack.py \
    --predictions reports/local/phase3/month_2026_03/champion_predictions.csv \
    --champion-model v2_day_tcn \
    --metrics-json reports/local/phase3/month_2026_03/champion_metrics_summary.json \
    --blend-weights-json reports/local/phase3/month_2026_03/blend_weights.json \
    --out-dir reports/local/phase3/export \
    --include-eval
```

## Cutoff Safety

This model strictly follows the SGDFNet Protocol B cutoff-safe design:

1. **D-1 15:00 cutoff**: No data after 15:00 on the decision day is used
2. **Lag-24 shift**: All actual-side history features are shifted by 24 hours
3. **No actual RT**: Realtime actual prices post-cutoff are never used
4. **No spike/negative labels**: Training does not use ground truth labels for spike/negative classification
5. **Blend weight window**: Blend weights are learned from D-30 to D-1 only

## Business Day Alignment

- Calendar day D, hour 00:00 → business_day D-1, hour_business 24
- Calendar day D, hour 01:00 → business_day D, hour_business 1
- Calendar day D, hour 23:00 → business_day D, hour_business 23

## Go/No-Go Thresholds

| Verdict | Criteria |
|---------|----------|
| PASS | Monthly avg sMAPE_floor50 < 15.0 |
| SOFT_PASS | Overall sMAPE_floor50 <= 15.8 |
| BASELINE_PASS | Overall sMAPE_floor50 <= 16.5902 (SGDFNet baseline) |
| NO-GO | Worse than baseline or leakage detected |

## Contact

For integration issues, refer to:
- `models/deep_sgdf_delta/integration_contract.py` — Schema validation
- `scripts/export_trend_prediction_pack.py` — Export tool
- `docs/PHASE3_REAL_DATA_RESULTS.md` — Experimental results
