# DeltaSupply-1: Production Deviation / Supply-Demand Deviation Risk Module

## Background

Phase DeepFinal-1 through DeepFinal-5 concluded with the following verdict:

```
ENGINEERING_COMPLETE
MODEL_NO_GO
ARCHIVE_AS_MAIN_REALTIME_MODEL
KEEP_UTILITIES_FOR_MAIN_SYSTEM
```

TrendKnightRT as a main realtime trend model is archived. Residual-only deep models, HGB/Ridge/MLP residual corrections all failed to beat DA anchor. The decision was made to stop pursuing TrendKnightRT as the primary realtime model.

However, the supply-demand deviation signal itself remains valuable. Even if we cannot predict the exact realtime price, we may still identify **when and how much** the realtime price is likely to deviate from the day-ahead anchor, and in which direction. This is the purpose of the DeltaSupply module.

---

## This Module Is Responsible For

- forecast vs actual supply-demand deviation feature construction
- realtime price deviation risk scoring
- upward deviation classification (rt_actual - da_anchor >= threshold)
- downward deviation classification (rt_actual - da_anchor <= -threshold)
- large absolute deviation classification (|rt_actual - da_anchor| >= threshold)
- deviation magnitude regression (clipped price_delta)
- feature importance / explainability report
- handoff fields for spike / negative / ledger modules

Output fields:

```
deviation_risk_score
deviation_direction
deviation_magnitude_pred
upward_deviation_prob
downward_deviation_prob
large_abs_deviation_prob
confidence
```

---

## This Module Is NOT Responsible For

- main realtime trend prediction
- day-ahead price prediction
- final spike correction
- final negative price correction
- ledger fusion
- production deployment
- TrendKnightRT revival

---

## Reusable Assets

This module reuses the following from the archived codebase:

```
models/deep_sgdf_delta/business_time.py          -- business day alignment
models/deep_sgdf_delta/realtime_column_mapping.py -- Chinese column mapping
models/deep_sgdf_delta/realtime_feature_contract.py -- feature contract reference
```

---

## Constraints

1. FULL_DAY mode by default: no target-day actual features.
2. Forecast-side features may use target-day forecasts.
3. Test actual only for evaluation, never for generating predictions.
4. No fabricated metrics.
5. If data coverage is insufficient, document the blocker clearly.
