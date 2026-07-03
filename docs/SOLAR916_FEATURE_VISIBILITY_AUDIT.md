# Solar916 Feature Visibility Audit

**Date:** 2026-07-03 13:49
**Phase:** 8 (No-Leak Revalidation)

## Feature Table

| Feature | Source | Uses Actual | Uses Current Target | Visible at Pred Time | Leakage Risk | Notes |
|---------|--------|-------------|--------------------|--------------------|-------------|-------|
| hour_business | business_time.py | no | no | yes | low | Temporal feature, known at prediction time |
| weekday | business_time.py | no | no | yes | low | Temporal feature |
| month | business_time.py | no | no | yes | low | Temporal feature |
| da_anchor | 日前电价 (DA price) | no | no | yes | low | Day-ahead price, known before RT |
| sgdfnet_pred | SGDFNet model output | no | no | yes | low | Base model prediction, available at correction time |
| forecast_load | 直调负荷预测值 | no | no | yes | low | Forecast, known at prediction time |
| forecast_wind | 风电总加预测值 | no | no | yes | low | Forecast |
| forecast_solar | 光伏总加预测值 | no | no | yes | low | Forecast |
| forecast_new_energy | 新能源总加预测值 | no | no | yes | low | Forecast |
| bidding_space | 竞价空间预测值 | no | no | yes | low | Forecast |
| net_load | Derived: forecast_load - forecast_new_energy | no | no | yes | low | Derived from forecasts only |
| renewable_share | Derived: (solar + wind) / net_load | no | no | yes | low | Derived from forecasts only |
| delta_lag_24 | Previous business_day same hour delta (merge-based) | YES | no | yes | low | Phase 8: merge-based, uses PAST RT actual from previous day same hour. Allowed. |
| delta_lag_168 | business_day - 7 same hour delta (merge-based) | YES | no | yes | low | Phase 8: merge-based, uses PAST RT actual from 7 days ago same hour. Allowed. |
| residual_lag_24 | Previous business_day same hour SGDFNet residual | YES | no | yes | low | Phase 8: merge-based, uses PAST residual. Allowed. |
| residual_lag_168 | business_day - 7 same hour SGDFNet residual | YES | no | yes | low | Phase 8: merge-based, uses PAST residual. Allowed. |
| rolling_residual_mean_7d | shift(1).rolling(7).mean() on sgdfnet_residual | YES | no | yes | low | Phase 8: shift(1) excludes current row. Uses only PAST residuals. Allowed. |
| rolling_residual_std_7d | shift(1).rolling(7).std() on sgdfnet_residual | YES | no | yes | low | Phase 8: shift(1) excludes current row. Allowed. |
| same_hour_residual_mean_7d | groupby(hour).shift(1).rolling(7).mean() | YES | no | yes | low | Phase 8: groupby + shift(1) excludes current row. Allowed. |
| same_hour_residual_std_7d | groupby(hour).shift(1).rolling(7).std() | YES | no | yes | low | Phase 8: groupby + shift(1) excludes current row. Allowed. |
| sgdfnet_residual | TARGET: rt_actual - sgdfnet_pred | YES | **YES** | **NO** | **HIGH** | TARGET VARIABLE — must NOT be used as feature. Used only as training target. |

## Audit Results

- High-risk features (excluding target): 0
- Features using current target: 1 (should only be sgdfnet_residual)
- Target used as feature: 0

## Verdict: **PASSED**

All features are visible at prediction time. No leakage detected.
sgdfnet_residual is correctly used only as target, not as feature.