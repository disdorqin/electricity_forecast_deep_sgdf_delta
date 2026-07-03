# DeltaSupply-1 Results Report

## 1. Module Scope

DeltaSupply is an independent supply-demand deviation risk module. It does NOT predict realtime price trends. It outputs structural risk signals (deviation direction, magnitude, confidence) for downstream spike/negative/ledger modules.

## 2. DeepFinal Archive Background

Phase DeepFinal-1 through DeepFinal-5 concluded:
- TrendKnightRT archived as main realtime model
- All residual-only deep models failed to beat DA anchor
- HGB/Ridge/MLP bias corrections all NO-GO
- Decision: stop pursuing TrendKnightRT, stop blind hyperparameter tuning

## 3. Why Pivot to Deviation Risk Module

Even though we cannot predict exact realtime prices, the supply-demand deviation signal may still carry structural information. If we can identify when and how much realtime prices deviate from DA anchor (and in which direction), this can serve as a risk feature for spike detection, negative price warning, and dynamic fusion weighting.

## 4. Target Definitions

```
price_delta = rt_actual - da_anchor

upward_deviation_label: price_delta >= 100
downward_deviation_label: price_delta <= -100
large_abs_deviation_label: |price_delta| >= 150
deviation_magnitude_target: clip(price_delta, -500, +500)
```

Overall statistics (full dataset):
- Upward rate: 9.4%
- Downward rate: 11.4%
- Large abs rate: 12.9%
- Mean delta: varies by period

## 5. Feature Definitions and Coverage

Feature audit verdict: **FORMAL_READY**

| Group | Coverage | Details |
|-------|----------|---------|
| Forecast | 71% | load_forecast, renewable_forecast, wind_forecast, solar_forecast, bidding_space_forecast available; tie_line_forecast and provincial_load_forecast missing |
| Lag | 77% | D-1/D-2 actual lags, previous day means, price_delta lags |
| Calendar | 100% | hour_sin/cos, dow_sin/cos, month_sin/cos, is_weekend, period_id |
| SGDFNet | Not available | No SGDFNet predictions in current data |

Total features: 24

Derived features: forecast_net_load, forecast_renewable_share, forecast_wind_share, forecast_solar_share, forecast_supply_demand_gap, forecast_bidding_pressure, forecast_thermal_pressure.

Leakage check: PASSED (FULL_DAY mode, no target-day actual used).

## 6. 2026-02 Cannon Run Results

Training: 35,087 samples, 24 features.
Test: 672 samples (28 days x 24 hours).

## 7. Classification Metrics

| Target | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|--------|-----------|--------|----|---------|--------|
| Upward deviation | 0.286 | 0.024 | 0.043 | 0.723 | 0.255 |
| Downward deviation | 0.409 | 0.173 | 0.243 | 0.821 | 0.371 |
| Large abs deviation | 0.553 | 0.181 | 0.273 | 0.795 | 0.471 |

Observations:
- AUC scores (0.72-0.82) indicate the model has some discriminative ability
- F1 scores are very low due to low positive rates (12-17%) and low recall
- The model can partially distinguish deviation directions but lacks precision

## 8. Correction Simulation

| Weight | DA sMAPE | Corrected sMAPE | Improvement |
|--------|----------|-----------------|-------------|
| 0.0 | 47.35% | 47.35% | 0.00pp |
| 0.1 | 47.35% | 47.56% | -0.21pp |
| 0.2 | 47.35% | 47.94% | -0.60pp |
| 0.3 | 47.35% | 48.41% | -1.06pp |
| 0.5 | 47.35% | 49.45% | -2.11pp |
| 1.0 | 47.35% | 52.57% | -5.23pp |

All correction weights make things worse. The magnitude predictions are too noisy (MAE=83.87, correlation=0.067) to provide useful corrections.

## 9. Feature Importance

Top features (from permutation importance):
- Calendar features (hour_sin, hour_cos, period_id) dominate
- Forecast-derived features (forecast_net_load, forecast_renewable_share) contribute
- Lag features provide modest additional signal

## 10. Does Supply-Demand Deviation Signal Exist?

**Partially.** The classification AUC (0.72-0.82) suggests there IS some structural signal in the supply-demand features that correlates with deviation direction. However:
- The signal is too weak for reliable classification (F1 < 0.3)
- The magnitude regression is essentially useless (correlation 0.067)
- The signal cannot improve DA anchor predictions

## 11. Multi-Month Backtest

**Not executed.** Per spec, multi-month backtest only runs when 2026-02 result is GO or LOW_VALUE. Since the result is NO-GO, backtest is skipped.

## 12. Multi-Month Results

N/A (not executed).

## 13. Recommendation for Spike / Negative / Ledger Risk Features

**Not recommended.** The deviation risk signals are too weak to serve as reliable features:
- Classification recall too low for early warning
- Magnitude predictions too noisy for correction
- Correction simulation shows consistent degradation

## 14. Final Verdict

**NO-GO**

The DeltaSupply module is well-engineered (FORMAL_READY features, proper leakage prevention, correct business time alignment), but the underlying signal is insufficient:
- Supply-demand forecast features have limited predictive power for price deviations
- The deviation magnitude is essentially unpredictable from available features
- Applying model predictions as corrections consistently worsens DA anchor performance

This does NOT mean the deviation concept is worthless. It means that with the current data (no SGDFNet predictions, limited forecast columns, no real-time intraday updates), the deviation risk module cannot produce actionable signals.

## 15. No Fabricated Metrics

All metrics reported above are from actual model training and evaluation runs. No metrics were fabricated or estimated.
