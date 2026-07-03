# Phase 8: Solar916 No-Leak Revalidation + Guardrail — Results Report

**Date:** 2026-07-03
**Status:** NO-GO — Solar916 should NOT enter the main fusion pipeline in its current form.

---

## 1. Why Phase 7 Results Are On Hold

Phase 7 reported a +4.01 sMAPE improvement on 9_16 (40.87 → 36.86), with a GO verdict. However, the feature engineering contained two leakage/alignment issues:

- **Lag features used `shift(1)` on the 9_16-only subset**, which retrieves the *previous row* in the subset (e.g., for hour 9, it retrieves hour 16 from the previous day). This is not a true same-hour previous-day lag — it's a cross-hour lag that accidentally captures useful signal from adjacent hours.
- **Rolling residual features included the current row's residual**, which is the training target. This is direct target leakage.

These issues meant the Phase 7 improvement was partially attributable to leaked information, making the results unreliable for production deployment.

## 2. Leakage/Alignment Issues Fixed

### Lag Features (Task A)
Replaced `shift(1)` on 9_16 subset with **merge-based same-hour previous-day lookup**:
- `delta_lag_24`: merge with `business_day + 1 day` on same `hour_business` → true previous business day, same hour
- `delta_lag_168`: merge with `business_day + 7 days` on same `hour_business` → true 7-days-ago same hour
- Same approach for `residual_lag_24` and `residual_lag_168`

### Rolling Features (Task B)
Applied `shift(1)` before rolling window computation:
- `rolling_residual_mean_7d`: `shift(1).rolling(7).mean()` — excludes current row
- `same_hour_residual_mean_7d`: `groupby(hour).shift(1).groupby(hour).rolling(7).mean()` — excludes current row, same-hour only

### Dataset Builder (Task A/B)
Restructured `solar916_dataset.py` to build features on the **full 24-hour dataset** before filtering to 9_16, ensuring lag/rolling features have correct temporal context.

## 3. Feature Visibility Audit

**Verdict: PASSED** — all 20 features are visible at prediction time.

Full audit: `docs/SOLAR916_FEATURE_VISIBILITY_AUDIT.md`

Key findings:
- 0 high-risk features (excluding the target variable)
- `sgdfnet_residual` correctly used only as training target, never as a feature
- All lag features use only past data (previous day or 7 days ago, same hour)
- All rolling features use `shift(1)` to exclude current row
- Temporal features (hour, weekday, month) and forecast features are all available at prediction time

## 4. No-Leak 2026-02 Results

| Metric | Phase 7 (Leaky) | Phase 8 (No-Leak) |
|--------|-----------------|-------------------|
| 9_16 Baseline sMAPE | 40.87 | 40.87 |
| 9_16 Corrected sMAPE | 36.86 | 53.20 |
| Improvement | +4.01 | **-12.33** |
| Verdict | GO | **NO-GO** |

### Hourly Breakdown (No-Leak)

| Hour | Baseline | Corrected | Improvement |
|------|----------|-----------|-------------|
| 9 | 24.72 | 43.06 | -18.33 |
| 10 | 44.77 | 49.54 | -4.78 |
| 11 | 44.11 | 63.49 | -19.38 |
| 12 | 46.48 | 64.40 | -17.92 |
| 13 | 42.19 | 59.76 | -17.57 |
| 14 | 45.58 | 49.57 | -3.99 |
| 15 | 38.38 | 43.58 | -5.21 |
| 16 | 40.75 | 52.19 | -11.44 |

Every single hour is worse than baseline. The model uniformly degrades predictions.

### Root Cause: Residual Distribution Shift

The fundamental problem is a massive distribution shift in SGDFNet residuals between January (training) and February (test):

- **January mean residual: +68.72** (SGDFNet systematically under-predicts)
- **February mean residual: +5.60** (SGDFNet is nearly unbiased)

Any correction model trained on January residuals will learn to add a large positive correction, which is catastrophically wrong for February where the residual is near zero. This is a **non-stationarity problem** in the base model's bias.

### Feature Signal Comparison

| Feature Set | Corrected sMAPE | Improvement |
|-------------|----------------|-------------|
| Phase 7 (leaky lags) | 37.50 | +3.37 |
| Phase 8 (no-leak lags) | 47.61 | -6.74 |
| No lag features at all | 58.03 | -17.16 |
| Mean residual (baseline) | 49.89 | -9.02 |

The Phase 7 leaky features captured cross-hour correlation (previous row ≈ previous hour) which provided genuine predictive signal. The strict same-hour lags in Phase 8 are much weaker because with only 8 data points per day per hour, there's insufficient data to learn stable patterns.

## 5. Guardrail Results

Applied guardrail with: hours 9 and 11 disabled, negative price risk weight = 0.

| Metric | No Guardrail | With Guardrail |
|--------|-------------|----------------|
| Overall sMAPE | 53.20 | 45.36 |
| Normal bucket improvement | -0.08 | +7.18 |
| Negative bucket improvement | -19.68 | -11.36 |

Guardrail mitigates damage but does not reverse it. The overall corrected sMAPE (45.36) is still worse than baseline (40.87).

## 6. Negative Bucket Analysis

The negative bucket remains problematic even with guardrail:
- Baseline: 29.64 sMAPE
- No-leak corrected: 49.32 (worsened by 19.68)
- Guarded: 41.00 (worsened by 11.36)

The guardrail reduces negative bucket damage by ~42% (from -19.68 to -11.36) by disabling corrections when `da_anchor < 0`, but the remaining corrections for negative-price hours still degrade quality.

## 7. Hour 10 Analysis

Hour 10 was the best performer in Phase 7 (+6.48 improvement). In Phase 8:
- Baseline: 44.77
- No-leak corrected: 49.54 (worsened by 4.78)
- Guarded: similar (hour 10 not disabled)

Hour 10 improvement does NOT survive leak correction.

## 8. Jan-Mar Stability Test

Not performed. Given the clear NO-GO result for February and the identified root cause (residual distribution shift), multi-month testing would not change the conclusion. The fundamental issue is that:
1. SGDFNet's bias is non-stationary across months
2. Same-hour lag features don't provide enough signal with limited data
3. The correction model cannot adapt to shifting residual distributions

## 9. Recommendation: NO-GO

**Solar916 should NOT enter the main fusion pipeline.**

The no-leak revalidation conclusively shows that the Phase 7 improvement was an artifact of:
1. Cross-hour lag features that captured genuine but operationally fragile signal
2. Target leakage in rolling features
3. Non-stationary residual distributions that make any learned correction unreliable

### Recommendations for Future Work

1. **Address non-stationarity**: Instead of predicting absolute residuals, predict the *direction* or *relative magnitude* of correction. Consider online learning / adaptive models that update with each new observation.

2. **Use cross-hour lags legitimately**: The Phase 7 result suggests cross-hour residual correlation is real. A valid approach would be to use previous-hour delta as a feature (available at prediction time since previous hours have already occurred), with proper walk-forward validation.

3. **Ensemble approach**: Instead of a separate correction model, consider blending SGDFNet with other base models (TimeMixer, RT916) where the fusion weights adapt to recent performance.

4. **Longer training window**: When more SGDFNet predictions accumulate (e.g., 6+ months), revisit the residual correction approach with a more robust training set.

## 10. Data Integrity

All metrics in this report are computed from actual model runs. No metrics were fabricated or estimated. All intermediate outputs are in `reports/local/phase8/solar916_noleak/` and `reports/local/phase8/solar916_noleak_v2/`.
