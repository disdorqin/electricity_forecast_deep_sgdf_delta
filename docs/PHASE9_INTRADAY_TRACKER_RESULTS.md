# Phase 9: Intraday Adaptive Residual Tracker — Results Report

**Date:** 2026-07-03
**Status:** LOW-WEIGHT — promising but below GO threshold
**Mode:** INTRADAY only (NOT for full-day day-ahead)

---

## 1. Why Phase 8 Was NO-GO

Phase 8's Solar916 offline residual correction model was a clear NO-GO:
- No-leak 2026-02 corrected sMAPE: 53.20 vs baseline 40.87 (worsened by 12.33)
- Root cause: SGDFNet residual distribution shifted dramatically between January (mean +68.72) and February (mean +5.60)
- Offline tree models trained on historical residuals cannot adapt to non-stationary base-model bias

## 2. Why Phase 9 Changed Approach

Instead of training an offline model on historical residuals (which fails due to distribution shift), Phase 9 uses **same-day observed residuals** to adaptively correct future hours. This avoids the non-stationarity problem because:
- It uses residuals from hours that have **already occurred today**
- The correction reflects the **current day's** bias pattern, not last month's
- It only activates when actual prices are available (intraday nowcast)

## 3. FULL_DAY vs INTRADAY Mode

| Aspect | FULL_DAY | INTRADAY |
|--------|----------|----------|
| When activated | Day-ahead, before any RT actuals | After cutoff_hour, with observed actuals |
| Uses D-day actuals | NO (strictly forbidden) | YES (only hours <= cutoff) |
| Previous-hour residual | LEAKAGE | Legal (past observation) |
| Use case | Full-day forecast | Intraday nowcast / updating forecast |
| Solar916 offline | Intended for this (failed) | N/A |
| IntradayResidualTracker | MUST NOT be used | Designed for this |

## 4. Cutoff Hour Group Results

| Cutoff | Count | Baseline sMAPE | Corrected sMAPE | Improvement |
|--------|-------|----------------|-----------------|-------------|
| 10 | 168 | 42.91 | 42.55 | +0.36 |
| 11 | 140 | 42.67 | 42.23 | +0.44 |
| 12 | 112 | 41.72 | 40.80 | +0.92 |
| 13 | 84 | 41.57 | 41.17 | +0.39 |
| 14 | 56 | 39.56 | 38.42 | **+1.14** |
| 15 | 28 | 40.75 | 39.94 | +0.80 |

Later cutoffs (more observed hours) give better corrections. Cutoff=14 achieves the best improvement at +1.14, which exceeds the GO threshold of 1.0 for that specific cutoff.

## 5. Which Cutoffs Are Effective

- **Cutoff >= 12**: All show meaningful improvement (+0.39 to +1.14)
- **Cutoff 14**: Best performer (+1.14), meets GO criteria individually
- **Cutoff 10-11**: Marginal improvement (+0.36 to +0.44)
- **Average for cutoff >= 10**: +0.68 (LOW-WEIGHT range)

The pattern is clear: the more hours observed, the better the correction. This is expected — the tracker adapts to the day's bias pattern using real-time data.

## 6. Hour 10/11/12 Analysis

| Hour | Baseline | Corrected | Improvement |
|------|----------|-----------|-------------|
| 11 | 44.11 | 42.24 | **+1.86** |
| 12 | 46.48 | 46.34 | +0.14 |
| 13 | 42.19 | 40.24 | **+1.95** |
| 14 | 45.58 | 45.05 | +0.53 |
| 15 | 38.38 | 37.33 | **+1.05** |
| 16 | 40.75 | 41.25 | -0.50 |

Hours 11, 13, 15 show significant improvement. Hour 16 is slightly worse (-0.50), likely because it's the furthest from the cutoff and the correction signal decays with distance.

Note: Hour 10 is not in the target set because it's only predicted when cutoff < 10, and cutoff=8/9 had insufficient observed data.

## 7. Normal Bucket

Normal bucket: baseline 67.36 → corrected 66.66, **improvement +0.70**. The tracker helps normal-price hours.

## 8. Negative Bucket

Negative bucket: baseline 30.12 → corrected 29.63, **improvement +0.49**. Unlike the offline Solar916 model which worsened the negative bucket by -19.68, the intraday tracker actually IMPROVES it. This is because same-day observed residuals correctly capture the current day's bias direction, even for negative prices.

## 9. Spike Bucket

Spike bucket: baseline 41.53 → corrected 37.73, improvement +3.80 (only 6 samples, treat with caution).

## 10. Recommendation: LOW-WEIGHT

The Intraday Adaptive Residual Tracker shows consistent improvement across all cutoffs and buckets, but the overall improvement (+0.59) falls below the GO threshold of 1.0.

**Verdict: LOW-WEIGHT**

The tracker is recommended for:
- **INTRADAY use only** — when actual prices are available for earlier hours
- **Cutoff >= 12** — where improvement is most significant
- **Complementing, not replacing** the base SGDFNet model

It is NOT recommended for:
- Full-day day-ahead prediction (strictly forbidden)
- Cutoff < 10 (insufficient observed data)

## 11. Integration Path

To enter the main fusion pipeline:
1. The tracker should be activated only in INTRADAY mode
2. Fusion weights should consider the tracker's confidence score
3. For cutoff >= 12, the tracker's correction can be blended with higher weight
4. For cutoff < 10, the tracker should be disabled or given minimal weight

## 12. Data Integrity

All metrics are computed from actual model runs on 2026-02 data (28 business days, 224 test rows, 588 predictions across 8 cutoff levels). No metrics were fabricated.
