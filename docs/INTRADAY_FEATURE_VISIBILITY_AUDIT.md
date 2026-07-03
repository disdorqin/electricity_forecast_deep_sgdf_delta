# Intraday Feature Visibility Audit

**Date:** 2026-07-03 14:13
**Phase:** 9 (Intraday Adaptive Residual Tracker)

## Key Principle

The IntradayResidualTracker uses **same-day observed residuals** — residuals from hours that have already occurred within the current business day. This is fundamentally different from the offline Solar916 model which used historical residuals from different months.

**In FULL_DAY mode:** previous-hour residual is LEAKAGE. The model must not use any D-day actuals.
**In INTRADAY mode:** previous-hour residual is LEGAL for hours <= cutoff_hour, because those hours have already occurred and their actuals are observable.

## Feature Table

| Feature | Source | Uses Actual | Uses Current Target | Visible FULL_DAY | Visible INTRADAY | Leakage Risk | Notes |
|---------|--------|-------------|--------------------|--------------------|--------------------|-------------|-------|
| sgdfnet_pred | SGDFNet model output | no | no | yes | yes | low | Base model prediction, available before RT actuals |
| da_anchor | Day-ahead price | no | no | yes | yes | low | DA price, known before RT |
| rt_actual (observed hours) | Real-time actual price for hours <= cutoff | yes | no | NO | yes (only hours <= cutoff) | low | INTRADAY only: observed actuals are past events, legal to use |
| residual (observed hours) | rt_actual - sgdfnet_pred for hours <= cutoff | yes | no | NO — LEAKAGE | yes (only hours <= cutoff) | low (in INTRADAY), HIGH (in FULL_DAY) | CRITICAL: previous-hour residual is LEAKAGE in FULL_DAY mode. Legal in INTRADAY because it's a past observation. |
| mean_residual_today | Mean of observed residuals (hours <= cutoff) | yes | no | NO — LEAKAGE | yes | low (in INTRADAY), HIGH (in FULL_DAY) | Computed from past observations only. Legal in INTRADAY. |
| ewm_residual_today | Exponential weighted mean of observed residuals | yes | no | NO — LEAKAGE | yes | low (in INTRADAY), HIGH (in FULL_DAY) | Same as mean_residual_today but with exponential weighting |
| last_residual | Most recent observed residual | yes | no | NO — LEAKAGE | yes | low (in INTRADAY), HIGH (in FULL_DAY) | Most recent past observation. Legal in INTRADAY. |
| residual_std_today | Std of observed residuals | yes | no | NO — LEAKAGE | yes | low (in INTRADAY), HIGH (in FULL_DAY) | Computed from past observations only |
| rt_actual (future hours) | Real-time actual for hours > cutoff | yes | YES | NO | NO | HIGH | NEVER used — this is the target variable for future hours |

## Audit Results

- Features with HIGH leakage risk (not used in INTRADAY): 1
- Features that are leakage in FULL_DAY but legal in INTRADAY: 5
- Tracker only activates in INTRADAY mode: YES
- Tracker requires min_observed_hours >= 2: YES
- Tracker never uses future actuals: YES

## Critical Rules

1. The tracker MUST NOT be used for FULL_DAY / day-ahead prediction.
2. The tracker only activates when observed actuals are available (cutoff_hour passed).
3. If no observed actuals exist, the tracker returns zero correction.
4. All residual-based features are computed from hours <= cutoff_hour only.
5. The correction is clipped to max_abs_correction to prevent extreme corrections.

## Verdict: **PASSED**

The IntradayResidualTracker correctly restricts feature usage to legally observable data in INTRADAY mode. No leakage detected for the intended use case.