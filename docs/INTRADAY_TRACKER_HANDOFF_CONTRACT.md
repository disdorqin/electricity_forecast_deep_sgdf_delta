# Intraday Tracker Handoff Contract

**Version:** 1.0
**Phase:** 9

## Overview

The Intraday Adaptive Residual Tracker provides same-day residual corrections for the 9_16 segment. It is strictly an INTRADAY-only module and must NOT be used for full-day day-ahead prediction.

## Interface

### Input

The tracker requires:
- `business_day`: the target business day
- `cutoff_hour`: the last observed hour (hours 9 to cutoff_hour are used)
- `sgdfnet_pred`: SGDFNet base model predictions for all 9_16 hours
- `rt_actual`: observed real-time prices for hours <= cutoff_hour
- `da_anchor`: day-ahead price anchor

### Output Fields

| Field | Type | Description |
|-------|------|-------------|
| business_day | timestamp | Target business day |
| cutoff_hour | int | Last observed hour |
| target_hour | int | Hour being corrected (cutoff_hour+1 to 16) |
| ds | timestamp | Full timestamp |
| base_model_name | str | Name of base model (e.g., "sgdfnet") |
| base_pred | float | Base model prediction |
| intraday_residual_state | JSON | State summary (n_observed, bias_direction) |
| intraday_correction | float | Correction amount (clipped to ±80) |
| intraday_corrected_pred | float | base_pred + correction |
| intraday_confidence | float | Confidence score (0-1) |
| intraday_trigger_flag | bool | Always True for INTRADAY |
| guardrail_reason | str | Reason if guardrail applied |
| mode | str | Always "INTRADAY" |

## Constraints

1. **Mode must be INTRADAY.** The tracker never outputs full-day corrections.
2. **Minimum observed hours:** At least 2 observed hours required. If insufficient, correction = 0.
3. **Max correction:** ±80 (configurable).
4. **Negative price guardrail:** When da_anchor < 0, correction weight reduced to 30%.
5. **Only future hours:** Hours <= cutoff_hour get zero correction.

## Activation Conditions

The tracker activates when:
- Business day has at least 2 observed 9_16 hours with actual prices
- Target hours are within 9_16 segment and > cutoff_hour

## Deactivation

The tracker does NOT activate when:
- No observed actuals are available (FULL_DAY mode)
- Fewer than 2 observed hours
- Confidence is below threshold (0.2)
