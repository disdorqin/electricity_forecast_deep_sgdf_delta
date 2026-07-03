# Intraday Tracker Handoff Contract

**Version:** 2.0
**Phase:** 10

## Overview

The Intraday Adaptive Residual Tracker provides same-day residual corrections for the 9_16 segment. It is strictly an INTRADAY-only module and must NOT be used for full-day day-ahead prediction.

Phase 10 introduces:
- Clear separation of correction pipeline stages (base → model_weight → pre_guardrail → guardrail_weight → final)
- Policy gating (DISABLED / SHADOW_ONLY / LOW_WEIGHT / HIGH_WEIGHT)
- Fusion weight assignment based on policy decision

## Interface

### Input

The tracker requires:
- `business_day`: the target business day
- `cutoff_hour`: the last observed hour (hours 9 to cutoff_hour are used)
- `sgdfnet_pred`: SGDFNet base model predictions for all 9_16 hours
- `rt_actual`: observed real-time prices for hours <= cutoff_hour
- `da_anchor`: day-ahead price anchor

### Output Fields (Phase 10)

| Field | Type | Description |
|-------|------|-------------|
| business_day | timestamp | Target business day |
| cutoff_hour | int | Last observed hour |
| target_hour | int | Hour being corrected (cutoff_hour+1 to 16) |
| ds | timestamp | Full timestamp |
| mode | str | Always "INTRADAY" |
| base_model_name | str | Name of base model (e.g., "sgdfnet") |
| base_pred | float | Base model prediction |
| intraday_base_correction | float | Unweighted base correction (constant for all hours in a day) |
| intraday_model_weight | float | confidence × distance_decay × std_penalty (varies by target_hour) |
| intraday_pre_guardrail_correction | float | base_correction × model_weight (clipped to ±max_abs_correction) |
| intraday_guardrail_weight | float | Product of all guardrail weights (negative/cutoff/confidence) |
| intraday_final_correction | float | pre_guardrail_correction × guardrail_weight (final applied correction) |
| intraday_corrected_pred | float | base_pred + final_correction |
| intraday_confidence | float | Confidence score (0-1) |
| policy_decision | str | DISABLED / SHADOW_ONLY / LOW_WEIGHT / HIGH_WEIGHT |
| fusion_weight | float | Weight for fusion (0 for DISABLED/SHADOW, 0.12 for LOW, 0.22 for HIGH) |
| shadow_only_flag | bool | True if policy is SHADOW_ONLY |
| guardrail_reason | str | Reason if guardrail applied |
| observed_hours | list[int] | Hours used for residual estimation |
| n_observed | int | Number of observed hours |
| residual_std_today | float | Standard deviation of today's residuals |
| bias_direction | str | positive / negative / mixed / insufficient |

### Eval Mode Additional Fields

| Field | Type | Description |
|-------|------|-------------|
| y_true | float | Observed real-time price |
| baseline_error | float | base_pred - y_true |
| corrected_error | float | corrected_pred - y_true |

## Correction Pipeline (Phase 10)

```
intraday_base_correction = 0.40 × mean_residual + 0.35 × ewm_residual + 0.25 × last_residual
intraday_model_weight = confidence × exp(-distance_decay × distance) × std_penalty
intraday_pre_guardrail_correction = clip(base_correction × model_weight, ±max_abs_correction)
intraday_guardrail_weight = past_hour_weight × negative_price_weight × confidence_floor
intraday_final_correction = clip(pre_guardrail_correction × guardrail_weight, ±max_abs_correction)
intraday_corrected_pred = base_pred + intraday_final_correction
```

## Policy Gating (Phase 10)

| Condition | Decision | Fusion Weight |
|-----------|----------|---------------|
| mode != INTRADAY | DISABLED | 0 |
| n_observed < 3 | DISABLED | 0 |
| cutoff_hour < 10 | DISABLED | 0 |
| cutoff_hour < 12 | SHADOW_ONLY | 0 |
| confidence < 0.35 | SHADOW_ONLY | 0 |
| residual_std > 180 | SHADOW_ONLY | 0 |
| negative price risk | LOW_WEIGHT | 0.08 |
| confidence >= 0.55 AND cutoff >= 14 | HIGH_WEIGHT | 0.22 |
| default | LOW_WEIGHT | 0.12 |

## Constraints

1. **Mode must be INTRADAY.** The tracker never outputs full-day corrections.
2. **Minimum observed hours:** At least 3 observed hours required for correction (policy enforces).
3. **Max correction:** ±80 (configurable).
4. **Negative price guardrail:** When da_anchor < 0, correction weight reduced to 30%.
5. **Only future hours:** Hours <= cutoff_hour get zero correction.
6. **Policy overrides:** Policy decision can further restrict or disable correction.

## Backward Compatibility

- `intraday_correction` is an alias for `intraday_final_correction`
- Old field names (`intraday_raw_correction`, `intraday_correction_weight`) are deprecated
