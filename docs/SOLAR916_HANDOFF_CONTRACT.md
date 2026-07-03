# Solar916 Handoff Contract

**Version:** 1.0
**Date:** 2026-07-03
**Status:** Phase 7 — GO

## Purpose

This document defines the interface contract between the Solar916 residual correction
specialist and the downstream ledger/fusion system.

## Scope

Solar916 operates **only on 9_16 hours** (business hours 9-16). It does not produce
corrections for 1_8 or 17_24 periods.

## Output Schema

The correction pack CSV contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| business_day | datetime | Business day (Phase 6 aligned) |
| hour_business | int | Business hour (9-16) |
| ds | datetime | Original timestamp |
| base_model_name | str | Name of base model (e.g. "sgdfnet") |
| base_pred | float | Base model prediction (RT price) |
| solar916_residual_pred | float | Predicted residual (rt - base_pred) |
| solar916_corrected_pred | float | base_pred + solar916_residual_pred |
| solar916_confidence | float | 0-1 confidence score |
| solar916_trigger_flag | bool | Always True for 9_16 hours |
| feature_missing_flag | bool | True if any features were missing |
| correction_reason | str | Reason for correction |

## Eval Mode Extensions

When `--mode eval` is used, additional columns are included:

| Column | Type | Description |
|--------|------|-------------|
| y_true | float | Actual RT price |
| residual_true | float | Actual residual (y_true - base_pred) |

## Constraints

1. **No future actuals:** Online mode must NOT contain y_true or residual_true.
2. **9_16 only:** All rows must have hour_business in [9, 10, 11, 12, 13, 14, 15, 16].
3. **Business day alignment:** Must use `business_time.py` rules.
4. **Ledger compatibility:** Can be read by the fusion/ledger system via standard CSV.

## Integration Points

The correction pack can be consumed by:
- `run_simple_fusion_trial.py --solar916-corrections <path>`
- Custom ledger integration scripts
- Offline evaluation pipelines

## Phase 7 Results Summary

- 9_16 baseline sMAPE: 40.87
- 9_16 corrected sMAPE: 36.86
- Improvement: 4.01
- Verdict: GO
- Normal bucket improvement: 18.47
- Hour 10 improvement: 6.48
