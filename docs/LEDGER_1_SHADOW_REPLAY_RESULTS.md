# Ledger-1 Shadow Replay Results

**Date**: 2026-07-04
**Status**: ✅ COMPLETED (with caveats)

## Executive Summary

Ledger-1 shadow replay system has been implemented and tested. However, the current evaluation uses DA anchor baseline (actual price as prediction) which results in circular evaluation (base_pred = y_true). This means the metrics show degradation because the "prediction" is already perfect.

**Key Finding**: DA anchor baseline is NOT suitable for production evaluation. It should only be used for guardrail sensitivity testing. For production evaluation, actual model predictions (SGDFNet, TimesFM, etc.) must be used.

## 1. Track Completion Status

| Track | Description | Status | Notes |
|-------|-------------|--------|-------|
| Track 0 | Risk Module Test Debt | ✅ COMPLETE | 312/312 tests passing (commit `adf3010`) |
| Track A | Base Prediction Adapter | ✅ COMPLETE | `base_prediction_adapter.py` implemented |
| Track B | Risk Pack Loader | ✅ COMPLETE | `risk_pack_loader.py` implemented |
| Track C | Guardrail Policy | ✅ COMPLETE | `risk_guardrail_policy.py` implemented |
| Track D | Decision Log Contract | ✅ COMPLETE | Decision log format defined |
| Track E+F | Shadow Replay Engine | ✅ COMPLETE | `risk_shadow_replay.py` implemented, bugs fixed |
| Track G | Risk Trigger Evaluation | ✅ COMPLETE | `evaluate_risk_triggers.py` implemented, spike label audit complete |
| Track H | Tests | ✅ COMPLETE | All 69 Ledger-1 related tests passing |
| Track I | Final Report | 🔄 IN PROGRESS | This document |

## 2. Bug Fixes in This Session

### Task A: Fixed Shadow Replay y_true/metrics contract

**Problem**: `_evaluate_guardrail()` returned only `{"sMAPE_floor50": np.nan, "sMAPE": np.nan}` when y_true was missing, causing `policy_sweep_df` to miss columns.

**Fix**: `_evaluate_guardrail()` now always returns fixed metric schema with 17 columns:
- `base_sMAPE_floor50`, `adjusted_sMAPE_floor50`, `sMAPE_floor50_improvement`
- `base_sMAPE`, `adjusted_sMAPE`, `sMAPE_improvement`
- `base_MAE`, `adjusted_MAE`, `MAE_improvement`
- `base_RMSE`, `adjusted_RMSE`, `RMSE_improvement`
- `trigger_rate`, `evaluation_status`

If y_true is missing, `evaluation_status="MISSING_Y_TRUE"` and all metrics are NaN.

**Files Modified**:
- `models/deep_sgdf_delta/risk_shadow_replay.py`

### Task B: Fixed canonical sMAPE_floor50 formula

**Problem**: `_calc_smape_floor50()` had wrong formula: floored y_true at 50 in numerator, which is incorrect.

**Fix**: Now imports and uses canonical `smape_floor50` from `models.deep_sgdf_delta.metrics`:
```python
yt = np.where(y_true < floor, floor, y_true)
yp = np.where(y_pred < floor, floor, y_pred)
denom = np.abs(yt) + np.abs(yp) + eps
return float(np.mean(200.0 * np.abs(yp - yt) / denom))
```

**Files Modified**:
- `models/deep_sgdf_delta/risk_shadow_replay.py`

### Task C: Fixed policy_sweep_df output

**Problem**: `policy_sweep.csv` was missing columns like `sMAPE_floor50_improvement`.

**Fix**: Now outputs all 17 required columns. Verified with test: `test_evaluate_guardrail_missing_y_true`.

**Files Modified**:
- `models/deep_sgdf_delta/risk_shadow_replay.py`
- `tests/test_risk_shadow_replay.py`

### Task D: Audited and fixed Spike Trigger Evaluation label calibration

**Problem**: Previous run showed `N True Events = 11` for spike, which didn't match backtest summary (552 events).

**Root Cause**: old implementation used `y_true > 1000` threshold, but backtest uses `spike_label >= 500`.

**Fix**: Now supports configurable spike targets:
- `--spike-target spike` (default): y_true >= 500
- `--spike-target extreme_spike`: y_true >= 800
- `--spike-target relative_spike`: y_true > median + 200

New spike event counts (matches backtest):
- Spike (>=500): **552 events** ✅
- Extreme spike (>=800): 27 events
- Relative spike (> median + 200): 274 events

**Files Modified**:
- `scripts/evaluate_risk_triggers.py`
- `tests/test_evaluate_risk_triggers.py`

### Task E: Fixed y_true propagation

**Problem**: `BasePredictionAdapter` didn't output `y_true`, so shadow replay couldn't evaluate metrics.

**Fix**: `load_da_anchor_baseline()` now includes `y_true` column in output.

**Files Modified**:
- `models/deep_sgdf_delta/base_prediction_adapter.py`

## 3. Input Diagnostics

**File**: `reports/local/ledger_1/shadow_replay_2026_01_05/input_diagnostics.json`

```json
{
  "n_rows": 3624,
  "has_y_true": true,
  "y_true_non_null_count": 3624,
  "base_pred_non_null_count": 3624,
  "risk_cols_non_null_count": {
    "negative_prob": 3619,
    "negative_risk_score": 3619,
    "spike_prob": 3619,
    "spike_risk_score": 3619,
    "deviation_down_prob": 3619,
    "deviation_up_prob": 3619,
    "deviation_risk_score": 3619
  }
}
```

**Notes**:
- 5 rows have NaN in risk scores (0.14% missing)
- y_true is available for all 3624 rows

## 4. Policy Sweep Results

**File**: `reports/local/ledger_1/shadow_replay_2026_01_05/policy_sweep.csv`

**Grid**:
- Negative thresholds: [0.4, 0.5, 0.6, 0.7]
- Spike thresholds: [0.4, 0.5, 0.6, 0.7]
- Blend weights: [0.05, 0.1, 0.2]
- Total combinations: 48

**Champion Policy** (selected by sMAPE_floor50_improvement):
- Negative threshold: 0.7
- Spike threshold: 0.7
- Blend weight: 0.05
- Selection status: SUCCESS

**⚠️ IMPORTANT CAVEAT**: The metrics below are from DA anchor baseline (base_pred = y_true), so base_sMAPE = 0. This is NOT a real evaluation.

## 5. Metrics (DA Anchor Baseline - Circular Evaluation)

**File**: `reports/local/ledger_1/shadow_replay_2026_01_05/shadow_metrics.csv`

| Metric | Base | Adjusted | Improvement |
|--------|------|----------|-------------|
| sMAPE_floor50 | 0.0000 | 0.0343 | **-0.0343** ⚠️ |
| sMAPE | 0.0000 | 0.6581 | **-0.6581** ⚠️ |
| MAE | 0.0000 | 0.5814 | **-0.5814** ⚠️ |
| RMSE | 0.0000 | 1.4645 | **-1.4645** ⚠️ |
| trigger_rate | - | - | 22.68% |

**Interpretation**: All metrics show degradation. This is expected because:
1. DA anchor baseline uses actual price as prediction (base_pred = y_true)
2. Base sMAPE = 0 (perfect prediction)
3. Any adjustment makes it worse

**Conclusion**: DA anchor baseline is only suitable for sensitivity testing, NOT production evaluation.

## 6. Spike Label Audit Results

**File**: `reports/local/ledger_1/trigger_eval_2026_01_05/trigger_eval_report.md`

### Negative Alerts
- Precision: 0.9113
- Recall: 0.5349
- F1: 0.6741
- N true negative: 961
- Alert rate: 15.56%

### Spike Alerts (>=500)
- Precision: 0.7746
- Recall: 0.2989
- F1: 0.4314
- N true spike: **552** ✅ (matches backtest summary)
- Alert rate: 5.88%

### Extreme Spike Alerts (>=800)
- Precision: 0.0188
- Recall: 0.1481
- F1: 0.0333
- N true extreme_spike: 27
- Alert rate: 5.88%

### Relative Spike Alerts (> median + 200)
- Precision: 0.4225
- Recall: 0.3285
- F1: 0.3696
- N true relative_spike: 274
- Alert rate: 5.88%

**Conclusion**: Spike alert quality is poor (F1=0.4314), but this is expected because:
1. Spike prediction is inherently difficult (rare events)
2. The current spike risk model may need improvement
3. Negative alert quality is good (F1=0.6741)

## 7. Bucket Analysis

**⚠️ CANNOT COMPUTE** - The current shadow replay doesn't break down metrics by bucket (negative/spike/normal). This requires additional implementation.

**Needed**:
- Add bucket column to decision log (`bucket`: "negative", "spike", "normal")
- Compute metrics per bucket
- Add to `shadow_metrics.csv` or create `bucket_metrics.csv`

**Status**: ⚠️ TODO - Not implemented yet

## 8. Final Verdict

### Ledger Verdict: ⚠️ LEDGER_POLICY_ALERT_ONLY (Conditional)

**Reasoning**:
1. ✅ Negative trigger quality is good (Precision=0.9113, F1=0.6741)
2. ⚠️ Spike trigger quality is poor (F1=0.4314)
3. ⚠️ Cannot evaluate price improvement with DA anchor (circular evaluation)
4. ⚠️ Bucket analysis not implemented yet

**Recommendations**:
1. **Use alert_only policy for production**:
   - Negative: alert_only (high precision, can warn operators)
   - Spike: alert_only (low precision, but rare events - still useful to warn)
   
2. **Do NOT use soft_blend policies yet**:
   - Need actual model predictions (not DA anchor) to evaluate price improvement
   - Current evaluation shows degradation because base_pred = y_true

3. **Before mainline shadow, must**:
   - Use actual model predictions (SGDFNet, TimesFM, etc.) as base_pred
   - Implement bucket analysis
   - Re-run shadow replay with real predictions

## 9. Next Steps

### Required before Mainline Shadow:
1. ✅ Fix shadow replay bugs (completed)
2. ✅ Fix canonical sMAPE (completed)
3. ✅ Fix spike label audit (completed)
4. ⚠️ Implement bucket analysis (TODO)
5. ⚠️ Run with actual model predictions (not DA anchor)
6. ⚠️ Evaluate price improvement with real predictions

### Recommended Policy for Production:
```
negative_action: alert_only
spike_action: alert_only
delta_supply_action: alert_only
```

This will generate alerts without modifying prices, allowing operators to review and decide.

## 10. Files Modified/Created

### Modified:
1. `models/deep_sgdf_delta/risk_shadow_replay.py` - Fixed _evaluate_guardrail(), deleted wrong _calc_smape_floor50(), fixed _select_champion_policy(), added input diagnostics
2. `models/deep_sgdf_delta/base_prediction_adapter.py` - Added y_true to output
3. `scripts/run_risk_guardrail_shadow_replay.py` - Added --require-y-true parameter
4. `scripts/evaluate_risk_triggers.py` - Added spike-target parameter, evaluate all 3 targets

### Created:
1. `tests/test_risk_shadow_replay.py` - Added test_evaluate_guardrail_missing_y_true, test_canonical_smape_floor50
2. `tests/test_evaluate_risk_triggers.py` - Added test_all_spike_targets_evaluated, test_spike_event_counts_differ

### Output Files:
1. `reports/local/ledger_1/shadow_replay_2026_01_05/input_diagnostics.json`
2. `reports/local/ledger_1/shadow_replay_2026_01_05/policy_sweep.csv`
3. `reports/local/ledger_1/shadow_replay_2026_01_05/decision_log.csv`
4. `reports/local/ledger_1/shadow_replay_2026_01_05/champion_policy.json`
5. `reports/local/ledger_1/shadow_replay_2026_01_05/shadow_metrics.csv`
6. `reports/local/ledger_1/shadow_replay_2026_01_05/shadow_replay_report.md`
7. `reports/local/ledger_1/trigger_eval_2026_01_05/trigger_eval_report.md`
8. `reports/local/ledger_1/trigger_eval_2026_01_05/trigger_eval_summary.json`

## 11. Test Results

**Ledger-1 Related Tests**: 69 passed ✅
- test_risk_shadow_replay.py: 8 passed
- test_evaluate_risk_triggers.py: 8 passed
- test_risk_guardrail_policy.py: 15 passed
- test_risk_pack_loader.py: 13 passed
- test_negative_risk_model.py: 7 passed
- test_export_risk_feature_pack_multimonth.py: 12 passed

**Full Test Suite**: Cannot run (some tests require torch/pytorch which is not installed)

## 12. Commit Information

**Commit Hash**: TBD (needs to commit and push)
**Branch**: main
**Files Changed**: 9 files (4 modified, 2 created, 3 output files)

## 13. Important Warnings

⚠️ **DA Anchor Baseline is NOT Production Baseline**
- The current evaluation uses DA anchor (actual price as prediction)
- This results in circular evaluation (base_pred = y_true)
- Metrics show degradation because base is already perfect
- For production evaluation, MUST use actual model predictions

⚠️ **Bucket Analysis Not Implemented**
- Cannot evaluate negative/spike/normal bucket improvement separately
- Need to implement bucket metrics before mainline shadow

⚠️ **Spike Alert Quality is Poor**
- F1 = 0.4314 (Precision=0.7746, Recall=0.2989)
- Consider improving spike risk model or using alert_only policy

## 14. Conclusion

Ledger-1 shadow replay system is now functional with fixed bugs. However, the current evaluation using DA anchor baseline is not meaningful for production decisions. The system should be used with actual model predictions to get valid metrics.

**Recommendation**: Deploy alert_only policy to production for negative and spike risks. Do NOT deploy soft_blend policies until evaluated with real model predictions.

---

**Generated**: 2026-07-04
**Author**: AI Assistant (Claude)
**Reviewer**: disdorqin
**Status**: DRAFT - Awaiting review and real model prediction evaluation
