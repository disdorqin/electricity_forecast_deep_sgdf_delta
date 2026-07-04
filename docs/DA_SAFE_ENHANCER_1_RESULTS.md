# DA-Safe-Enhancer-1 Results

## Executive Summary

**Goal**: Improve RT price prediction by safely modifying DA predictions (not replacing them).

**Target**: Mean monthly sMAPE improvement >= 0.3pp, no month worse than DA by > 0.5pp.

**Result**: **DA_SAFE_STABLE** (mean improvement = 0.02pp, stable but < 0.3pp target).

**Key Achievement**: Safety guards successfully prevented 2026-02 catastrophic failure (sMAPE: 72.55 → 27.87).

---

## 1. Why DA-Safe-Enhancer?

Previous attempts (DeepRT-SOTA-2B, 3, 3C) all failed:
- Global residual-to-DA models: NO_GO (best 26.77 vs DA 26.69)
- Conditional specialist: Short-term KEEP, but 2026-02 disaster (fire rate 27.53%, sMAPE 72.55)
- Residual unpredictability: corr(residual, lag_24h) ≈ 0

**Conclusion**: DA-only is a very strong baseline (corr(RT, DA) = 0.85). We should not try to replace it; instead, make small, safe corrections.

---

## 2. Phase A: 2026-02 Failure Audit

**Goal**: Understand why 2026-02 had catastrophic failure.

**Findings**:
1. **2026-02 is unusual**: 22.8% negative DA (vs 9.8% in 2026-01)
2. **Low DA quantile has very high error**: DA sMAPE = 92.64
3. **Hours 11-14 are worst**: sMAPE > 68
4. **Fire rate too high**: 27.53% of hours were corrected

**Fire rate simulation**:
- Best improvement with fire rate caps: **0.0006pp** (essentially 0)
- **No config achieves >= 0.3pp improvement**

**Conclusion**: Even with safety limits, improvement is tiny. But safety limits are necessary to prevent disasters.

---

## 3. Phase B: DA-Safe Correction Simulator

**Goal**: Test conservative correction strategies without training new models.

**Strategies tested** (66 total):
1. top-k only correction (1%, 3%, 5%, 10%)
2. small-alpha correction (0.02, 0.05, 0.10, 0.20)
3. small-clip correction (5, 10, 20, 30, 50)
4. bucket-only correction (low_da, negative_da)

**Results**:
- Best validation improvement: **0.029pp** (bucket_only, alpha=0.10, clip=20)
- Best test improvement: **0.014pp** (top_k, alpha=0.10, clip=30)
- Mean test improvement: **0.0049pp**
- Max damage: **-0.0065pp**

**Verdict**: **AUX_KEEP** (improvement > 0 but < 0.3pp, no significant damage).

---

## 4. Phase C: Safety Guard Design

**Goal**: Implement safety gates to prevent catastrophic failures.

**Guards implemented** (`models/deep_sgdf_delta/da_safe_guard.py`):
1. **Max fire rate per month** (default 5%)
2. **Max absolute correction** (default 20)
3. **Validation regret guard** (block if validation improvement ≤ 0)
4. **Distribution shift guard** (block if feature distribution drifts)
5. **Low confidence guard** (block if trigger score margin < 10%)
6. **Normal bucket damage guard** (block if normal bucket harmed > 0.2pp)

**Test results**:
- With default config (max_fire_rate=5%, max_correction=20):
  - Fire rate correctly limited to **5.10%**
  - 939/1000 corrections blocked by fire rate guard
- With validation regret guard (improvement=0):
  - Fire rate = **0%** (all blocked)

---

## 5. Phase D: Safe Specialist Backtest

**Goal**: Re-run specialist with safety guards on 2026-02 to 2026-05.

**Method**:
- Walk-forward backtest (train on all prior data, test on target month)
- Use validation to select alpha/clip/top-k
- Apply safety guards

**Results**:

| Month | Model sMAPE | DA sMAPE | Improvement | Fire Rate |
|-------|--------------|-----------|-------------|------------|
| 2026-02 | 27.87 | 27.87 | **+0.00** | 5.06% |
| 2026-03 | 19.53 | 19.59 | **+0.06** | 5.11% |
| 2026-04 | 15.41 | 15.43 | **+0.02** | 5.14% |
| 2026-05 | 16.58 | 16.58 | **+0.00** | 5.11% |
| **Mean** | **19.85** | **19.87** | **+0.02** | **5.10%** |

**Key achievements**:
1. ✅ **2026-02 no longer catastrophic** (27.87 vs previous 72.55)
2. ✅ **All months have improvement ≥ 0** (4/4 months)
3. ✅ **Fire rate correctly limited** to ~5%
4. ✅ **No month damaged** (max damage = 0)

**Verdict**: **DA_SAFE_STABLE**
- Mean improvement > 0 (+0.02pp)
- No month worse than DA by > 0.2pp (max damage = 0)
- But **far from 0.3-2.0pp target**

---

## 6. Phase E: Uncertainty Sidecar (Skipped)

**Reason**: DA-safe price correction achieved DA_SAFE_STABLE (not NO_GO).

**Decision**: Skip uncertainty sidecar. Focus on improving correction signal instead.

---

## 7. Conclusion

### What Worked
1. **Safety guards are essential**: Prevented 2026-02 disaster
2. **Conservative corrections are stable**: No month damaged
3. **Fire rate limiting works**: Correctly limited to ~5%

### What Didn't Work
1. **Improvements are tiny** (< 0.1pp)
2. **Residual unpredictability**: All feature correlations < 0.1
3. **Cannot reach 0.3pp target** with current features/models

### Why Improvement is Hard
1. **DA is very strong baseline** (corr(RT, DA) = 0.85)
2. **Residual ≈ white noise** (no autocorrelation)
3. **Available features have weak signal** (|pearson corr| < 0.1)

---

## 8. Final Verdict

**Result**: **DA_SAFE_STABLE**

**Metrics**:
- Mean model sMAPE: **19.85**
- Mean DA sMAPE: **19.87**
- Mean improvement: **+0.02pp** (stable but < 0.3pp target)
- Months with improvement: **4/4**
- Max month damage: **0.00pp**

**Achievements**:
1. ✅ Implemented safety guards (prevent disasters)
2. ✅ Stable across all test months
3. ✅ No catastrophic failures

**Limitations**:
1. ❌ Cannot reach 0.3pp improvement target
2. ❌ Corrections are tiny (essentially DA-only)
3. ❌ Not worth deploying (complexity > benefit)

---

## 9. Recommendations

### Option A: Accept DA-Only (Recommended)
- DA-only is already very strong (19.87 sMAPE)
- Corrections add complexy but little value
- Focus on other tasks (e.g., uncertainty quantification, feature engineering)

### Option B: Continue Research
- Try deep learning models (LSTM, Transformer) with safety guards
- Collect more features (weather, grid status, etc.)
- Try different targets (quantile regression, classification)

### Option C: Deploy with Sidecar
- Deploy DA-only as main model
- Add uncertainty sidecar (warn when DA might be wrong)
- Use trigger model for risk alerts (not corrections)

---

## 10. Files Modified/Created

### Scripts
1. `scripts/analyze_2026_02_specialist_failure.py` (Phase A)
2. `scripts/run_da_safe_correction_simulator.py` (Phase B)
3. `models/deep_sgdf_delta/da_safe_guard.py` (Phase C)
4. `scripts/run_da_safe_specialist_backtest.py` (Phase D)

### Reports
1. `docs/DA_SAFE_ENHANCER_1_RESULTS.md` (this file)
2. `reports/local/deep_rt_sota/da_safe_enhancer/failure_audit_2026_02/`
3. `reports/local/deep_rt_sota/da_safe_enhancer/simulator_2026_01_05/`
4. `reports/local/deep_rt_sota/da_safe_enhancer/backtest_2026_02_05/`

---

## 11. Next Steps

1. **Decision**: Accept DA_SAFE_STABLE or continue research?
2. **If continue**: Try deep models (Phase D of original plan)
3. **If stop**: Focus on uncertainty sidecar or other tasks
4. **Recommendation**: Accept DA-only, investigate why 2026-02 has 22.8% negative DA

---

## 12. Commit Hash

To be generated after pushing to `disdorqin/electricity_forecast_deep_sgdf_delta`.

---

## 13. Metric Integrity

- ✅ All metrics computed with day-level sMAPE
- ✅ Walk-forward backtest used (no data leakage)
- ✅ No test actual used for strategy selection
- ✅ No oracle baselines
- ✅ No fake metrics

**Final verdict**: **DA_SAFE_STABLE** (stable but insufficient improvement).
