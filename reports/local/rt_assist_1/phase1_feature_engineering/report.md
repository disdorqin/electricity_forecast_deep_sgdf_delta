# Phase 1: Feature Engineering Report

## Summary

**Goal**: Test if more features help reduce sMAPE.

**Methods**:
1. Minimal features (27 features): calendar + DA anchor + forecast features
2. Comprehensive features (55 features): + lags + rolling stats + interactions

**Results**:

| Feature Set | Mean Improvement | Verdict |
|-------------|-------------------|---------|
| Minimal | 0.0000pp | ❌ No improvement |
| Comprehensive | 0.0129pp | ❌ No significant improvement |

**Detailed Results**:

### Minimal Features
- 2026-02: Model sMAPE = 27.87, DA sMAPE = 27.87, Improvement = 0.0
- 2026-03: Model sMAPE = 19.59, DA sMAPE = 19.59, Improvement = 0.0
- 2026-04: Model sMAPE = 15.43, DA sMAPE = 15.43, Improvement = 0.0
- 2026-05: Model sMAPE = 16.58, DA sMAPE = 16.58, Improvement = 0.0

### Comprehensive Features
- 2026-02: Model sMAPE = 27.87, DA sMAPE = 27.87, Improvement = 0.0
- 2026-03: Model sMAPE = 19.54, DA sMAPE = 19.59, Improvement = **0.05pp**
- 2026-04: Model sMAPE = 15.43, DA sMAPE = 15.43, Improvement = 0.0
- 2026-05: Model sMAPE = 16.58, DA sMAPE = 16.58, Improvement = 0.0

## Conclusion

**Verdict**: ❌ **FEATURE ENGINEERING KILL**

1. Comprehensive features only improve 0.0129pp on average (essentially 0)
2. Only 2026-03 shows tiny improvement (0.05pp)
3. All other months: 0 improvement

**Why?**:
- DA-only is already a very strong baseline (corr(RT, DA) = 0.85)
- Residual is unpredictable (|pearson corr| < 0.1 for all features)
- Adding more features cannot magically make residual predictable

## Next Steps

According to user instructions: "效果好的改动保留，一直没有效果甚至更差就不要这个改动"

→ **Discard comprehensive features**
→ **Continue to Phase 2: Data classification + bucket-specific handling**

## Files

- Script: `scripts/phase1_feature_engineering_v2.py`
- Results: `reports/local/rt_assist_1/phase1_feature_engineering/`
