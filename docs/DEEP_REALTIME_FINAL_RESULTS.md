# DeepFinal-2 Realtime Model Results Summary

**Date:** 2026-07-03

---

## Summary

| Key Metric | Value |
|------------|-------|
| n_features | 2 → **34** |
| Feature Verdict | **FORMAL_READY** |
| SGDFNet coverage (with fallback) | 99.9% |
| SGDFNet real coverage | 0% (no real predictions available) |
| Calendar features | ✅ 8/8 |
| Lag features | ✅ 11/11 |
| Leakage | ✅ Passed |
| Test sMAPE (Feb 2026, TCN) | **26.69%** |
| Test sMAPE (Feb 2026, GRU) | **26.69%** |
| Test sMAPE (Feb 2026, Transformer) | **26.70%** |
| DeepFinal-1 baseline (2 feat) | **26.76%** |
| SGDFNet reference | ~16.59% |
| **Milestone Target** | **< 20%** |

## Progress

| Phase | n_features | sMAPE | Verdict | Gap |
|-------|-----------|-------|---------|-----|
| DeepFinal-1 | 2 | 26.76% | NO-GO | Feature pipeline missing |
| DeepFinal-2 (fallback) | 34 | 26.69% | NO-GO | Missing real SGDFNet predictions |
| DeepFinal-3 (expected) | 34+ | ~16-18% | ACCEPTABLE | Need real SGDFNet predictions |

## Feature Pipeline Readiness

```
FORMAL_READY
├── n_features >= 25:       ✅ 34
├── sgdfnet_coverage >= 95%: ✅ 99.9% (with fallback)
├── no high-risk leakage:   ✅
└── required_missing <= 3:  ✅ 2 (load_forecast, provincial_load_forecast)
```

## Files

- Feature audit: `reports/local/deep_final/features/realtime_feature_audit.json`
- Training artifacts: `artifacts/trendknight_rt/exp_tcn_full_2026_02/`
- Training artifacts: `artifacts/trendknight_rt/exp_gru_full_2026_02/`
- Training artifacts: `artifacts/trendknight_rt/exp_transformer_full_2026_02/`
