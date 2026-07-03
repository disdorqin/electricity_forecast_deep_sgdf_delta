# DeepFinal-2 Realtime Model Results Summary

**Date:** 2026-07-03

---

## Summary

| Key Metric | DeepFinal-1 | DeepFinal-2 (修正后) |
|------------|-------------|---------------------|
| n_features | 2 | **36** |
| Feature Verdict | N/A | **FALLBACK_READY** (not FORMAL_READY) |
| formal_train_ready | False | **False** (requires real SGDFNet) |
| FULL_DAY leakage | 未知 | ✅ **无** (business_day aligned) |
| SGDFNet real coverage | 0% | **0%** (no real predictions) |
| SGDFNet effective coverage | 99.9% (fallback) | **99.9%** (fallback) |
| Calendar features | ❌ | ✅ 8/8 |
| Lag features | ❌ | ✅ 13/13 |
| Leakage | — | ✅ Passed |
| Metric status | 未标记 | **SMOKE_ONLY** |
| Test sMAPE (Feb 2026, TCN) | 26.76% | **26.69%** (SMOKE_ONLY) |
| SGDFNet reference | ~16.59% | — |

## Verdict

```
FORMAL_READY:          ❌ (sgdfnet_real_coverage=0%, fallback used)
FALLBACK_READY:        ✅ (n_features=36, effective coverage ≥95%)
DeepFinal-3 blocked:   YES — missing real SGDFNet predictions
```

## Progress

| Phase | n_features | sMAPE | Verdict | Blocker |
|-------|-----------|-------|---------|---------|
| DeepFinal-1 | 2 | 26.76% | NO-GO | Feature pipeline missing |
| DeepFinal-2 (修正前) | 34 | 26.69% | FORMAL_READY (误判) | — |
| DeepFinal-2 (修正后) | 36 | 26.69% (SMOKE_ONLY) | FALLBACK_READY | Real SGDFNet predictions |
| DeepFinal-3 | 36+ | — | — | **缺少真实 SGDFNet 预测文件** |

## Files

- Feature audit (fallback): `reports/local/deep_final/features_fallback/`
- Feature hardening report: `docs/DEEPFINAL_3A_FEATURE_HARDENING_REPORT.md`
- Training artifacts: `artifacts/trendknight_rt/exp_tcn_full_2026_02/`
- SGDFNet prediction contract: `docs/SGDFNET_PREDICTION_INPUT_CONTRACT.md`
