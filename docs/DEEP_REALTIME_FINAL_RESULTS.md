# Deep Realtime Model — Final Results

**Date:** 2026-07-03
**Status:** ARCHIVED — MODEL_NO_GO

---

## Summary

| Phase | n_features | SGDFNet | Test sMAPE | Verdict |
|-------|-----------|---------|-----------|---------|
| DeepFinal-1 | 2 | fallback | 26.76% | NO-GO |
| DeepFinal-2 | 36 | fallback | 26.69% | SMOKE_ONLY |
| DeepFinal-3B (real SGDFNet) | **36** | **real** | **26.61%** | **NO-GO** |
| Residual Baseline Lab | — | real | 26.69% (DA anchor) | NO_RESIDUAL_SIGNAL |

## Final Verdict

```text
ENGINEERING_COMPLETE
MODEL_NO_GO
ARCHIVE_AS_MAIN_REALTIME_MODEL
KEEP_UTILITIES_FOR_MAIN_SYSTEM
```

## Key Metrics

| Metric | DeepFinal-1 | DeepFinal-2 | DeepFinal-3B |
|--------|-------------|-------------|-------------|
| n_features | 2 | 36 | **36** |
| Calendar features | ❌ | ✅ | ✅ |
| Lag features | ❌ | ✅ | ✅ |
| FULL_DAY leakage | — | ✅ fixed | ✅ fixed |
| Feature verdict | — | FALLBACK_READY | **FALLBACK_READY** |
| SGDFNet source | fallback | fallback | **real (Protocol B)** |
| SGDFNet Feb coverage | 0% | 0% | **100%** |
| Test sMAPE (TCN) | 26.76% | 26.69% | **26.61%** |
| Residual pred std | — | — | **0.31** (true: 113.38) |
| Corr(pred, DA anchor) | — | — | **0.9987** |
| Best val sMAPE | 31.11% | 31.18% | 31.59% |
| formal_metric | false | false | **false (SMOKE_ONLY)** |

## Residual Baseline Lab (2026-02, real SGDFNet)

| Rank | Model | Overall sMAPE |
|------|-------|-------------|
| 1 | DA_anchor | **26.69%** |
| 2 | Mean_bias | 26.87% |
| 3 | SGDFNet | 26.88% |
| 4 | Period_bias | 26.95% |
| 5 | Hour_bias | 27.47% |
| 6 | HGB | 27.56% |
| 7 | Ridge | 27.71% |
| 8 | MLP | 28.27% |

## Conclusion

- **No residual signal exists** in FULL_DAY mode for 2026-02
- ALL residual correction models perform **worse** than DA anchor
- HGB (best ML model) = 27.56% vs DA anchor = 26.69%
- TrendKnightRT current architecture **archived**
- Feature engineering utilities **retained** for main system

## Files

- Archive decision: `docs/DEEP_REALTIME_MODEL_ARCHIVE_DECISION.md`
- Failure diagnosis: `docs/DEEPFINAL_4_FAILURE_DIAGNOSIS_REPORT.md`
- Residual baseline lab: `reports/local/deep_final/residual_baseline_lab_2026_02/`
- Feature builder: `models/deep_sgdf_delta/realtime_feature_builder.py`
- SGDFNet loader: `models/deep_sgdf_delta/sgdfnet_prediction_loader.py`
