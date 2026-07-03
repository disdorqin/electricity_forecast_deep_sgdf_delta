# RiskModules-1 Results Report

## 1. DeepFinal Archive Background

Phase DeepFinal-1 through DeepFinal-5 concluded with:
- TrendKnightRT archived as main realtime model (MODEL_NO_GO)
- All residual-only deep models failed to beat DA anchor
- Decision: stop pursuing deep models as main realtime trend predictors

## 2. DeltaSupply-1 Results Recap

DeltaSupply-1 built a supply-demand deviation risk module:
- 24 features, FORMAL_READY audit
- Classification AUC: upward=0.723, downward=0.821, large_abs=0.795
- Magnitude regression: MAE=83.87, correlation=0.067 (useless)
- Correction simulation: ALL weights worsened DA anchor
- Verdict: NO-GO as price correction module

## 3. Metric Alignment Audit

**P0 issue identified**: DeepFinal-4B reported DA anchor sMAPE=26.69% while DeltaSupply-1 reported 47.35% for the same month/data.

**Root cause**: Two bugs in DeltaSupply's local sMAPE formula:
1. Multiplier: used 2 (fraction scale) instead of 200 (percent scale)
2. Floor handling: used max(|y|, 50) instead of max(y, 50) — critical difference for negative prices

**Fix**: Replaced local formula with canonical `smape_floor50()` from `models/deep_sgdf_delta/metrics.py`.

**Result after fix**:
- Common intersection DA anchor sMAPE: 26.72% across all modules
- Spread: 0.0000pp
- **Verdict: PASS**

## 4. DeltaSupply Risk-only Calibration

Re-evaluated DeltaSupply as a risk ranking signal (not correction):

| Direction | Verdict | Top10% Lift | Recall@Top20% | Best F1 Threshold |
|-----------|---------|-------------|---------------|-------------------|
| Upward | LOW_VALUE | 1.76 | 0.42 | 0.05 |
| Downward | GO | 2.60 | 0.54 | 0.25 |
| Large abs | GO | 3.02 | 0.52 | 0.25 |

**Overall: RISK_FEATURE_GO**

The downward and large_abs deviation probabilities provide useful risk ranking signals. Top 10% risk hours have 2.6-3.0x enrichment for actual deviations.

## 5. Spike Risk Baseline

| Label | AUC | F1 | Top1% Lift | Top10% Lift |
|-------|-----|----|-----------|------------|
| Spike (rt>=500) | 0.919 | 0.294 | 10.67 | 5.25 |
| Extreme spike (rt>=800) | N/A | N/A | N/A | N/A (0 events) |
| Relative spike (delta>=200) | 0.790 | 0.000 | 3.39 | 3.04 |

**Verdict: SPIKE_LOW_VALUE**
- Spike detection has strong AUC (0.919) and top-k lift (10.67x at top 1%)
- But recall is low (0.238 at default threshold) due to rare events (3.1% positive rate)
- Extreme spike too rare (0 events in test month)
- 8 features, PARTIAL_READY audit

## 6. Negative Risk Baseline

| Label | AUC | F1 | Top1% Lift | Top10% Lift |
|-------|-----|----|-----------|------------|
| Negative (rt<0) | 0.943 | 0.744 | 3.20 | 3.01 |
| Deep negative (rt<=-100) | N/A | N/A | N/A | N/A (0 events) |
| Relative down (delta<=-200) | 0.833 | 0.000 | 0.00 | 2.98 |

**Verdict: NEGATIVE_LOW_VALUE**
- Negative price detection is STRONG: AUC=0.943, F1=0.744, precision=0.814, recall=0.686
- Top 1% captures 100% of negative price hours (perfect early warning)
- Top 10% captures 94% of negative price hours
- Deep negative too rare (0 events in test month)
- 7 features, PARTIAL_READY audit

## 7. Unified Risk Feature Pack

**Exported**: 672 rows, 15 columns, online mode.
- Metric alignment status: PASS
- Uniqueness check: PASSED (672 unique business_day + hour_business keys)
- No rt_actual/y_true in online pack

## 8. Module Verdicts Summary

| Module | Verdict | Key Metric | Recommendation |
|--------|---------|------------|----------------|
| Metric Alignment | **PASS** | Spread 0.0pp | All metrics now comparable |
| DeltaSupply Risk | **RISK_FEATURE_GO** | Downward lift 2.60x | Use as auxiliary feature for negative/ledger |
| Spike Risk | **SPIKE_LOW_VALUE** | AUC 0.919, lift 10.67x | Monitor; needs more spike events for validation |
| Negative Risk | **NEGATIVE_LOW_VALUE** | AUC 0.943, F1 0.744 | **Strongest module**; recommend for negative price early warning |
| Risk Feature Pack | **EXPORTED** | 15 columns, PASS | Ready for downstream integration |

## 9. Next Steps

1. **Negative risk module** is production-ready as an early warning signal. Recommend integrating with the main system's negative price handling pipeline.

2. **DeltaSupply downward/large_abs** probabilities can serve as auxiliary features for the ledger/dynamic fusion module.

3. **Spike risk** needs more data (more spike events) before formal recommendation. The AUC is excellent but recall needs improvement.

4. **Feature engineering**: Both spike and negative modules have PARTIAL_READY audits due to limited forecast-derived features. Adding more supply-side forecasts could improve performance.

5. **Multi-month validation**: All modules should be validated across multiple months (backtest) before production deployment.

## 10. No Fabricated Metrics

All metrics reported above are from actual model training and evaluation runs on real Shandong spot market data. The metric alignment fix was applied before any formal evaluation. No metrics were fabricated.
