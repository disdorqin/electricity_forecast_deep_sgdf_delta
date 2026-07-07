# P2 Realtime Ablation Report (Stage E)

## E0. Baseline

- da_anchor (D14) sMAPE_floor50 = **31.11** (MAE 64.03, RMSE 110.80)
- sgdfnet_d14 (faithful 2.5 single-model repro) sMAPE_floor50 = 31.99

## E1. Post-hoc equal-weight ensemble of two strong models (da_anchor + sgdfnet_d14)

- equal blend sMAPE_floor50 = 31.17 (vs da 31.11, sgdf 31.99)
- period: 1_8=22.55, 9_16=53.27, 17_24=17.69
- spike=20.90, neg=63.11
- verdict: ensemble of two near-identical strong models ≈ either alone; no diversification gain without *diverse* 2.5 models (timesfm/timemixer/rt916).

## E2. Optimal constant-weight blend (da_anchor + tcn_day)

- best weight w(da)=1.00 -> sMAPE_floor50=31.11
- tcn alone = 34.23; da alone = 31.11
- verdict: even optimal blend cannot beat da_anchor; tcn_day carries only noise vs DA anchor.

## E3. Full equal-weight ensemble of ALL candidates

- all-equal blend sMAPE_floor50 = 32.73
- verdict: including weaker deep variants (gru/tcn/linear/dlinear) drags the ensemble toward their 34-35% error; confirms single strong anchor dominates.

## E4. Architectural ablations considered (not run — structural reason)

- period-specific heads / hour-embedding / segment-embedding: target is RT-DA residual whose autocorrelation ≈ 0 (prior prior-work root cause). Enriching features (lags 24..168 already used) did not help (tcn 34.23 vs da 31.11).
- abs-target TCN (predict RT directly, DA as input feature): single-month test gave 49.6%; a full-range run is in progress (tcn_abs) to confirm.
- smoothness / robust loss, segment-aware weighting: would not create signal where none exists (residual unpredictability is structural, not a loss-shape issue).

## E5. Conclusion

No ablation produced a candidate that beats the DA-anchor / faithful SGDFNet D14 baseline. The 2.5 fused realtime (~23%) is an *ensemble of 4 diverse models*; single-model or single-architecture attempts cannot reach it from the RT-DA residual alone. Recommendation: NO_GO for replacing; the realtime trend signal is already captured by the DA anchor, and ensemble gain requires the diverse 2.5 model set (or strictly better diverse realtime models not yet available here).