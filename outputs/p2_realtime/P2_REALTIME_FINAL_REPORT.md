# P2 Realtime Deep Open Exploration Report

_Generated: 2026-07-06T21:09:29_

## 1. Task Summary

- repo: `disdorqin/electricity_forecast_deep_sgdf_delta` (working repo)
- branch: main (clone HEAD 479ee3c)
- run_id: p2_realtime_20260706
- source 2.5 repo/path: `disdorqin/electricity_forecast_model2.5` (= local efm3.0); SGDFNet src via `electricity_forecast_model2.0_exp/SGDFNet`
- target task: realtime_trend (D+1 realtime price trend / delta learning, cutoff-safe)
- cutoff: **D14 (D日 14:00)** — strictly enforced; 2.5 prior D15 (decision_hour=15) results were NOT reused
- test data range: 2025-01-01 .. 2026-06-30 (536 business days)
- test months: all 18 months 2025-01..2026-06 (no skip; data complete)

## 2. Baseline Understanding

- 2.5 realtime models: timesfm / sgdfnet / timemixer / rt916 — all use D14 (`realtime_cutoff_hour=14`); produce rt = da_anchor + delta_pred.
- SGDFNet role: gradient-boosted residual (RT-DA) learner; production config default decision_hour=15 → reproduced at D14 by overriding config.
- RT916 role: spike-aware residual net (RT); reused only as experience, not re-run here.
- TimeMixer role: GPU temporal-mixing residual net; reused as experience.
- reused logic: data_contract (RT/DA cols), sMAPE_floor50 metric (floor=50), hour_business/period mapping, D14 cutoff protocol, 24-hour completeness check.
- replaced logic: from-scratch unified walk-forward framework (this repo) replaces per-model ad-hoc pipelines for fair comparison.
- metric implementation: `p2_common.capped_smape(floor=50)`, MAE/RMSE standard.
- cutoff prevention: features assembled only from business_day < T-1 full days + D-1 hours 1..14; lag-24 post-cutoff masked; target actual used ONLY for y_true/metrics.

## 3. Models Tried

| Model | Status | Notes |
| --- | --- | --- |
| DA-anchor (baseline) | SUCCESS | rt = DA price; strong single baseline 31.11% |
| SGDFNet D14 (faithful repro) | SUCCESS | 31.99%; confirms 2.5 single-model level |
| TCN (delta) | SUCCESS | 34.23%; ≈ DA anchor + noise |
| GRU (delta) | SUCCESS | 34.16%; best new deep model |
| DLinear (delta) | SUCCESS | 35.24% |
| Linear (delta) | SUCCESS | 35.35% |
| TCN (abs-target) | SUCCESS | 46.31%; directly predicting RT worse |
## 4. Overall Metrics

| Model | MAE | RMSE | sMAPE_floor50 | Train s | Infer s | NaN | Failed | Cutoff |
| --- | --: | --: | --: | --: | --: | --: | --: | --: |
| da_anchor | 64.0 | 110.8 | **31.11** | 0.0 | 0.0 | 0 | 0 | D14 |
| sgdfnet_d14 | 64.5 | 109.5 | **31.99** | 0.0 | 0.0 | 0 | 0 | D14 |
| gru_day | 67.1 | 110.4 | **34.16** | 0.0 | 0.0 | 0 | 0 | D14 |
| tcn_day | 68.1 | 112.3 | **34.23** | 0.0 | 0.0 | 0 | 0 | D14 |
| dlinear_day | 68.8 | 111.1 | **35.24** | 0.0 | 0.0 | 0 | 0 | D14 |
| linear_day | 69.1 | 111.6 | **35.35** | 0.0 | 0.0 | 0 | 0 | D14 |
| tcn_abs | 97.1 | 134.8 | **46.31** | 0.0 | 0.0 | 0 | 0 | D14 |
## 5. Period Metrics

| Model | 1_8 sMAPE | 9_16 sMAPE | 17_24 sMAPE |
| --- | --: | --: | --: |
| da_anchor | 22.61 | 53.07 | 17.65 |
| sgdfnet_d14 | 22.87 | 55.11 | 17.99 |
| gru_day | 23.80 | 60.44 | 18.24 |
| tcn_day | 23.71 | 59.87 | 19.12 |
| dlinear_day | 24.05 | 62.37 | 19.32 |
| linear_day | 24.39 | 62.78 | 18.88 |
| tcn_abs | 35.66 | 78.33 | 24.94 |
## 6. Spike / Negative Metrics

| Model | Spike sMAPE | Negative sMAPE | Normal Degradation |
| --- | --: | --: | --: |
| da_anchor | 20.61 | 64.06 | 31.11 |
| sgdfnet_d14 | 21.30 | 64.32 | 31.99 |
| gru_day | 21.69 | 79.29 | 34.16 |
| tcn_day | 22.47 | 77.55 | 34.23 |
| dlinear_day | 22.30 | 83.08 | 35.24 |
| linear_day | 23.53 | 83.28 | 35.35 |
| tcn_abs | 34.30 | 122.62 | 46.31 |
## 7. 2025 / 2026 Month Breakdown (vs DA-anchor baseline)

| Month | Baseline sMAPE | Best New Candidate (gru) | Candidate sMAPE | Winner |
| --- | --: | --- | --: | --- |
| 2025-01 | 30.40 | gru_day | 31.38 | baseline |
| 2025-02 | 27.21 | gru_day | 28.59 | baseline |
| 2025-03 | 31.61 | gru_day | 33.36 | baseline |
| 2025-04 | 25.03 | gru_day | 30.15 | baseline |
| 2025-05 | 26.63 | gru_day | 32.52 | baseline |
| 2025-06 | 29.86 | gru_day | 33.19 | baseline |
| 2025-07 | 22.11 | gru_day | 23.86 | baseline |
| 2025-08 | 16.95 | gru_day | 17.37 | baseline |
| 2025-09 | 18.88 | gru_day | 21.32 | baseline |
| 2025-10 | 18.29 | gru_day | 18.93 | baseline |
| 2025-11 | 31.92 | gru_day | 32.86 | baseline |
| 2025-12 | 34.86 | gru_day | 38.06 | baseline |
| 2026-01 | 50.18 | gru_day | 51.29 | baseline |
| 2026-02 | 47.34 | gru_day | 58.49 | baseline |
| 2026-03 | 38.39 | gru_day | 41.38 | baseline |
| 2026-04 | 36.77 | gru_day | 43.25 | baseline |
| 2026-05 | 35.12 | gru_day | 39.77 | baseline |
| 2026-06 | 45.05 | gru_day | 45.33 | baseline |
## 8. Cutoff Safety Report

- D14 mode implemented: YES (framework decision_hour=14; SGDFNet config overridden to 14)
- post-D14 realtime actual used: NO
- future (target-day) actual used: NO
- leakage risks: none found; audit ALL PASS
- verdict: **PASS (D14)**

## 9. Ablation Summary

| Change | Effect | Keep/Drop |
| --- | --- | --- |
| delta-target vs abs-target (TCN) | abs 46.3% >> delta 34.2% | DROP abs |
| blend da_anchor + sgdfnet (equal) | 31.17% ≈ da 31.11% | DROP (no gain) |
| optimal blend da+tcn | best w(da)=1.0 → 31.11% | DROP (tcn noise) |
| richer lags (24..168 already used) | no improvement | KEEP as-is |
| post-hoc all-equal ensemble | dragged to ~34% | DROP |
## 10. Candidate Package

- export path: `exports/efm3_candidates/realtime_trend/p2_realtime_20260706/`
- trend_predictions.csv: gru_day (best new deep candidate)
- metrics.json, comparison_report.md, cutoff_safety_report.md, ablation_report.md: present
- manifest.json, promotion_decision.json: present

## 11. Risks

- risk: RT-DA residual near-unpredictable from cutoff-safe features → mitigation: use DA anchor as trend, pursue ensemble diversity
- risk: single deep model cannot match 2.5 4-model fusion (~23%) → mitigation: reproduce timesfm/timemixer/rt916 to build diverse ensemble
- risk: over-fitting on 1 missing day → mitigation: lag-168 fallback, explicit failed_days record

## 12. Recommendation

REALTIME_P2_RECOMMENDATION: NO_GO

## 13. Final Verdict

P2_REALTIME_RESULT: PARTIAL

> No candidate beats the DA-anchor / faithful SGDFNet D14 single-model baseline; the open exploration is PARTIAL (framework + faithful baselines established, 7 candidates compared, cutoff-safe, ablation done) but yields NO_GO for promotion. The realtime trend signal is already captured by the DA anchor; ensemble gain requires the diverse 2.5 model set, not a new solo deep model.
