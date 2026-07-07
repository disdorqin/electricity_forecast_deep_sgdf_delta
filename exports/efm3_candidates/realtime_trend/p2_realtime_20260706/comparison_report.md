# P2 Realtime Candidate Comparison Report (Stage D)

- cutoff: D14 (D日14:00); range 2025-01 .. 2026-06 (0 days)

## Overall Metrics (lower is better)

| Model | MAE | RMSE | sMAPE_floor50 | 1_8 | 9_16 | 17_24 | spike sMAPE | neg sMAPE | train_s | infer_s | NaN | failed | cutoff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| da_anchor | 64.03 | 110.8 | **31.11** | 22.61 | 53.07 | 17.65 | 20.61 | 64.06 | 0 | 0 | 0 | 0 | D14 |
| sgdfnet_d14 | 64.54 | 109.48 | **31.99** | 22.87 | 55.11 | 17.99 | 21.3 | 64.32 | 0 | 0 | 0 | 0 | D14 |
| gru_day | 67.11 | 110.41 | **34.16** | 23.8 | 60.44 | 18.24 | 21.69 | 79.29 | 0 | 0 | 0 | 0 | D14 |
| tcn_day | 68.12 | 112.28 | **34.23** | 23.71 | 59.87 | 19.12 | 22.47 | 77.55 | 0 | 0 | 0 | 0 | D14 |
| dlinear_day | 68.85 | 111.07 | **35.24** | 24.05 | 62.37 | 19.32 | 22.3 | 83.08 | 0 | 0 | 0 | 0 | D14 |
| linear_day | 69.14 | 111.62 | **35.35** | 24.39 | 62.78 | 18.88 | 23.53 | 83.28 | 0 | 0 | 0 | 0 | D14 |

> Reference strong single baseline **da_anchor** sMAPE_floor50 = 31.11%
> External 2.5 fused realtime reference ≈ 23% (multi-model ensemble; not reproduced in this repo).

## Period Metrics

| Model | 1_8 sMAPE | 9_16 sMAPE | 17_24 sMAPE |
|---|---:|---:|---:|
| da_anchor | 22.61 | 53.07 | 17.65 |
| sgdfnet_d14 | 22.87 | 55.11 | 17.99 |
| gru_day | 23.8 | 60.44 | 18.24 |
| tcn_day | 23.71 | 59.87 | 19.12 |
| dlinear_day | 24.05 | 62.37 | 19.32 |
| linear_day | 24.39 | 62.78 | 18.88 |

## Spike / Negative / Normal

| Model | Spike sMAPE | Negative sMAPE | Normal count | Spike count | Neg count |
|---|---:|---:|---:|---:|---:|
| da_anchor | 20.61 | 64.06 | - | - | - |
| sgdfnet_d14 | 21.3 | 64.32 | - | - | - |
| gru_day | 21.69 | 79.29 | - | - | - |
| tcn_day | 22.47 | 77.55 | - | - | - |
| dlinear_day | 22.3 | 83.08 | - | - | - |
| linear_day | 23.53 | 83.28 | - | - | - |

## 2025 / 2026 Month Breakdown (vs DA-anchor baseline)

| Month | Baseline sMAPE | Best Candidate | Best sMAPE | Winner |
|---|---:|---:|---:|:--:|
| 2025-01 | 30.40 | gru_day | 31.38 | baseline |
| 2025-02 | 27.21 | gru_day | 28.59 | baseline |
| 2025-03 | 31.61 | gru_day | 33.36 | baseline |
| 2025-04 | 25.03 | tcn_day | 27.60 | baseline |
| 2025-05 | 26.63 | sgdfnet_d14 | 26.28 | sgdfnet_d14 |
| 2025-06 | 29.86 | sgdfnet_d14 | 31.03 | baseline |
| 2025-07 | 22.11 | dlinear_day | 22.73 | baseline |
| 2025-08 | 16.95 | tcn_day | 16.69 | tcn_day |
| 2025-09 | 18.88 | sgdfnet_d14 | 19.31 | baseline |
| 2025-10 | 18.29 | sgdfnet_d14 | 18.23 | sgdfnet_d14 |
| 2025-11 | 31.92 | sgdfnet_d14 | 30.54 | sgdfnet_d14 |
| 2025-12 | 34.86 | sgdfnet_d14 | 35.53 | baseline |
| 2026-01 | 50.18 | gru_day | 51.29 | baseline |
| 2026-02 | 47.34 | tcn_day | 48.43 | baseline |
| 2026-03 | 38.39 | tcn_day | 39.42 | baseline |
| 2026-04 | 36.77 | sgdfnet_d14 | 33.74 | sgdfnet_d14 |
| 2026-05 | 35.12 | sgdfnet_d14 | 33.26 | sgdfnet_d14 |
| 2026-06 | 45.05 | sgdfnet_d14 | 42.25 | sgdfnet_d14 |