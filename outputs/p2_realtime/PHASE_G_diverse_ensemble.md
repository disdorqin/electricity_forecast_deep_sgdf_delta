# P2 Phase G — Diverse Ensemble Validation (production models)

- window: **2026-01-25 .. 2026-02-25** (32 days), D14 cutoff
- models in realtime ledger: ['rt916', 'sgdfnet', 'timemixer', 'timesfm']
- DA anchor = day-ahead price (日前电价) for target day; metric = capped sMAPE floor=50.0

## Result (lower is better)

| model | sMAPE_floor50 | 1_8 | 9_16 | 17_24 | spike | neg | MAE | RMSE |
|---|---|---|---|---|---|---|---|---|
| rt916 | 64.44 | 44.09 | 73.01 | 76.21 | 49.72 | 142.77 | 114.73 | 157.23 |
| sgdfnet | 48.67 | 34.85 | 54.74 | 56.41 | 30.84 | 82.40 | 82.63 | 126.85 |
| timemixer | 61.50 | 41.30 | 66.83 | 76.37 | 54.40 | 106.63 | 126.31 | 186.24 |
| timesfm | 51.46 | 36.01 | 55.46 | 62.91 | 42.09 | 96.70 | 95.29 | 138.78 |
| da_anchor | 45.49 | 32.13 | 50.71 | 53.63 | 31.31 | 76.87 | 80.57 | 127.00 |
| ensemble_equal_4 | 55.77 | 38.69 | 61.63 | 67.01 | 43.55 | 116.40 | 94.43 | 138.20 |
| ensemble_opt_4 {'rt916': 0.0, 'sgdfnet': 1.0, 'timemixer': 0.0, 'timesfm': 0.0} | 48.67 | 34.85 | 54.74 | 56.41 | 30.84 | 82.40 | 82.63 | 126.85 |
| ensemble_rolling_opt | 49.63 | 35.90 | 53.22 | 59.78 | 42.82 | 97.44 | 86.14 | 130.01 |

## Reading

- DA-anchor baseline on this window: **45.49%** (P2 full-536d figure was 31.11%; winter is harder).
- Best single production model: **sgdfnet = 48.67%** (still above DA anchor).
- Equal-weight 4-model ensemble: **55.77%** — WORSE than DA anchor (bad models drag it down).
- Static optimal-weight ensemble: **48.67%** (degenerates to sgdfnet-only).
- **Rolling optimal-weight ensemble (adaptive, mimics Ledger): 49.63%**.
- **2.5 FULL pipeline real measurement (2026-02-24): y_fused = 51.32%** vs DA anchor same day 46.97% → 2.5 fusion still below DA anchor.

## Hypothesis test (documented next step) — REFINED

- Single WEAK proxy architectures (tcn/gru/dlinear/linear) lost to DA anchor → confirmed earlier (NO_GO).
- Raw production diverse models (best sgdfnet 48.67%) are ALL still below DA anchor 45.49% on this window.
- Naive static blend (equal 55.77% / opt 48.67%) does NOT beat DA anchor — diversity ALONE is insufficient.
- Adaptive weighting (rolling-opt 49.63%) is STILL below DA anchor 45.49% on this window — adaptive weighting does not by itself recover the gap in winter.
- Even the 2.5 FULL pipeline on its one available day (2026-02-24) = 51.32% vs DA anchor 46.97% → also below DA anchor.
- **Revised conclusion (verified)**: on this 32-day WINTER window, EVERYTHING is below the DA anchor — raw production models, naive/adaptive blends, AND the 2.5 full fusion stack on the one available day. The realtime price in winter is dominated by extreme spike/negative regimes no method anticipates, so the day-ahead price is the strongest available predictor. The external '~23%' 2.5 reference is therefore period/metric-specific, NOT a universal win over the DA anchor.
- P2 NO_GO is reinforced: a solo deep net cannot help, and even the production fusion stack does not beat the DA anchor here. The realtime-trend task needs EITHER a calmer/longer evaluation window (where fusion may show its value) OR a genuinely new signal beyond the RT-DA residual. Re-running Phase G on a calm window (e.g. spring/summer 2025) is the decisive next test.

## Caveats

- Window is only 32 days (2026 winter). Small sample; extend via `ledger_backfill` for a firmer sMAPE.
- The 4-model ensemble here uses raw realtime predictions; 2.5 further applies Ledger dynamic weighting + extrem-price classifier correction, which explains the ~23% vs this naive blend.
- No training performed; this is a read-only validation of already-backfilled production predictions.