# P2.5 Lite Fusion Runtime Report

| Metric | Value |
|--------|------:|
| Analysis runtime | 1.2 min |
| Data windows (timesfm) | 2025-03, 2025-09, 2026-05 |
| Data windows (sgdfnet only) | All 10 |
| Models used | DA_anchor, sgdfnet, timesfm |
| Fusion variants | 10 |
| Gating model | LogisticRegression (hour-level) |
| P3 risk proxy | Proxy features (no actual P3 shadow) |
| Machine | CPU-only (epf-2 conda) |
| Inference cost per variant | Negligible (precomputed blend) |
