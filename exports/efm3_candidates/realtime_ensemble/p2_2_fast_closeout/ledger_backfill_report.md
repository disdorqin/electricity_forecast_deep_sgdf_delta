# P2.2 Ledger Backfill Report

## Data Status

| Month | sgdfnet | timemixer | rt916 | timesfm | Has DA Actual | Has RT Actual |
|------|--------:|---------:|------:|-------:|:-----------:|:-----------:|
| 2025-03 | 31/31 | 1/31 | 4/31 | 1/31 | yes | yes |
| 2025-04 | 30/30 | 0/30 | 2/30 | 0/30 | yes | yes |
| 2025-05 | 31/31 | 0/31 | 3/31 | 0/31 | yes | yes |
| 2025-06 | 30/30 | 0/30 | 2/30 | 0/30 | yes | yes |
| 2025-09 | 30/30 | 16/30 | 2/30 | 0/30 | yes | yes |
| 2025-10 | 31/31 | 0/31 | 2/31 | 0/31 | yes | yes |
| 2026-03 | 31/31 | 0/31 | 0/31 | 0/31 | yes | yes |
| 2026-04 | 30/30 | 0/30 | 0/30 | 0/30 | yes | yes |
| 2026-05 | 31/31 | 0/31 | 0/31 | 0/31 | yes | yes |
| 2026-06 | 29/30 | 0/30 | 0/30 | 0/30 | yes | yes |

## Limitations (explicit)
- **rt916**: only 19/363 days collected (~5%), insufficient for any fusion variant or scene breakdown.
- **timesfm**: 0/363 days (worker was never dispatched due to GPU slot contention). Skipped entirely.
- **timemixer**: 35/363 days (Jan 1-19, Sep 1-16). Partial only; 2025-02~06, 2025-10, 2026-03~06 = 0 days.
- **sgdfnet**: 363/363 = 100% coverage. Only model with full dataset.
- **2.5 four-model fused (fused_2p5)**: not computable — only sgdfnet has sufficient coverage; GEF requires 4 models with 30d trailing window.
- **All 5 blend variants**: skipped due to insufficient multi-model data.
- **All scene breakdowns**: limited to DA_anchor vs sgdfnet comparison.