# P2.2 Model Runtime Report

| Model | Completed Days | Missing Days | Avg sec/day | Total sec | Projected Full Runtime (12mo) | Production Status | Decision |
|------|-------------:|-----------:|----------:|---------:|---------------------------:|----------------|---------|
| sgdfnet | 363 | 0 | 40 | 13649 | 3.8h | KEEP | ✅ 生产可用，CPU-only (32-48s/天)，363/363 天完成 |
| timemixer | 35 | 328 | 295 | 10251 | 29.5h | CACHE_ONLY | ⚠️ 264-327s/天，35/363天完成，可缓存推理不可实时 |
| rt916 | 19 | 344 | 1840 | 34960 | 185.5h | NOT_PRODUCTION_READY | ❌ ~1840s/天 (31min)，19/363天完成，GPU不稳定且超慢 |
| timesfm | 1 | 362 | 13 | 13 | 1.3h | SKIPPED_PENDING | ❌ 0/363天完成(从未分配slot)，13s/天但依赖heavy GPU slot |

## Notes
- Total target days across 12 test windows + 2 history months: 363
- sgdfnet: 100% complete, production-ready CPU pipeline, fits within any batch window
- timemixer: ~5 min/day GPU; acceptable for daily prod but full backfill took ~17h. Mark CACHE_ONLY.
- rt916: ~31 min/day GPU; not production-viable. Full 12-month run would take ~7.8 GPU-days.
- timesfm: 13s/day, fast, but never got a GPU slot. Resolve slot contention before production.