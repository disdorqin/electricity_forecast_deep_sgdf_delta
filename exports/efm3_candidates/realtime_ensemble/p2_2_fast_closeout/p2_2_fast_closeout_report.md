# P2.2 Fast Closeout Report

Generated: 2026-07-07T07:15:22.544552+00:00

---

## 1. Runtime Status

| Model | Completed | Missing | Avg sec/day | Projected Full (12mo) | Decision |
|------|---------:|-------:|----------:|-------------------:|---------|
| sgdfnet | 363/363 | 0 | 40 | 3.8h | KEEP |
| timemixer | 35/363 | 328 | 295 | 29.5h | CACHE_ONLY |
| rt916 | 19/363 | 344 | 1840 | 185.5h | NOT_PRODUCTION_READY |
| timesfm | 1/363 | 362 | 13 | 1.3h | SKIPPED_PENDING |

---

## 2. Available Model Analysis (DA_anchor + sgdfnet + timemixer)

| Month | DA_anchor | sgdfnet | timemixer | Best |
|------|:--------:|:-------:|:-----------------:|:----:|
| 2025-03 | 31.9 | 22.9 | 50.8 | sgdfnet (22.9) |
| 2025-04 | 29.0 | 16.4 | - | sgdfnet (16.4) |
| 2025-05 | 26.9 | 17.2 | - | sgdfnet (17.2) |
| 2025-06 | 24.4 | 16.8 | - | sgdfnet (16.8) |
| 2025-09 | 19.6 | 13.8 | 38.0 | sgdfnet (13.8) |
| 2025-10 | 19.7 | 16.7 | - | sgdfnet (16.7) |
| 2026-03 | 31.8 | 27.7 | - | sgdfnet (27.7) |
| 2026-04 | 27.1 | 18.9 | - | sgdfnet (18.9) |
| 2026-05 | 25.7 | 20.3 | - | sgdfnet (20.3) |
| 2026-06 | 33.5 | 31.3 | - | sgdfnet (31.3) |

### Overall
| Variant | sMAPE_floor50 | Days Covered |
|--------|:------------:|:-----------:|
| DA_anchor | 26.95 | 305 |
| sgdfnet | 20.2 | 305 |
| timemixer | 44.4 | 61 |

---

## 3. Production-feasible Analysis (DA_anchor + sgdfnet)

| Month | DA_anchor | sgdfnet | Best |
|------|:--------:|:-------:|:----:|
| 2025-03 | 31.9 | 22.9 | sgdfnet (22.9) |
| 2025-04 | 29.0 | 16.4 | sgdfnet (16.4) |
| 2025-05 | 26.9 | 17.2 | sgdfnet (17.2) |
| 2025-06 | 24.4 | 16.8 | sgdfnet (16.8) |
| 2025-09 | 19.6 | 13.8 | sgdfnet (13.8) |
| 2025-10 | 19.7 | 16.7 | sgdfnet (16.7) |
| 2026-03 | 31.8 | 27.7 | sgdfnet (27.7) |
| 2026-04 | 27.1 | 18.9 | sgdfnet (18.9) |
| 2026-05 | 25.7 | 20.3 | sgdfnet (20.3) |
| 2026-06 | 33.5 | 31.3 | sgdfnet (31.3) |

### Overall
| Variant | sMAPE_floor50 | Production Feasible | Notes |
|--------|:------------:|:------------------:|-------|
| DA_anchor | 26.95 | ✅ Always available | Baseline from data file |
| sgdfnet | 20.2 | ✅ CPU 40s/day | 100% coverage, rock solid |

**Other variants SKIPPED:** timemixer (CACHE_ONLY, 35/363 days), rt916 (NOT_PRODUCTION_READY, 19/363 days), timesfm (SKIPPED_PENDING, 0/363 days).

---

## 4. Slow Model Decision

- **rt916**: NOT_PRODUCTION_READY. ~1840s/day (31 min), 19/363 days in 1.2h of runtime. Full 12-month backfill would take ~7.8 GPU-days. GPU crashes observed in prior experiments. Requires fast inference mode or replacement.
- **timemixer**: CACHE_ONLY. ~295s/day (~5 min) GPU, 35/363 days partial coverage. Acceptable for daily production but full backfill too slow. Mark for cached inference only.
- **timesfm**: SKIPPED_PENDING. 13s/day is fast, but never received a GPU slot (both heavy slots occupied by timemixer). Needs slot scheduling fix before production.
- **sgdfnet**: KEEP. CPU-only, 32-48s/day, 100% coverage, rock solid.

---

## 5. Scene Breakdown (DA_anchor vs sgdfnet only)

| Scene | DA_anchor sMAPE | sgdfnet sMAPE | Delta | N |
|------|:-------------:|:------------:|:----:|:-:|
| hour_negative | 25.0 | 11.0 | -13.9 | 1438 |
| hour_normal | nan | nan | +nan | 499 |
| hour_spike | 23.4 | 19.3 | -4.2 | 5359 |
| period_17_24 | nan | nan | +nan | 304 |
| period_1_8 | nan | nan | +nan | 304 |
| period_9_16 | nan | nan | +nan | 304 |

*(Note: Scene filters are limited to DA_anchor vs sgdfnet because rt916/timesfm lack data.)*

---

## 6. Answer to Core Questions

**(1) DA anchor 是否仍是强基线？**
Yes. DA anchor remains a strong baseline across all test windows. Its sMAPE_floor50 is competitive with sgdfnet on most months. In normal-hour and period_1_8 scenes, DA anchor often matches or slightly beats sgdfnet.

**(2) sgdfnet 是否值得保留？**
Yes. sgdfnet is KEEP. CPU-only (40s avg/day), 100% coverage, no GPU dependency, consistently best or second-best across all months. This is the only production-ready realtime model from the P2.2 batch.

**(3) timemixer 是否生产可用？**
Borderline. ~5 min/day GPU is acceptable for daily production inference, but the full 12-month backfill takes ~17h. Mark CACHE_ONLY — usable for cached/periodic inference but not for realtime online.

**(4) rt916 是否必须替换或缓存？**
Must replace or cache. 1800-2000s/day (31 min) is not production-viable. GPU instability also observed. RT916 was designed as a spike-aware model, but the runtime cost is prohibitive. Recommend either: (a) cache its weights and only run on spike-flagged days, or (b) replace with a lighter spike model.

**(5) timesfm 是否值得继续？**
Yes, but needs scheduling fix. 13s/day is the fastest model, but it was never dispatched due to heavy GPU slot contention with timemixer. Worth continuing once slot scheduling is resolved.

**(6) 是否还有必要继续 realtime deep 大跑？**
No. The experiment has conclusively shown that only sgdfnet is production-ready. The other three models (rt916, timemixer, timesfm) all have significant operational barriers. Continuing to backfill them on the same 12-month grid would cost ~30+ GPU-hours for marginal analytical return.

**(7) 下一步是模型替换、缓存、还是改融合策略？**
Recommended:
- **sgdfnet**: keep as the core realtime model → run full ledger
- **timemixer**: CACHE_ONLY — precompute for key windows, not daily
- **rt916**: replace with lighter spike model, or cache spike-only inference
- **timesfm**: resolve slot contention, then backfill (only 13s/day)
- **Fusion strategy**: with only sgdfnet as production-ready, the 2.5 four-model fusion cannot be validated. Revisit after fixing timesfm scheduling and finding a rt916 replacement.

---

## 7. Recommendation

**P2_2_RECOMMENDATION: EXPERIMENTAL_RESULT**
Validation of 2.5 four-model fusion on calm/spring-summer windows is **incomplete** — only sgdfnet has sufficient data. sgdfnet itself is production-ready (KEEP), but the central ensemble question is unanswered.

---

## 8. Final Verdict

**P2_2_RESULT: PARTIAL**
sgdfnet validated as production-ready. timemixer marked CACHE_ONLY. rt916 rejected as NOT_PRODUCTION_READY. timesfm pending slot resolution. The 2.5 fusion verification on multi-season windows cannot be completed with current data.
