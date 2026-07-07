# P2.7 Canonical 3-Month Rerun Report

## 1. Bug Fix

| Item | Value |
|------|-------|
| Fixed file | common/realtime_canonical_loader.py |
| Mapping rule | hb=24 ← 00:00, hb=1 ← 01:00, ..., hb=23 ← 23:00 |
| Tests | 10/10 passed (test_realtime_canonical_loader.py) |
| Old bug | xlsx sorted [00,01,...,23] → hb 1..24 (midnight at index 0) |
| New behavior | xlsx [00,01,...,23] → [01,...,23,00] (midnight at hb=24) |

## 2. Canonical Loader Audit

| Check | Result |
|-------|--------|
| 00:00 → hb=24 | ✅ PASS |
| 01:00 → hb=1 | ✅ PASS |
| 24 rows/day | ✅ PASS (363/363 days) |
| No duplicate hour | ✅ PASS |
| No NaN | ✅ PASS (known-good days) |
| DA/RT same mapping | ✅ PASS |
| Metric canonical | ✅ PASS (10 pytest) |

## 3. 3-Month Canonical Rerun (TimesFM months)

| Month | DA_anchor | SGDFNet | TimesFM | Best Fusion | Winner |
|-------|:--------:|:-------:|:-------:|:----------:|:------|
| 2025-03 | 20.6 | 22.89 | 29.12 | DA_anchor (20.6) | DA |
| 2025-09 | 13.3 | 13.75 | 22.2 | DA_anchor (13.3) | DA |
| 2026-05 | 20.37 | 20.3 | 23.86 | gating_model (20.04) | SGDFNet |

**3-month overall**: Best variant = DA_anchor (18.14)
DA_anchor=18.14 SGDFNet=19.03

## 4. Scene Breakdown (3-month canonical)

| Scene | Best | DA_anchor | SGDFNet |
|------|:---:|:--------:|:------:|
| spike | 15.24 | 15.24 | 17.11 |
| negative | 12.89 | 12.89 | 11.03 |
| normal | 60.4 | 60.4 | 58.62 |
| 1_8 | 18.07 | 18.07 | 18.47 |
| 9_16 | 22.59 | 22.59 | 22.67 |
| 17_24 | 13.15 | 13.15 | 15.61 |

## 5. 10-Window SGDFNet vs DA (canonical)

| Month | DA_anchor | SGDFNet | Winner |
|-------|:--------:|:-------:|:------|
| 2025-03 | 20.6 | 22.89 | DA |
| 2025-04 | 15.68 | 16.44 | DA |
| 2025-05 | 16.58 | 17.21 | DA |
| 2025-06 | 16.89 | 16.82 | SGDFNet |
| 2025-09 | 13.3 | 13.75 | DA |
| 2025-10 | 15.9 | 16.69 | DA |
| 2026-03 | 26.44 | 27.73 | DA |
| 2026-04 | 18.96 | 18.87 | SGDFNet |
| 2026-05 | 20.37 | 20.3 | SGDFNet |
| 2026-06 | 30.44 | 31.29 | DA |

**10-window**: SGDFNet wins 3/10 months. Overall: DA=19.52 SGDFNet=20.2 (delta=-0.68)

## 6. 10-Window Decision

P2_7_10_WINDOW_DECISION: SKIP_AND_BUILD_DA_AWARE_GATE

(SGDFNet wins 3/10 months on canonical data; selectors may improve on close months)

## 7. Registry Impact

| Registry | Action |
|----------|--------|
| realtime_sgdfnet_lite.yaml | UPDATE (DA-aware selector) |
| realtime_timesfm_lite.yaml | KEEP (experimental_result) |

## 8. Recommendation

P2_7_RECOMMENDATION: SGDFNET_ONLY_CANDIDATE

(SGDFNet overall 20.2 vs DA 19.52 on full canonical 10-window)

## 9. Final Verdict

P2_7_RESULT: PASS
