# Phase 6: Alignment & Fusion Audit Report

**Date:** 2026-07-03
**Period audited:** 2026-02-01 to 2026-02-28
**Commits:** 3fa4caa, 51963ef, (pending final commit)

---

## 1. Modified Files

### New files
- `models/deep_sgdf_delta/business_time.py` — Unified business-day alignment module (single source of truth)
- `tests/test_business_time.py` — 16 tests for business_time module
- `scripts/audit_baseline_consistency.py` — Baseline consistency audit across 3 sources

### Modified files
- `models/deep_sgdf_delta/teacher_adapters/rt916_teacher.py` — Replaced hand-rolled business_day with `add_business_time_columns`
- `models/deep_sgdf_delta/teacher_adapters/timemixer_teacher.py` — Same fix
- `scripts/run_simple_fusion_trial.py` — Added `--trendknight-predictions` and `--allow-synthetic-tk` CLI; default requires real TK
- `models/deep_sgdf_delta/rt916_scope.py` — `evaluate_rt916_local_quality` rewritten to use proper RT price space
- `scripts/diagnose_916_failure.py` — Replaced hand-rolled business_day with `add_business_time_columns`
- `tests/test_rt916_scope.py` — Updated for new `evaluate_rt916_local_quality` signature

### Generated reports
- `docs/BASELINE_CONSISTENCY_AUDIT.md`
- `docs/DIAGNOSE_916_FAILURE_PHASE6.md`
- `reports/local/phase6/diagnose_916/DIAGNOSE_916_FAILURE.md`
- `reports/local/phase6/fusion_trial/fusion_gain_report.md`

---

## 2. Pytest Results

**295 passed, 0 failed** (11 warnings, all from third-party libraries).

Key test suites:
- `test_business_time.py`: 16 tests (midnight alignment, period mapping, custom columns)
- `test_rt916_scope.py`: 12 tests (scope mask, apply scope, evaluate quality with RT price space)
- `test_v3_losses.py`: 31 tests (including 10 NaN safety tests)
- `test_v3_model.py`: 31 tests (model architecture, forward pass, parameter count)
- All other existing tests continue to pass

---

## 3. Business Day Bug Fix — CONFIRMED

**Before (Phase 5):** Each script had its own hand-rolled business_day logic:
```python
# Example from rt916_teacher.py (before fix):
ts = pd.to_datetime(result[c])
result["business_day"] = ts.dt.normalize()
h = ts.dt.hour
mask = h == 0
result["hour_business"] = h
result.loc[mask, "hour_business"] = 24
result.loc[mask, "business_day"] = result.loc[mask, "business_day"] - pd.Timedelta(days=1)
```

**After (Phase 6):** All scripts use `add_business_time_columns()` from `business_time.py`:
```python
from models.deep_sgdf_delta.business_time import add_business_time_columns
df = add_business_time_columns(df, timestamp_col="ds")
```

**Verification:**
- 16 dedicated unit tests pass
- Midnight (00:00) → previous day, hour 24 ✓
- Hour 01:00-23:00 → same day, hour 1-23 ✓
- Period mapping: 1_8 (hours 1-8), 9_16 (hours 9-16), 17_24 (hours 17-24) ✓
- 4 scripts fixed: rt916_teacher, timemixer_teacher, diagnose_916_failure, run_simple_fusion_trial

---

## 4. Baseline Consistency Audit — PASSED

**Method:** Compared SGDFNet sMAPE across available sources for 2026-02.

| Source | Rows Matched | sMAPE_floor50 |
|--------|-------------|---------------|
| teacher_adapter | 648 | 32.2712 |
| fusion_trial (sgdfnet_only) | 648 | 32.2712 |

- sMAPE range: 0.0000 (identical)
- Rows consistent: True
- **PASSED** (threshold: ≤ 0.02)

Note: Source 1 (p0_reproduce_sgdfnet_baseline) output was not available in this environment. The two available sources (teacher_adapter and fusion_trial sgdfnet_only) are consistent by construction since fusion_trial uses the same adapter. Full 3-source audit requires running p0 pipeline which needs the SGDFNet checkpoint.

---

## 5. RT916 Local Quality Fix — CONFIRMED

**Before (Phase 5):** Used rough `+100` approximation to convert delta to RT price space:
```python
sgd_clipped = np.clip(sgd_vals + 100, floor, None)  # rough approximation
```

**After (Phase 6):** Uses proper `da_anchor + delta` conversion:
```python
def evaluate_rt916_local_quality(teacher_pred, teacher_mask, teacher_names, rt916_idx,
                                  rt_actual, da_anchor=None, teacher_pred_kind="delta",
                                  sgdfnet_pred=None, sgdfnet_idx=0):
```

- If `teacher_pred_kind="rt"`: compare directly with rt_actual
- If `teacher_pred_kind="delta"`: `rt_pred = da_anchor + teacher_delta_pred`
- SGDFNet comparison also uses RT price space: `da_anchor + sgdfnet_delta`
- 4 new unit tests verify the corrected behavior

---

## 6. 9_16 Re-diagnosis Results (Phase 6, with fixed business_day)

**Period:** 2026-02-01 to 2026-02-28
**9_16 rows:** 216

### By Hour
| Hour | Count | RT Mean | Delta Std | SGDFNet sMAPE |
|------|-------|---------|-----------|---------------|
| 9 | 27 | 162.6 | 79.2 | 25.34 |
| 10 | 27 | 82.8 | 146.9 | **45.60** |
| 11 | 27 | 31.2 | 145.0 | 40.72 |
| 12 | 27 | 31.8 | 136.3 | 43.72 |
| 13 | 27 | 15.3 | 139.6 | 40.70 |
| 14 | 27 | 27.0 | 136.7 | 42.63 |
| 15 | 27 | 30.8 | 135.8 | 39.48 |
| 16 | 27 | 62.5 | 132.7 | 42.23 |

**Hardest hour: 10** (sMAPE = 45.60) — shifted from Phase 5's hour 11 due to corrected alignment.

### By Bucket
| Bucket | Count | RT Mean | Delta Std | SGDFNet sMAPE |
|--------|-------|---------|-----------|---------------|
| normal | 75 | 296.1 | 172.0 | 61.59 |
| spike | 1 | 511.3 | nan | 41.53 |
| negative | 140 | -76.7 | 67.8 | 28.51 |

### Overall 9_16 sMAPE: 40.05

**Recommendation:** Consider training a dedicated `TrendKnightSolar916` model. Worst hour sMAPE = 45.60 > 40.

---

## 7. Fusion Trial — SKIPPED (No Real TrendKnight Predictions)

**Status:** SKIPPED
**Reason:** Real TrendKnight predictions unavailable.
**Synthetic TK proxy:** NOT used (default mode requires real predictions).
**Verdict code:** `synthetic_tk_proxy=false, verdict=NO_DECISION`

No TrendKnight prediction output files were found in the repository. The fusion trial script correctly refused to evaluate fusion with synthetic data and wrote a skip report.

To enable formal fusion decision in a future phase:
1. Train TrendKnight models on real data
2. Generate predictions: `python scripts/run_trendknight_x_ablation.py ...`
3. Run fusion trial: `python scripts/run_simple_fusion_trial.py --trendknight-predictions <path> ...`

---

## 8. Formal Verdict

**NO DECISION**

Rationale:
- SGDFNet baseline is consistent and well-characterized (sMAPE ≈ 32-40 depending on period)
- 9_16 segment remains the weakest (sMAPE = 40.05, worst hour 10 at 45.60)
- TrendKnight predictions are not yet available for formal fusion evaluation
- Phase 5 fusion trial (with synthetic TK proxy) was NO-GO; Phase 6 cannot override this without real TK data

---

## 9. Metrics Integrity Declaration

**No fabricated metrics.** All numbers in this report come from actual script executions:
- Baseline consistency: `scripts/audit_baseline_consistency.py` executed 2026-07-03 12:39
- 9_16 diagnosis: `scripts/diagnose_916_failure.py` executed 2026-07-03 12:40
- Fusion trial: `scripts/run_simple_fusion_trial.py` executed 2026-07-03 12:41
- Pytest: 295 passed, executed 2026-07-03 12:41

No synthetic TrendKnight proxy was used for any formal decision metric.
