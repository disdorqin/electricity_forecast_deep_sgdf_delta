# Phase 6 Frozen Baseline

**Frozen date:** 2026-07-03
**Status:** ACTIVE — all subsequent phases must respect these rules.

## Rules

1. **`models/deep_sgdf_delta/business_time.py`** is the single source of truth for business-day and hour-business alignment.
   - timestamp D 01:00-23:00 → business_day D, hour_business 1-23
   - timestamp D 00:00 → business_day D-1, hour_business 24
   - period: 1_8 (hours 1-8), 9_16 (hours 9-16), 17_24 (hours 17-24)

2. **All scripts must call `business_time.py`** via `add_business_time_columns()` or the individual functions. No hand-rolled alternatives are permitted.

3. **Baseline consistency (Phase 6):**
   - teacher_adapter sMAPE = 32.2712 (648 rows, 2026-02)
   - fusion_trial sgdfnet_only sMAPE = 32.2712 (648 rows, 2026-02)
   - Status: `PARTIAL_PASS_2_SOURCE` (p0 source not available in current environment)

4. **Fusion formal verdict:** `NO_DECISION`
   - No real TrendKnight predictions available
   - Phase 5 synthetic TK proxy NO-GO is NOT a real TrendKnight verdict

5. **9_16 confirmed as primary weakness:**
   - 2026-02 9_16 overall sMAPE = 40.05
   - Hardest hour: 10 (sMAPE = 45.60)
   - Normal bucket sMAPE = 61.59

6. **Prohibited patterns:**
   - No re-introducing hand-written `dt.normalize() - pd.Timedelta(days=1)` logic
   - No `hour == 0` → business_day adjustment outside `business_time.py`
   - Enforcement: `scripts/check_no_handrolled_business_time.py`

## Change Protocol

To modify business-day rules:
1. Edit `business_time.py` only
2. Update `tests/test_business_time.py`
3. Run full pytest suite
4. Update this document with new frozen values
