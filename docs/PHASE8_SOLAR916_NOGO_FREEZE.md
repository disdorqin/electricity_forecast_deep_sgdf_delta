# Phase 8: Solar916 Offline Residual Correction — NO-GO Freeze

**Date:** 2026-07-03
**Status:** FROZEN — NO-GO
**Decision:** Solar916 offline residual correction is NOT approved for production use.

---

## Findings

1. **Solar916 offline residual correction = NO-GO.**
   The approach of training an offline tree model (HistGradientBoosting) to predict SGDFNet residuals for 9_16 hours does not produce reliable improvements.

2. **Phase 7 leaky result (+4.01) cannot be used for production.**
   Phase 7 reported a +4.01 sMAPE improvement, but the feature engineering contained cross-hour lag misalignment (shift(1) on 9_16 subset retrieved previous row, not same-hour previous day) and rolling target leakage (rolling residual features included current row's residual).

3. **No-leak 2026-02 corrected sMAPE = 53.20, worse than baseline 40.87.**
   After fixing all leakage issues with merge-based same-hour lag lookup and shift(1) rolling features, the model uniformly degraded predictions across all hours. Improvement: -12.33.

4. **Guardrail does not rescue the model.**
   Even with guardrail (hours 9/11 disabled, negative risk weight=0), the overall corrected sMAPE = 45.36, still worse than baseline 40.87.

5. **Root cause: residual distribution non-stationarity.**
   SGDFNet's mean residual shifted from +68.72 in January to +5.60 in February. Any model trained on historical residuals fails to adapt to this shift.

## Prohibitions

- **Do NOT** continue optimizing the Solar916 offline residual model as a main fusion candidate.
- **Do NOT** use Phase 7 leaky results for production decisions or reporting.
- **Do NOT** include Solar916 offline corrections in the fusion pipeline.

## Allowed New Direction

A new approach is permitted: **Intraday Adaptive Residual Tracker** (Phase 9).

This approach differs fundamentally from the offline model:
- It uses **same-day observed residuals** (hours that have already occurred) rather than historical residuals from different months.
- It operates in **INTRADAY mode only**, activated after a cutoff hour when some actual prices are available.
- It does **not** attempt full-day day-ahead correction.

This avoids the non-stationarity problem because it adapts to the current day's bias pattern using real-time observations.
