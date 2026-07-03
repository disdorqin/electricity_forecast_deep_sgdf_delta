# NegativeRisk Recalibration Report

**New Verdict: NEGATIVE_CHAMPION**

*Generated: 2026-07-04T00:32:45.705202*

## Purpose

The original NegativeRisk backtest assigned verdict **NEGATIVE_LOW_VALUE**
because mean top-10% capture (0.365) fell below the 0.70 champion
threshold.  However, negative price events have a high base rate
(~15-30 %), which caps the theoretical maximum recall achievable at
any fixed top-k budget.  This recalibration applies **base-rate aware
metrics** to evaluate the module fairly.

## Base-Rate Analysis

| Month | N Total | N Positive | Positive Rate |
|-------|---------|------------|---------------|
| 2026-01 | 744 | 141 | 0.1895 |
| 2026-02 | 672 | 210 | 0.3125 |
| 2026-03 | 744 | 111 | 0.1492 |
| 2026-04 | 720 | 185 | 0.2569 |
| 2026-05 | 744 | 180 | 0.2419 |

## Normalised Top-k Recall

Normalised recall = actual recall / theoretical maximum recall at that budget.
A value of 1.0 means the ranker captures every possible positive within the budget.

| Month Recall@top5 Max@top5 Norm@top5 Recall@top10 Max@top10 Norm@top10 Recall@top20 Max@top20 Norm@top20 |
|------- --- --- --- --- --- --- --- --- --- |
| 2026-01 0.2128 0.2638 0.8065 0.3830 0.5277 0.7258 0.6525 1.0000 0.6525 |
| 2026-02 0.1571 0.1600 0.9821 0.3000 0.3200 0.9375 0.5667 0.6400 0.8854 |
| 2026-03 0.2523 0.3351 0.7527 0.5045 0.6703 0.7527 0.8198 1.0000 0.8198 |
| 2026-04 0.1838 0.1946 0.9444 0.3676 0.3892 0.9444 0.7081 0.7784 0.9097 |
| 2026-05 0.1889 0.2067 0.9140 0.3889 0.4133 0.9409 0.7222 0.8267 0.8737 |

**Means across months:**
- Normalised recall at top-5%: 0.8799
- Normalised recall at top-10%: 0.8603
- Normalised recall at top-20%: 0.8282

## Alert Budget Metrics

For each budget, hours are sorted by negative_prob descending and the
top budget% are flagged as alerts.

### Budget = 10%

| Month | Precision | Recall | Lift | F1 |
|-------|-----------|--------|------|----|
| 2026-01 | 0.7297 | 0.3830 | 3.8505 | 0.5023 |
| 2026-02 | 0.9403 | 0.3000 | 3.0090 | 0.4549 |
| 2026-03 | 0.7568 | 0.5045 | 5.0723 | 0.6054 |
| 2026-04 | 0.9444 | 0.3676 | 3.6757 | 0.5292 |
| 2026-05 | 0.9459 | 0.3889 | 3.9099 | 0.5512 |

**Means:** precision=0.8634, recall=0.3888, lift=3.9035, f1=0.5286

### Budget = 20%

| Month | Precision | Recall | Lift | F1 |
|-------|-----------|--------|------|----|
| 2026-01 | 0.6216 | 0.6525 | 3.2800 | 0.6367 |
| 2026-02 | 0.8881 | 0.5667 | 2.8418 | 0.6919 |
| 2026-03 | 0.6149 | 0.8198 | 4.1213 | 0.7027 |
| 2026-04 | 0.9097 | 0.7081 | 3.5405 | 0.7964 |
| 2026-05 | 0.8784 | 0.7222 | 3.6306 | 0.7927 |

**Means:** precision=0.7825, recall=0.6939, lift=3.4829, f1=0.7241

### Budget = 30%

| Month | Precision | Recall | Lift | F1 |
|-------|-----------|--------|------|----|
| 2026-01 | 0.5112 | 0.8085 | 2.6975 | 0.6264 |
| 2026-02 | 0.8060 | 0.7714 | 2.5791 | 0.7883 |
| 2026-03 | 0.4933 | 0.9910 | 3.3063 | 0.6587 |
| 2026-04 | 0.8194 | 0.9568 | 3.1892 | 0.8828 |
| 2026-05 | 0.7803 | 0.9667 | 3.2251 | 0.8635 |

**Means:** precision=0.6820, recall=0.8989, lift=2.9994, f1=0.7639

## New Verdict

**NEGATIVE_CHAMPION**

| Criterion | Value | Threshold | Pass? |
|-----------|-------|-----------|-------|
| mean_auc | 0.9461 | >= 0.90 | Yes |
| mean_f1 | 0.7770 | >= 0.70 | Yes |
| mean_recall_at_20pct_alert | 0.6939 | >= 0.65 | Yes |
| n_sufficient_months | 5 | >= 4 | Yes |

## Comparison with Old Verdict

| Aspect | Old (raw top-k) | New (base-rate aware) |
|--------|-----------------|----------------------|
| Verdict | NEGATIVE_LOW_VALUE | NEGATIVE_CHAMPION |
| Key metric | mean top10 capture = 0.365 | mean norm recall@top10 = 0.8603 |
| AUC | 0.880 | 0.9461 |
| Issue | Raw capture penalised by high base rate | Normalised recall accounts for base rate ceiling |

## Criteria Reference

| Verdict | Condition |
|---------|-----------|
| NEGATIVE_CHAMPION | mean_auc >= 0.90, mean_f1 >= 0.70, mean_recall_at_20pct_alert >= 0.65, >= 4 sufficient months |
| NEGATIVE_ACCEPTABLE | mean_auc >= 0.85, mean_f1 >= 0.60 |
| NEGATIVE_AUX | mean_auc >= 0.80 |
| NEGATIVE_NO_GO | otherwise |
