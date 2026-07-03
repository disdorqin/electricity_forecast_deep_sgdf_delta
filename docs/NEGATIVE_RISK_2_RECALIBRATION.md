# NegativeRisk-2 Recalibration Report

## Purpose

Base-rate aware recalibration of the NegativeRisk module selection verdict.

The original NegativeRisk backtest assigned verdict **NEGATIVE_LOW_VALUE**
because the mean top-10% capture rate (0.365) fell below the 0.70 champion
threshold. However, this raw capture metric is unfairly penalised when
negative price events have a high base rate (~15-30%). When the positive
rate is high, even a perfect ranker cannot achieve 70% recall at top-10%
because the theoretical ceiling is `min(1.0, 0.10 / positive_rate)`.

This recalibration replaces raw top-k capture with **normalised recall**
(actual recall divided by theoretical maximum) and **alert-budget metrics**
(precision, recall, lift, F1 at operationally relevant alert budgets).

## Old Verdict

| Aspect | Value |
|--------|-------|
| Verdict | NEGATIVE_LOW_VALUE |
| Mean AUC | 0.880 |
| Mean top-10% capture | 0.365 |
| Sufficient months | 5 |

The old verdict was NEGATIVE_LOW_VALUE because top10 capture = 0.365 < 0.70.

## Base-Rate Analysis

TODO: run recalibration script

| Month | N Total | N Positive | Positive Rate |
|-------|---------|------------|---------------|
| TODO | TODO | TODO | TODO |

## Normalised Top-k Recall

Normalised recall = actual recall / theoretical maximum recall at that budget.
A value of 1.0 means the ranker captures every possible positive within the
budget given the base rate.

TODO: run recalibration script

| Month | Recall@top5 | Max@top5 | Norm@top5 | Recall@top10 | Max@top10 | Norm@top10 | Recall@top20 | Max@top20 | Norm@top20 |
|-------|-------------|----------|-----------|--------------|-----------|------------|--------------|-----------|------------|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**Mean normalised recall:**
- top-5%: TODO
- top-10%: TODO
- top-20%: TODO

## Alert Budget Metrics

For each budget, hours are sorted by `negative_prob` descending and the top
budget% are flagged as alerts.

### Budget = 10%

TODO: run recalibration script

| Month | Precision | Recall | Lift | F1 |
|-------|-----------|--------|------|----|
| TODO | TODO | TODO | TODO | TODO |

### Budget = 20%

TODO: run recalibration script

| Month | Precision | Recall | Lift | F1 |
|-------|-----------|--------|------|----|
| TODO | TODO | TODO | TODO | TODO |

### Budget = 30%

TODO: run recalibration script

| Month | Precision | Recall | Lift | F1 |
|-------|-----------|--------|------|----|
| TODO | TODO | TODO | TODO | TODO |

## New Verdict

TODO: run recalibration script

| Criterion | Value | Threshold | Pass? |
|-----------|-------|-----------|-------|
| mean_auc | TODO | >= 0.90 | TODO |
| mean_f1 | TODO | >= 0.70 | TODO |
| mean_recall_at_20pct_alert | TODO | >= 0.65 | TODO |
| n_sufficient_months | TODO | >= 4 | TODO |

**New verdict: TODO**

## Comparison with Old Verdict

| Aspect | Old (raw top-k) | New (base-rate aware) |
|--------|-----------------|----------------------|
| Verdict | NEGATIVE_LOW_VALUE | TODO |
| Key metric | mean top10 capture = 0.365 | mean norm recall@top10 = TODO |
| AUC | 0.880 | TODO |
| Issue | Raw capture penalised by high base rate | Normalised recall accounts for base rate ceiling |

## New Champion Criteria

| Verdict | Condition |
|---------|-----------|
| NEGATIVE_CHAMPION | mean_auc >= 0.90, mean_f1 >= 0.70, mean_recall_at_20pct_alert >= 0.65, >= 4 sufficient months |
| NEGATIVE_ACCEPTABLE | mean_auc >= 0.85, mean_f1 >= 0.60 |
| NEGATIVE_AUX | mean_auc >= 0.80 |
| NEGATIVE_NO_GO | otherwise |

## How to Populate

Run the recalibration script to fill in all TODO values:

```bash
python scripts/recalibrate_negative_risk_selection.py \
    --backtest-root reports/local/risk_modules/negative_risk_backtest_2026_01_05 \
    --out-dir reports/local/risk_modules/negative_recalibration
```

Outputs:
- `reports/local/risk_modules/negative_recalibration/negative_recalibration_summary.json`
- `reports/local/risk_modules/negative_recalibration/negative_recalibration_monthly.csv`
- `reports/local/risk_modules/negative_recalibration/negative_recalibration_report.md`
