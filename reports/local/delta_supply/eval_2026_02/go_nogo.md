# DeltaSupply Evaluation: NO-GO

## Verdict: NO-GO

**Reason**: No improvement (best=0.0000)

## Classification Metrics

| label | n_valid | n_positive | positive_rate | precision | recall | f1 | roc_auc | pr_auc | top_k_capture_rate_5pct | top_k_capture_rate_10pct | top_k_capture_rate_20pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| upward_deviation_label | 672 | 85 | 0.1265 | 0.2857 | 0.0235 | 0.0435 | 0.7234 | 0.2550 | 0.2424 | 0.2239 | 0.2687 |
| downward_deviation_label | 672 | 104 | 0.1548 | 0.4091 | 0.1731 | 0.2432 | 0.8210 | 0.3709 | 0.3636 | 0.4030 | 0.4179 |
| large_abs_deviation_label | 672 | 116 | 0.1726 | 0.5526 | 0.1810 | 0.2727 | 0.7954 | 0.4715 | 0.6061 | 0.5224 | 0.4478 |

## Regression Metrics

{
  "n_valid": 672,
  "mae": 83.8734023795683,
  "rmse": 122.2346866789573,
  "magnitude_corr": 0.06693247753552711
}

## Correction Simulation

| correction_weight | da_anchor_smape | corrected_smape | improvement_pp | normal_bucket_da | normal_bucket_corrected | normal_bucket_delta | negative_bucket_da | negative_bucket_corrected | negative_bucket_delta | spike_bucket_da | spike_bucket_corrected | spike_bucket_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 0.4735 | 0.4735 | 0.0000 | 0.3578 | 0.3578 | 0.0000 | 0.7397 | 0.7397 | 0.0000 | 0.2395 | 0.2395 | 0.0000 |
| 0.1000 | 0.4735 | 0.4756 | -0.0021 | 0.3578 | 0.3623 | 0.0045 | 0.7397 | 0.7373 | -0.0025 | 0.2395 | 0.2377 | -0.0018 |
| 0.2000 | 0.4735 | 0.4794 | -0.0060 | 0.3578 | 0.3679 | 0.0101 | 0.7397 | 0.7381 | -0.0016 | 0.2395 | 0.2359 | -0.0036 |
| 0.3000 | 0.4735 | 0.4841 | -0.0106 | 0.3578 | 0.3741 | 0.0163 | 0.7397 | 0.7399 | 0.0002 | 0.2395 | 0.2342 | -0.0053 |
| 0.5000 | 0.4735 | 0.4945 | -0.0211 | 0.3578 | 0.3882 | 0.0304 | 0.7397 | 0.7440 | 0.0043 | 0.2395 | 0.2329 | -0.0067 |
| 1.0000 | 0.4735 | 0.5257 | -0.0523 | 0.3578 | 0.4355 | 0.0777 | 0.7397 | 0.7441 | 0.0044 | 0.2395 | 0.2359 | -0.0037 |

## Summary

- DA anchor sMAPE: 0.4735
- Best corrected sMAPE: 0.4735
- Best correction weight: 0.0
- Improvement: 0.0000 (0.00pp)
