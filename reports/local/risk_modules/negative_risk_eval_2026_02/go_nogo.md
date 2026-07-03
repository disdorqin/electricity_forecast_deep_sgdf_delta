# NegativeRisk Evaluation: NEGATIVE_LOW_VALUE

## Verdict: NEGATIVE_LOW_VALUE

**Reason**: Max lift across top-k = 3.52 >= 1.3

## DA Anchor sMAPE (canonical floor50)

- DA anchor sMAPE floor50: 26.6976

## Classification Metrics

| label | n_valid | n_positive | positive_rate | precision | recall | f1 | roc_auc | pr_auc | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| negative_label | 672 | 210.0000 | 0.3125 | 0.8136 | 0.6857 | 0.7442 | 0.9431 | 0.8767 | nan |
| deep_negative_label | 672 | nan | 0.0000 | nan | nan | nan | nan | nan | single_class |
| relative_down_label | 672 | 37.0000 | 0.0551 | 0.0000 | 0.0000 | 0.0000 | 0.8334 | 0.1513 | nan |

## Top-k Capture, Lift, Alert Rate

| label | top_k_pct | k | n_top_k | n_positive_in_top_k | capture_rate | lift | alert_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| negative_label | 1 | 6 | 6 | 6 | 1.0000 | 3.2000 | 0.0089 |
| negative_label | 3 | 20 | 20 | 20 | 1.0000 | 3.2000 | 0.0298 |
| negative_label | 5 | 33 | 33 | 33 | 1.0000 | 3.2000 | 0.0491 |
| negative_label | 10 | 67 | 67 | 63 | 0.9403 | 3.0090 | 0.0997 |
| negative_label | 20 | 134 | 134 | 119 | 0.8881 | 2.8418 | 0.1994 |
| deep_negative_label | 1 | 6 | 6 | 0 | 0.0000 | nan | 0.0089 |
| deep_negative_label | 3 | 20 | 20 | 0 | 0.0000 | nan | 0.0298 |
| deep_negative_label | 5 | 33 | 33 | 0 | 0.0000 | nan | 0.0491 |
| deep_negative_label | 10 | 67 | 67 | 0 | 0.0000 | nan | 0.0997 |
| deep_negative_label | 20 | 134 | 134 | 0 | 0.0000 | nan | 0.1994 |
| relative_down_label | 1 | 6 | 6 | 0 | 0.0000 | 0.0000 | 0.0089 |
| relative_down_label | 3 | 20 | 20 | 1 | 0.0500 | 0.9081 | 0.0298 |
| relative_down_label | 5 | 33 | 33 | 2 | 0.0606 | 1.1007 | 0.0491 |
| relative_down_label | 10 | 67 | 67 | 11 | 0.1642 | 2.9818 | 0.0997 |
| relative_down_label | 20 | 134 | 134 | 26 | 0.1940 | 3.5240 | 0.1994 |

## Summary

- Total predictions: 672
- Verdict: NEGATIVE_LOW_VALUE
