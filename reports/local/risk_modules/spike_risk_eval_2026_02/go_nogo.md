# SpikeRisk Evaluation: SPIKE_LOW_VALUE

## Verdict: SPIKE_LOW_VALUE

**Reason**: Max lift across top-k = 10.67 >= 1.3

## DA Anchor sMAPE (canonical floor50)

- DA anchor sMAPE floor50: 26.6976

## Classification Metrics

| label | n_valid | n_positive | positive_rate | precision | recall | f1 | roc_auc | pr_auc | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spike_label | 672 | 21.0000 | 0.0312 | 0.3846 | 0.2381 | 0.2941 | 0.9190 | 0.2636 | nan |
| extreme_spike_label | 672 | nan | 0.0000 | nan | nan | nan | nan | nan | single_class |
| relative_spike_label | 672 | 33.0000 | 0.0491 | 0.0000 | 0.0000 | 0.0000 | 0.7901 | 0.1611 | nan |

## Top-k Capture, Lift, Alert Rate

| label | top_k_pct | k | n_top_k | n_positive_in_top_k | capture_rate | lift | alert_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| spike_label | 1 | 6 | 6 | 2 | 0.3333 | 10.6667 | 0.0089 |
| spike_label | 3 | 20 | 20 | 6 | 0.3000 | 9.6000 | 0.0298 |
| spike_label | 5 | 33 | 33 | 10 | 0.3030 | 9.6970 | 0.0491 |
| spike_label | 10 | 67 | 67 | 11 | 0.1642 | 5.2537 | 0.0997 |
| spike_label | 20 | 134 | 134 | 20 | 0.1493 | 4.7761 | 0.1994 |
| extreme_spike_label | 1 | 6 | 6 | 0 | 0.0000 | nan | 0.0089 |
| extreme_spike_label | 3 | 20 | 20 | 0 | 0.0000 | nan | 0.0298 |
| extreme_spike_label | 5 | 33 | 33 | 0 | 0.0000 | nan | 0.0491 |
| extreme_spike_label | 10 | 67 | 67 | 0 | 0.0000 | nan | 0.0997 |
| extreme_spike_label | 20 | 134 | 134 | 0 | 0.0000 | nan | 0.1994 |
| relative_spike_label | 1 | 6 | 6 | 1 | 0.1667 | 3.3939 | 0.0089 |
| relative_spike_label | 3 | 20 | 20 | 4 | 0.2000 | 4.0727 | 0.0298 |
| relative_spike_label | 5 | 33 | 33 | 5 | 0.1515 | 3.0854 | 0.0491 |
| relative_spike_label | 10 | 67 | 67 | 10 | 0.1493 | 3.0393 | 0.0997 |
| relative_spike_label | 20 | 134 | 134 | 21 | 0.1567 | 3.1913 | 0.1994 |

## Summary

- Total predictions: 672
- Verdict: SPIKE_LOW_VALUE
