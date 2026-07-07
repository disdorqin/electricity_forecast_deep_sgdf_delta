# P2.9 ML Gate Report

## Models

| Gate | Type | Overall |
|------|------|-------:|
| LogisticSelector | LogisticRegression (C=0.1, balanced) | 19.91 |
| LightweightTree | DecisionTree (max_depth=4) | 19.31 |

## Time Split (train 2025 → test 2026)

| Model | Test DA | Test SGD | Gate | Beats Baseline? |
|-------|:------:|:--------:|:----:|:--------------:|
| Logistic | 23.58 | 23.99 | 23.96 | NO |
| Tree | 23.58 | 23.99 | 23.58 | NO |

## LOMO Summary
Logistic beats baselines on 
