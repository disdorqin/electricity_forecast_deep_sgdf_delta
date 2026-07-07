# P2.5 Lightweight Gating Model Report

## Model
- Algorithm: LogisticRegression (L2, C=1.0)
- Training: hour-level from 2208 samples
- Input features (9): da, sgd, tfm, hour_norm, dow_norm, gap_norm, disagreement, neg_flag, spike_flag
- Output: P(timesfm better) → blend weight w_sgd = 1 - P(tfm_better)
- Scaler: StandardScaler
- Training time: sub-second

## Performance
- Accuracy (which model is better): 0.624
- Overall sMAPE: 23.79

## Coefficients
| Feature | Coefficient |
|---------|:----------:|
| da | 0.2508 |
| sgd | 0.1403 |
| tfm | -0.2869 |
| hour_norm | 0.0575 |
| dow_norm | 0.1289 |
| gap_norm | 0.6931 |
| disagreement | -0.5276 |
| neg_flag | -0.0750 |
| spike_flag | -0.2966 |

## Decision
sgdfnet weight = 1 - P(timesfm_better)

Coefficient interpretation:
- Positive coefficient → more weight to TimesFM when feature is high
- Negative coefficient → more weight to SGDFNet when feature is high

## Production Feasibility
- CPU inference: ✅ ~microseconds per hour
- No GPU dependency: ✅
- Retraining: can be periodic (weekly/monthly)
