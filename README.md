# DeepSGDFDelta / TrendKnight

Lightweight deep learning model for realtime electricity price trend prediction. This is a side branch of the Shandong electricity spot market forecasting project, focusing on normal trend prediction using SGDFNet's cutoff-safe protocol and feature engineering.

## Architecture

- **Input**: SGDFNet Protocol B / D15 cutoff / walk-forward feature frame
- **Target**: delta = realtime_price - dayahead_price
- **Final prediction**: rt_pred = da_anchor + delta_pred
- **Backbone**: TCN or GRU (configurable, lightweight)
- **Heads**: Segment-conditioned (1_8 / 9_16 / 17_24) + global residual
- **Parameters**: < 200k typical

## Directory Structure

```
models/deep_sgdf_delta/
  dataset.py      - Sequence dataset from SGDFNet features
  model.py        - DeepSGDFDelta architecture
  losses.py       - Combined loss (sMAPE + delta_mae + period + smoothness)
  train.py        - Training with early stopping
  predict.py      - Prediction with blend modes
  evaluate.py     - Metrics and go/no-go reporting
  metrics.py      - sMAPE_floor50 and derived KPIs
  config.yaml     - Default configuration

scripts/
  train_deep_sgdf_delta.py       - Training entry point
  predict_deep_sgdf_delta.py     - Prediction entry point
  evaluate_deep_sgdf_delta.py    - Evaluation entry point
  p0_reproduce_sgdfnet_baseline.py - P0: SGDFNet baseline reproduction

tests/
  test_deep_sgdf_delta.py        - pytest test suite
```

## Blend Modes

- `deep_only`: Use only the deep model
- `sgdfnet_blend`: Weighted average with SGDFNet (w from D-30 to D-1 validation)
- `sgdfnet_residual`: Deep model predicts residual on top of SGDFNet

## Go/No-Go Criteria

- **PASS**: Monthly avg sMAPE_floor50 < 15.0
- **SOFT PASS**: Overall sMAPE_floor50 <= 15.8
- **BASELINE PASS**: Better than SGDFNet corrected baseline (16.5902)
- **NO-GO**: Worse than SGDFNet baseline or leakage detected
