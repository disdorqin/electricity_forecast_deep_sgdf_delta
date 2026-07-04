# DeepRT-SOTA Champion Seed (2026-02)

**Status**: Champion seed from 2026-02 experiment (Group 4). **NOT final SOTA** - requires multi-month backtest validation.

## Configuration

```yaml
model_profile: deep_rt_tcn
target_granularity: day
target_mode: residual_to_da
seq_len_days: 7
risk_features: off
loss: huber  # Note: train_working.py uses SmoothL1Loss
epochs: 60
batch_size: 32
lr: 0.001
hidden_dim: 128
num_layers: 3
dropout: 0.1
output_dim: 24
```

## Results (2026-02)

```
sMAPE_floor50: 17.26
MAE: 82.23
RMSE: 119.52
pred_std: 31.95
target_std: 113.73
pred_std / target_std: 0.28
beats_naive_baseline: true (63.42)
beats_da_anchor: true (26.69)
test_samples: 28
test_days: 28
```

## Diagnostics (Preliminary)

```
da_anchor_std: (need to compute)
rt_actual_std: 113.73
residual_true_std: (need to compute)
residual_pred_std: 31.95
final_rt_pred_std: (need to compute)
corr_da_rt: 0.85
corr_residual_true_pred: (need to compute)
corr_final_pred_true: (need to compute)
```

**WARNING**: `pred_std / target_std = 0.28` suggests possible model collapse. Need full diagnostics from Track B.

## Legality Verification

```
da_anchor == rt_actual? False
Correlation (da_anchor, rt_actual): 0.85
Example: 368.05 vs 328.082 (clearly different)
Oracle baseline: NO
```

## Artifacts

- `config.yaml` - Model configuration
- `metrics_summary.json` - Evaluation metrics
- `predictions.csv` - Predictions for 2026-02
- `diagnostics.json` - Diagnostic metrics (after Track B)
- `feature_manifest.json` - Feature list
- `train_data_audit.json` - Data audit (after Track C)

## Next Steps

1. **Track B**: Add full diagnostics to training script
2. **Track D**: Run multi-month walk-forward backtest (2026-01 to 2026-05)
3. **Track E**: Expand residual model configurations (12 experiments)
4. **Track F**: Bucket/period evaluation
5. **Track G**: Export model pack (if SOTA_CANDIDATE)
6. **Track H**: Final report

## Warning

**This is a 2026-02 champion seed, not final SOTA.**

Multi-month backtest required before announcing SOTA. Do NOT use only 2026-02 for final evaluation.
