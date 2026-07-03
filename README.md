# DeepSGDFDelta / TrendKnight

Lightweight deep learning model for realtime electricity price trend prediction in the Shandong electricity spot market. This is a side branch of the main forecasting project, focusing exclusively on normal trend prediction using SGDFNet's cutoff-safe protocol and feature engineering.

## Project Goal

Predict the main body trend of realtime electricity prices. Spike prediction and negative-price correction are handled by separate modules. This model's job is to deliver a stable, accurate, cutoff-safe trend prediction that downstream modules can build upon.

**Target**: monthly average realtime sMAPE_floor50 < 15.0

## Relationship with SGDFNet

DeepSGDFDelta reuses SGDFNet's core ideas without modifying SGDFNet's code:

- **cutoff-safe visible frame**: Protocol B / D15 cutoff / walk-forward
- **delta prediction**: target = realtime_price - dayahead_price; rt_pred = da_anchor + delta_pred
- **business_day + hour_business alignment**: 00:00 maps to previous day hour 24
- **forecast-side features**: all features available before D15 cutoff
- **historical lagged actuals**: only lag-24 shifted actual values as features
- **segment split**: 1_8 / 9_16 / 17_24

The SGDFNet bridge (`models/deep_sgdf_delta/sgdfnet_bridge.py`) locates the SGDFNet source directory and imports its data contract, metrics, and protocol functions. SGDFNet code is never copied into this repo.

## Boundary with Spike / Negative Modules

This model outputs `trend_pred` (the main trend). It does NOT:

- Correct extreme spike prices (|price| > 500)
- Correct negative prices (price < 0)
- Apply any post-hoc extreme-value adjustments

Spike and negative modules consume `trend_pred` and apply their own corrections. The `output_contract.py` defines the standard interface (`trend_pred`, `residual_for_spike_module`, `residual_for_negative_module`) for clean handoff.

## Architecture

### V1: Per-Hour Model (baseline)

- Input: 7-day sequence window of per-hour features
- Backbone: TCN or GRU (configurable)
- Heads: Segment-conditioned (1_8 / 9_16 / 17_24) + global residual
- Output: single delta_pred per sample
- Parameters: < 200k

### V2: Day-Level 24h Decoder (Phase 2)

- Input: one full business_day with 24 hourly feature rows
- Backbone: TCN / GRU / small TransformerEncoder
- Hour + segment embeddings
- 24-hour decoder head: outputs delta_pred_24 of shape [batch, 24]
- Optional residual correction against SGDFNet baseline
- Parameters: < 300k

V2's main advantage: learns full intraday curve shape, making smoothness loss effective across the 24h sequence.

## Directory Structure

```
models/deep_sgdf_delta/
  sgdfnet_bridge.py  - SGDFNet dependency finder
  dataset.py         - V1 sequence dataset
  model.py           - V1 architecture (per-hour)
  losses.py          - Combined loss functions
  train.py           - V1 training
  predict.py         - V1 prediction + blend
  evaluate.py        - Metrics + go/no-go
  metrics.py         - sMAPE_floor50 and KPIs
  output_contract.py - Standard output schema
  config.yaml        - Default configuration
  dataset_v2.py      - V2 day-level dataset
  model_v2.py        - V2 24h decoder architecture
  train_v2.py        - V2 training
  predict_v2.py      - V2 prediction + blend
  model.py           - V1 model (preserved)

scripts/
  train_phase2_trendknight.py     - Unified training (7 profiles)
  evaluate_phase2_trendknight.py  - Evaluation + go/no-go
  run_phase2_monthly_backtest.py  - Monthly walk-forward backtest
  search_phase2_champion.py       - Champion search across all candidates
  p0_reproduce_sgdfnet_baseline.py - SGDFNet baseline reproduction
  check_sgdfnet_bridge.py         - Bridge diagnostic
  train_deep_sgdf_delta.py        - V1 training entry
  predict_deep_sgdf_delta.py      - V1 prediction entry
  evaluate_deep_sgdf_delta.py     - V1 evaluation entry

tests/
  test_deep_sgdf_delta.py         - Core tests (19 tests)
  test_sgdfnet_bridge.py          - Bridge tests
  test_output_contract.py         - Output schema tests
  test_business_day_hour24.py     - Alignment tests
  test_no_leakage_cutoff.py       - Cutoff safety tests
  test_v2_day_decoder_shapes.py   - V2 shape tests
  test_smoothness_loss_v2.py      - Loss tests
  test_blend_weight_window.py     - Blend window tests
  test_go_nogo.py                 - Verdict tests
```

## How to Reproduce SGDFNet Baseline

```bash
python scripts/p0_reproduce_sgdfnet_baseline.py \
  --sgdfnet-root ../electricity_forecast_model2.0_exp/SGDFNet \
  --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.xlsx \
  --start-date 2026-01-01 \
  --end-date 2026-05-12
```

Output: `reports/local/phase2/baseline_sgdfnet/` with predictions, metrics, and go_nogo report.

## How to Train DeepSGDFDelta

### V1 (per-hour, lightweight)

```bash
python scripts/train_phase2_trendknight.py \
  --profile v1_hourly_tcn \
  --start-date 2026-01-01 --end-date 2026-03-31 \
  --sgdfnet-root ../electricity_forecast_model2.0_exp/SGDFNet \
  --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.xlsx
```

### V2 (day-level 24h decoder)

```bash
python scripts/train_phase2_trendknight.py \
  --profile v2_day_tcn \
  --start-date 2026-01-01 --end-date 2026-03-31 \
  --sgdfnet-root ../electricity_forecast_model2.0_exp/SGDFNet \
  --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.xlsx
```

### Fast dev run (smoke test)

```bash
python scripts/train_phase2_trendknight.py \
  --profile v2_day_tcn --fast-dev-run \
  --sgdfnet-root ../electricity_forecast_model2.0_exp/SGDFNet \
  --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.xlsx
```

## How to Run Monthly Backtest

```bash
python scripts/run_phase2_monthly_backtest.py \
  --start-date 2026-01-01 --end-date 2026-03-31 \
  --profile v2_day_tcn \
  --sgdfnet-root ../electricity_forecast_model2.0_exp/SGDFNet \
  --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.xlsx \
  --device cuda
```

## How to Run Champion Search

```bash
python scripts/search_phase2_champion.py \
  --start-date 2026-01-01 --end-date 2026-03-31 \
  --sgdfnet-root ../electricity_forecast_model2.0_exp/SGDFNet \
  --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.xlsx \
  --device cuda
```

Output: `reports/local/phase2/champion_search/` with leaderboard, champion predictions, and go_nogo report.

## Go/No-Go Criteria

| Verdict | Condition |
|---------|-----------|
| PASS | Monthly average realtime sMAPE_floor50 < 15.0 |
| SOFT PASS | Overall realtime sMAPE_floor50 <= 15.8 |
| BASELINE PASS | Not worse than SGDFNet corrected baseline (16.5902) and 9_16 improved |
| NO-GO | Worse than SGDFNet baseline or leakage risk detected |

## Output File Format

`predictions.csv` contains these standard columns:

```
business_day, hour_business, period, ds, da_anchor, y_true,
deep_delta_pred, deep_rt_pred, sgdfnet_pred, blend_pred,
trend_pred, trend_model_name, trend_confidence,
normal_trend_flag, high_price_bucket_flag, negative_bucket_flag,
residual_for_spike_module, residual_for_negative_module
```

Eval-only columns (`high_price_bucket_flag`, `negative_bucket_flag`, `residual_for_*`) require `y_true` and are stripped for online prediction.

## Leakage Risk

This model strictly enforces cutoff safety:

- No post-D15 actual realtime prices are used as features
- No post-D15 actual-side load/renewable values are used
- All history features use lag-24 shift
- Blend weights are learned only from D-30 to D-1 validation window
- Spike/negative labels from y_true never enter the training loss weights

## Profiles

| Profile | Version | Backbone | Blend |
|---------|---------|----------|-------|
| v1_hourly_tcn | V1 | TCN | deep_only |
| v1_hourly_gru | V1 | GRU | deep_only |
| v2_day_tcn | V2 | TCN | deep_only |
| v2_day_gru | V2 | GRU | deep_only |
| v2_day_transformer_tiny | V2 | Transformer | deep_only |
| v2_residual_sgdfnet | V2 | TCN | sgdfnet_residual |
| v2_blend_sgdfnet | V2 | TCN | sgdfnet_blend |
