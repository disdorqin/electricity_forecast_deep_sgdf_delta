# Phase 2: DeepSGDFDelta / TrendKnight Plan

## Objective

Upgrade the DeepSGDFDelta model from a demo-level per-hour predictor to a production-ready realtime trend model that can achieve monthly average sMAPE_floor50 < 15 on the Shandong electricity spot market.

## Scope

**In scope:**
- V2 day-level 24h decoder architecture
- SGDFNet residual/blend champion search
- Cutoff-safe walk-forward backtesting
- Output contract for downstream modules (spike, negative, fusion)
- Quality gates and CI

**Out of scope:**
- Spike prediction (separate module)
- Negative price correction (separate module)
- Modifying SGDFNet's frozen baseline
- Deep hyperparameter tuning beyond Phase 2 defaults

## Architecture Decisions

### V2 Day-Level Decoder

The V1 model predicts one hour at a time with a 7-day lookback window. This ignores intraday structure: each prediction is independent, so the model cannot learn smooth intraday curves.

V2 takes a full business_day's 24 hours of features as input and outputs 24 delta predictions simultaneously. This allows:
- Smoothness loss to penalize jagged intraday transitions
- Hour embeddings to learn time-of-day patterns
- Segment heads to specialize per time-of-day block

### Backbone Selection

Three backbones are supported:
- **TCN**: Causal convolutions, fast, ~38k params. Good for local pattern capture.
- **GRU**: Sequential recurrence, ~63k params. Good for temporal dependencies.
- **Transformer-tiny**: Multi-head attention, ~80k params. Good for long-range dependencies within 24h.

All three are well under the 300k parameter budget.

### Blend Strategy

Three blend modes allow combining the deep model with SGDFNet:
- `deep_only`: Trust the deep model entirely
- `sgdfnet_blend`: Weighted average, weight learned from D-30 to D-1 validation
- `sgdfnet_residual`: Deep model corrects SGDFNet's residuals

The residual mode is theoretically most powerful: SGDFNet captures the bulk signal, and the deep model learns what SGDFNet misses.

### Loss Function

The combined loss targets both accuracy and smoothness:
- 55% sMAPE_floor50: business-aligned primary metric
- 25% delta MAE: stable gradient for delta prediction
- 10% period 9-16 weighted: extra attention to solar-volatile hours
- 10% smoothness: penalize hour-to-hour jumps (V2 only, where it applies to 24h sequence)

### Cutoff Safety

All features are computed under Protocol B / D15 cutoff:
- On decision day D at 15:00, actual RT prices after 15:00 are replaced with DA prices
- All history features use lag-24 shift (previous day's same hour)
- Blend weights are learned only from D-30 to D-1 (never future data)

## Development Sequence

| Step | Task | Deliverables |
|------|------|-------------|
| 1 | Repository audit | Verify all imports, run existing tests |
| 2 | SGDFNet bridge | `sgdfnet_bridge.py`, `check_sgdfnet_bridge.py`, tests |
| 3 | Baseline reproduction | `p0_reproduce_sgdfnet_baseline.py` with full metrics |
| 4 | V2 model | `model_v2.py`, `dataset_v2.py`, `train_v2.py`, `predict_v2.py` |
| 5 | CPU smoke test | `--fast-dev-run` mode |
| 6 | Monthly backtest | `run_phase2_monthly_backtest.py` |
| 7 | Champion search | `search_phase2_champion.py` |
| 8 | Output contract | `output_contract.py`, tests |
| 9 | CI & tests | GitHub Actions, 7 new test files |
| 10 | Documentation | README update, this plan document |

## Evaluation Metrics

### Primary
- **overall_sMAPE_floor50**: sMAPE with floor-50 capping across all hours
- **monthly_avg_sMAPE_floor50**: average of per-month sMAPE values

### Breakdown
- **1_8_sMAPE_floor50**: night/early morning hours
- **9_16_sMAPE_floor50**: solar-volatile hours (historically hardest)
- **17_24_sMAPE_floor50**: evening hours

### Bucket
- **normal_bucket**: 0 <= price <= 500
- **high_price_bucket**: price > 500 (or top 10% quantile)
- **negative_bucket**: price < 0

### Secondary
- **MAE**: mean absolute error
- **coverage_rate**: fraction of target hours with valid predictions
- **rows_total / rows_missing**: data completeness

## Go/No-Go Thresholds

| Verdict | Condition |
|---------|-----------|
| PASS | Monthly avg sMAPE_floor50 < 15.0 |
| SOFT_PASS | Overall sMAPE_floor50 <= 15.8 (awaiting spike/negative module fusion) |
| BASELINE_PASS | Not worse than SGDFNet baseline 16.5902, and 9_16 improved |
| NO-GO | Worse than baseline or leakage detected |

## Known Risks

1. **9-16 segment weakness**: SGDFNet already struggles here (21.19 sMAPE). The deep model may or may not improve this.
2. **Data freshness**: The data file may lag behind the current date. Training requires recent data.
3. **Training time**: V2 day-level model trains faster than V1 per-hour (24 samples per forward pass vs 1), but walk-forward over multiple months still takes significant compute.
4. **Overfitting to normal trend**: The model intentionally underfits extreme prices. This is by design -- spike/negative modules handle those.

## Expected Outcomes

Best case: V2 + sgdfnet_residual achieves SOFT_PASS or PASS on a representative month.
Realistic case: V2 + blend achieves BASELINE_PASS, with 9_16 segment showing measurable improvement over SGDFNet alone.
Fallback: V1 deep_only achieves BASELINE_PASS, establishing the architecture for future iteration.
