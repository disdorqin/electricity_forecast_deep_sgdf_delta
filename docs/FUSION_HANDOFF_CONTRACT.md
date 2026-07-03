# Fusion Handoff Contract

> TrendKnight-X -> Fusion -> Final Price Prediction

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TrendKnight-X (Deep Trend Model)                  │
│                                                                      │
│  features_24h ──► Backbone (TCN/GRU) ──► Multiscale Decomposition   │
│                                              │                       │
│  segment_id ──► PeriodBranch ────────────────┤                       │
│                                              │                       │
│  teacher_preds ──► TeacherFusionGate ────────┤                       │
│                                              ▼                       │
│                                     DayDecoderHead                   │
│                                     ConfidenceHead                   │
│                                     ShockSensitivityHead             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           │  fusion_pack.csv
                           │  (FUSION_COLUMNS schema)
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Fusion Layer                                  │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐       │
│  │  日前线模块   │  │  CatBoost     │  │  零样本模型           │       │
│  │  (DA anchor) │  │  (residual)   │  │  (zero-shot)         │       │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘       │
│         │                 │                      │                   │
│         ▼                 ▼                      ▼                   │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │              Trend Fusion (trend_pred)                    │       │
│  │   final_trend = w1*da + w2*trend_pred + w3*zero_shot     │       │
│  └──────────────────────────┬───────────────────────────────┘       │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────┐  ┌──────────────────┐                             │
│  │  产差模块     │  │  负价模块         │                             │
│  │  (spread)    │  │  (negative price) │                             │
│  └──────┬───────┘  └────────┬─────────┘                             │
│         │                   │                                        │
│         ▼                   ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │           Final Price = trend + spread + neg_adj         │       │
│  └──────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

## Fusion Pack Schema

### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `business_day` | Timestamp | Trading day (date only, normalized to midnight) |
| `hour_business` | int | Business hour (1-24); hour 0 maps to previous day hour 24 |
| `period` | str | Segment label: `"1_8"`, `"9_16"`, or `"17_24"` |
| `ds` | Timestamp | Wall-clock timestamp for this hour |
| `model_name` | str | Deep model identifier (e.g., `"trendknight_x"`) |
| `trend_pred` | float | Final realtime price prediction from deep model |
| `trend_delta_pred` | float | Predicted delta (rt - da) from deep model |
| `trend_confidence` | float | Confidence score in [0.1, 0.95]; higher = more reliable |
| `shock_sensitivity` | float | Shock sensitivity in [0, 1]; higher = more volatile |
| `teacher_used` | str | Which teacher(s) were fused: `"none"`, `"sgdfnet"`, `"sgdfnet+rt916"`, etc. |
| `sgdfnet_pred` | float | SGDFNet teacher prediction (NaN if unavailable) |
| `rt916_pred` | float | RT916 teacher prediction (NaN if unavailable) |
| `timemixer_pred` | float | TimeMixer teacher prediction (NaN if unavailable) |
| `runtime_profile` | str | Profile used for inference: `"v3_fast_tcn"`, `"v3_multiscale_tcn"`, etc. |

### Eval-Only Columns (offline analysis)

| Column | Type | Description |
|--------|------|-------------|
| `y_true` | float | Actual realtime price (ground truth) |
| `residual_for_spike` | float | `y_true - trend_pred`; consumed by spike module |
| `residual_for_negative` | float | `y_true - trend_pred`; consumed by negative price module |

## How to Combine with Downstream Modules

### 1. 日前线 (Day-Ahead Anchor)

The `da_anchor` value is already embedded in `trend_pred` (since `trend_pred = da_anchor + delta_pred`). The fusion layer should:

- Use `trend_pred` directly as the deep model's realtime price estimate.
- If the fusion layer wants to blend with a separate DA forecast, use `trend_delta_pred` (the delta component only) and add it to the DA forecast:
  ```
  final_price = da_forecast_from_catboost + trend_delta_pred
  ```

### 2. CatBoost (Residual Correction)

CatBoost operates on the residual between the trend prediction and actual price:
```
catboost_input = [trend_pred, trend_confidence, shock_sensitivity, period, hour, ...]
catboost_correction = CatBoost.predict(catboost_input)
final_price = trend_pred + catboost_correction
```

The `trend_confidence` column tells CatBoost how much to trust the deep model:
- High confidence (>0.7): small correction expected
- Low confidence (<0.4): larger correction likely needed

### 3. 零样本模型 (Zero-Shot Model)

The zero-shot model provides predictions for hours/days with no training data. Fusion strategy:
```
if trend_confidence < 0.3:
    # Deep model uncertain -> rely more on zero-shot
    final_trend = 0.3 * trend_pred + 0.7 * zero_shot_pred
else:
    final_trend = 0.7 * trend_pred + 0.3 * zero_shot_pred
```

### 4. 产差模块 (Spread Module)

The spread module adjusts for the difference between node price and system price:
```
spread = SpreadModule.predict(hour, period, season, shock_sensitivity)
final_node_price = final_trend + spread
```

The `shock_sensitivity` column is critical here: high sensitivity means the spread module should prepare for larger deviations.

### 5. 负价模块 (Negative Price Module)

The negative price module handles hours where prices go below zero:
```
if shock_sensitivity > 0.7 and period == "9_16":
    neg_prob = NegativeModule.predict(features)
    if neg_prob > 0.5:
        final_price = NegativeModule.predict_price(features)
```

The `residual_for_negative` column (eval pack only) helps train the negative module by showing where the trend model systematically over-predicts in negative-price hours.

## Online vs Eval Pack

### Online Pack (Production Serving)

- **No** `y_true`, `residual_for_spike`, `residual_for_negative`
- Generated by: `export_deep_trend_for_fusion.py --predictions ... --out-dir ...`
- Contains only FUSION_COLUMNS
- Safe for real-time serving; no ground-truth leakage

### Eval Pack (Offline Analysis)

- **Includes** `y_true`, `residual_for_spike`, `residual_for_negative`
- Generated by: `export_deep_trend_for_fusion.py --predictions ... --include-eval --out-dir ...`
- Used for: metric computation, ablation studies, module training
- **Never** deploy eval pack to production

## Confidence and Shock Sensitivity Usage

### trend_confidence

The confidence score is computed by the `ConfidenceHead` in TrendKnight-X v3. It reflects:

1. **Teacher agreement**: When SGDFNet/RT916/TimeMixer agree with the deep model, confidence is high.
2. **Delta magnitude**: Large predicted deltas relative to the anchor price reduce confidence.
3. **Historical accuracy**: The head is calibrated so that confidence ~ 0.8 means ~80% of predictions in that bucket are within 10% of actual.

**Fusion usage:**
- `confidence > 0.7`: Trust the deep model; use minimal correction.
- `confidence 0.4-0.7`: Moderate trust; apply CatBoost correction.
- `confidence < 0.4`: Low trust; consider fallback to SGDFNet or zero-shot.

### shock_sensitivity

The shock sensitivity is computed by the `ShockSensitivityHead`. It reflects:

1. **Multiscale shock component**: High values when the shock decomposition are large.
2. **Teacher disagreement**: When teachers disagree strongly, the model signals that the hour is volatile.
3. **Period context**: 9_16 hours tend to have higher sensitivity due to solar/wind variability.

**Fusion usage:**
- `shock > 0.7`: Trigger spike detection pipeline; widen prediction intervals.
- `shock 0.3-0.7`: Normal volatile hour; standard fusion.
- `shock < 0.3`: Stable hour; deep model prediction is likely reliable.

## Teacher Predictions Usage in Fusion

### Why Teachers?

Teacher models (SGDFNet, RT916, TimeMixer) provide diverse perspectives:
- **SGDFNet**: Graph-based spatial correlation across nodes
- **RT916**: Rule-based heuristic tuned for 9-16 peak hours
- **TimeMixer**: Multiscale temporal decomposition

### How Teachers Are Used in TrendKnight-X

1. **Training (Residual Distillation)**: The student model learns to predict the teacher's residual, inheriting the teacher's strengths without copying its weaknesses.

2. **Inference (TeacherFusionGate)**: A learned gate decides how much to blend each teacher's prediction:
   ```
   gate_weights = softmax(W * [teacher_preds, context_features])
   fused_teacher = sum(gate_weights[i] * teacher_pred[i])
   delta_pred = backbone_output + alpha * fused_teacher
   ```

3. **Fallback**: If a teacher is unavailable at inference time, the gate automatically re-weights to the remaining teachers.

### Fusion Layer Teacher Usage

The fusion layer receives teacher predictions as separate columns (`sgdfnet_pred`, `rt916_pred`, `timemixer_pred`) and can:

1. **Direct ensemble**: Weighted average of all available teachers + deep model
2. **Disagreement detection**: Large spread between teachers signals high uncertainty
3. **Period-specific selection**: Use RT916 for 9_16, SGDFNet for 1_8/17_24
4. **Confidence calibration**: Teacher agreement -> higher confidence score

```python
# Example: disagreement-based fusion
teacher_preds = [sgdfnet_pred, rt916_pred, timemixer_pred]
teacher_preds = [p for p in teacher_preds if not np.isnan(p)]

if len(teacher_preds) >= 2:
    teacher_std = np.std(teacher_preds)
    if teacher_std > 50:  # high disagreement
        # Fall back to deep model with lower confidence
        final_pred = trend_pred * 0.8 + np.mean(teacher_preds) * 0.2
    else:
        # Teachers agree -> blend evenly
        final_pred = 0.5 * trend_pred + 0.5 * np.mean(teacher_preds)
else:
    final_pred = trend_pred
```

## File Formats

### fusion_pack.csv

```csv
business_day,hour_business,period,ds,model_name,trend_pred,trend_delta_pred,trend_confidence,shock_sensitivity,teacher_used,sgdfnet_pred,rt916_pred,timemixer_pred,runtime_profile
2026-03-15,1,1_8,2026-03-15 00:00:00,trendknight_x,285.3,12.1,0.82,0.15,sgdfnet,283.1,,310.2,v3_teacher_residual
2026-03-15,9,9_16,2026-03-15 08:00:00,trendknight_x,320.5,45.2,0.78,0.32,sgdfnet+rt916,318.0,322.1,315.7,v3_teacher_residual
```

### manifest.json

```json
{
  "timestamp": "2026-03-15T10:30:00",
  "source_predictions": "reports/local/phase3/v3_predictions.csv",
  "model_name": "trendknight_x",
  "runtime_profile": "v3_teacher_residual",
  "include_eval": false,
  "n_rows": 720,
  "fusion_columns": ["business_day", "hour_business", ...],
  "validation": {
    "is_valid": true,
    "errors": []
  },
  "teacher_status": {
    "sgdfnet": {"availability": "available", "n_predictions": 720},
    "rt916": {"availability": "unavailable"},
    "timemixer": {"availability": "available", "n_predictions": 720}
  },
  "summary": {
    "trend_pred": {"mean": 305.2, "min": 50.0, "max": 890.3, "std": 120.5},
    "trend_confidence": {"mean": 0.72, "min": 0.15, "max": 0.93},
    "shock_sensitivity": {"mean": 0.28, "min": 0.0, "max": 0.85},
    "date_range": {"start": "2026-03-01", "end": "2026-03-30", "n_days": 30},
    "teacher_usage": {"sgdfnet+rt916": 240, "sgdfnet": 480},
    "period_distribution": {"1_8": 240, "9_16": 240, "17_24": 240}
  }
}
```

## CLI Reference

```bash
# Basic export (online pack)
python scripts/export_deep_trend_for_fusion.py \
    --predictions reports/local/phase3/v3_predictions.csv \
    --model-name trendknight_x \
    --out-dir reports/local/phase3/fusion_export

# With teacher status and eval columns
python scripts/export_deep_trend_for_fusion.py \
    --predictions reports/local/phase3/v3_predictions.csv \
    --model-name trendknight_x \
    --teacher-status-json outputs/teacher_status.json \
    --runtime-profile v3_teacher_residual \
    --include-eval \
    --out-dir reports/local/phase3/fusion_export
```

## Validation Rules

The `validate_fusion_pack()` function checks:

1. All FUSION_COLUMNS are present
2. No NaN in `business_day`, `ds`, `model_name`
3. `hour_business` is in [1, 24]
4. `period` matches `hour_business` mapping
5. `trend_confidence` is in [0, 1]
6. `shock_sensitivity` is in [0, 1]
7. `teacher_used` contains valid teacher names
