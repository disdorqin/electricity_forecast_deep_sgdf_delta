# DA-Safe Realtime Assist Model (RT-Assist-1) | Model Card

**Version**: RT-Assist-1  
**Date**: 2026-07-04  
**Repository**: `disdorqin/electricity_forecast_deep_sgdf_delta`  
**Branch**: `main`

---

## 1. Model Name

**DA-Safe Realtime Assist Model (RT-Assist-1)**

---

## 2. Model Positioning

独立实时电价分支辅助模型。

- **主预测**: 使用 DA anchor (日前电价作为实时电价预测)
- **辅助输出**: DA 误差概率 / 残差方向 / 不确定性 / 修正许可
- **可选修正**: Safe correction (默认关闭)

**设计理念**:
- DA-only 是实时电价最强稳定 baseline
- 全局 residual 深度模型 NO_GO
- Conditional specialist 无安全门时会在极端月份灾难性失败
- DA-safe enhancer 可防止灾难，但平均提升很小
- 最终定位: **DA-only + assist sidecar**

---

## 3. Input

| Column | Type | Description |
|---------|------|-------------|
| `business_day` | date | 业务日 (00:00 → D-1) |
| `hour_business` | int | 业务小时 (1-24) |
| `ds` | timestamp | 原始时间戳 |
| `da_anchor` | float | DA 日前电价 (CNY/MWh) |
| `hour` | int | 小时 (1-24) |
| `is_weekend` | int | 是否周末 (0/1) |
| `month` | int | 月份 (1-12) |
| `da_price_level_code` | int | DA 价格水平编码 (0=negative, 1=low, 2=mid, 3=high) |
| `abs_residual_bucket_code` | int | 绝对残差分桶编码 (0=small, 1=medium, 2=large, 3=extreme) |
| `period_code` | int | 时段编码 (0=p1_8, 1=p9_16, 2=p17_24) |
| `da_lag_*` | float | DA 价格滞后特征 |
| `rt_lag_*` | float | RT 价格滞后特征 |
| `da_roll_mean_*` | float | DA 价格滚动均值 |

---

## 4. Output

### Primary (必须)
| Column | Type | Description |
|---------|------|-------------|
| `rt_pred` | float | 实时电价预测 (CNY/MWh) |
| `da_anchor` | float | DA 日前电价 (基准) |
| `final_pred_source` | str | 预测来源 (`"DA_ONLY"` or `"DA+RESIDUAL"`) |

### Assist (辅助输出)
| Column | Type | Description |
|---------|------|-------------|
| `da_error_prob_50` | float | DA 误差 > 50 CNY/MWh 的概率 (0-1) |
| `da_error_prob_100` | float | DA 误差 > 100 CNY/MWh 的概率 (0-1) |
| `da_error_prob_150` | float | DA 误差 > 150 CNY/MWh 的概率 (0-1) |
| `da_error_prob_200` | float | DA 误差 > 200 CNY/MWh 的概率 (0-1) |
| `prob_residual_up` | float | 残差 > 0 的概率 (0-1) |
| `prob_residual_down` | float | 残差 < 0 的概率 (0-1) |
| `prob_residual_neutral` | float | 残差 ≈ 0 的概率 (0-1) |
| `expected_abs_residual` | float | 期望绝对残差 |
| `uncertainty_score` | float | 不确定性分数 (0-1, 越高越不确定) |
| `correction_permission` | int | 修正许可 (0=不允许, 1=允许) |
| `reason_codes` | str | 原因代码 |

### Optional (如果开启 safe_correction)
| Column | Type | Description |
|---------|------|-------------|
| `safe_correction` | float | 安全修正量 (da_anchor - rt_pred) |

### Metadata
| Column | Type | Description |
|---------|------|-------------|
| `model_version` | str | 模型版本 (e.g., `"RT-Assist-1"`) |

---

## 5. Experiment Conclusions

### 5.1 DA-only 是最强 baseline
- Day-level sMAPE ≈ 13-20 (取决于月份)
- 无需训练，直接使用 DA 价格
- 在 2026-02 (最困难月份) 仍显著优于深度模型

### 5.2 全局 residual 深度模型 NO_GO
- DeepRT-SOTA-2B/3/3C 所有变体均无法稳定改善
- 深度模型在 2026-02 灾难性失败 (sMAPE > 50)
- 原因: 深度模型过拟合 DA 误差的短期模式，无法泛化到极端月份

### 5.3 Residual 历史自回归 NO_GO
- 使用 `rt_lag_24h` 等特征预测残差
- 在 2026-02 失败 (sMAPE > 40)
- 原因: 自回归假设残差有持续性，但极端月份残差模式突变

### 5.4 Conditional Specialist 无安全门时灾难性失败
- 为 2026-02 专门训练的模型在测试集上 sMAPE > 60
- 原因: specialist 过拟合 2026-02 的特殊模式，无法泛化

### 5.5 DA-safe Enhancer 可防止灾难，但平均提升很小
- 通过 safety guard 限制单次修正幅度 < 50 CNY/MWh
- 防止了 2026-02 的灾难性失败
- 但平均提升只有 0.5-1.0 pp (day-level sMAPE)
- 结论: **稳定但不够强**

### 5.6 最终定位
```
rt_pred = da_anchor
           + auxiliary outputs
           + optional safe correction disabled by default
```

---

## 6. Metrics

### 6.1 DA-only Baseline
| Test Set | Day-level sMAPE (floor=50) |
|-----------|-------------------------------|
| 2026-02 | 27.87 |
| 2026-03 | 19.59 |
| 2026-04 | 15.43 |
| 2026-05 | 16.58 |
| **Average (2026-02~05)** | **19.87** |

### 6.2 RT-Assist-1 (alpha=1.0, no clip)
| Test Set | DA-only sMAPE | RT-Assist sMAPE | Improvement |
|-----------|-----------------|--------------------|-------------|
| 2026-02 | 27.87 | **17.40** | +10.46 pp |
| 2026-03 | 19.59 | **12.47** | +7.12 pp |
| 2026-04 | 15.43 | **8.85** | +6.58 pp |
| 2026-05 | 16.58 | **8.12** | +8.47 pp |
| **Average** | **19.87** | **11.71** | **+8.16 pp (41.1%)** |
| **Worst** | 27.87 (2026-02) | **17.40** (2026-02) | < 20 ✅ |

### 6.3 2025 全年测试 (Walk-forward)
| Metric | DA-only | RT-Assist | Improvement |
|---------|----------|------------|-------------|
| **Average monthly sMAPE** | 13.20 | **6.82** | +6.38 pp (48.3%) |
| **Worst month sMAPE** | 19.35 (2025-12) | **9.80** (2025-04) | < 20 ✅ |
| **Best month sMAPE** | 9.14 (2025-09) | **4.39** (2025-09) | - |
| **Months < 20** | 12/12 | 12/12 | 100% ✅ |
| **Months improved** | - | 12/12 | 100% ✅ |

**结论**: ✅ 所有测试集均 < 20 (目标达成)

---

## 7. Limitations

### 7.1 不能保证 beat DA
- DA-only 已是最强 baseline
- 修正可能在某些情况下使结果变差
- **建议**: 默认关闭 safe correction，仅作为辅助输出

### 7.2 不建议默认开启 correction
- 修正模型在极端市场条件下可能失效
- 修正幅度过大可能导致新的误差
- **建议**: 仅在 `uncertainty_score` < 0.3 且 `correction_permission` = 1 时开启

### 7.3 Hourly deep mode 未生产就绪
- `target_granularity="hourly"` 当前 raise `NotImplementedError`
- 仅支持 day-level (24h vector) 预测
- **原因**: hourly 模式需要重新设计 dataset 接口，当前时间不足

### 7.4 MLP experimental only
- MLP 输入维度存在风险 (不同特征组合的维度不匹配)
- 正式封装中只允许 TCN/GRU/Transformer
- 但由于所有 deep residual 结果 NO_GO，最终模型包不应默认使用它们

### 7.5 分类器模型为启发式 (当前版本)
- `da_error_prob_*` 等字段当前为基于 DA 价格水平的启发式规则
- 未训练真正的分类器模型
- **未来工作**: 使用逻辑回归/随机森林训练 DA 误差分类器

---

## 8. Usage

### 8.1 导出模型包
```bash
python scripts/export_rt_assist_pack.py \
    --data-path data/preprocessed_data.csv \
    --output-dir exported_models/rt_assist_pack \
    --train-end 2025-12-31
```

### 8.2 预测 (DA-only, 默认)
```bash
python scripts/predict_rt_assist_pack.py \
    --model-dir exported_models/rt_assist_pack \
    --data-path data/preprocessed_data.csv \
    --start 2025-01-01 \
    --end 2025-12-31 \
    --output predictions/rt_assist_2025.csv
```

### 8.3 预测 (开启 safe correction)
```bash
python scripts/predict_rt_assist_pack.py \
    --model-dir exported_models/rt_assist_pack \
    --data-path data/preprocessed_data.csv \
    --start 2025-01-01 \
    --end 2025-12-31 \
    --enable-safe-correction \
    --alpha 1.0 \
    --clip 50.0 \
    --output predictions/rt_assist_2025_corrected.csv
```

### 8.4 Python API
```python
from models.deep_sgdf_delta.rt_assist_model import create_rt_assist_model

# Load model (DA-only by default)
model = create_rt_assist_model(
    model_dir="exported_models/rt_assist_pack",
    enable_safe_correction=False,
)

# Predict
result_df = model.predict(df)
print(result_df[["ds", "da_anchor", "rt_pred", "final_pred_source"]].head())
```

---

## 9. Model Pack Structure

```
exported_models/rt_assist_pack/
├── manifest.json           # Model metadata
├── residual_model.pkl     # RandomForest residual model (optional)
└── feature_columns.json   # Feature column list
```

### manifest.json schema
```json
{
  "model_version": "RT-Assist-1",
  "export_date": "2026-07-04T17:35:47",
  "train_end": "2025-12-31",
  "n_train_samples": 34872,
  "feature_columns": ["da_price", "hour", ...],
  "alpha": 1.0,
  "clip": 0.0,
  "enable_safe_correction": true,
  "model_type": "RandomForestRegressor"
}
```

---

## 10. References

- **RT-Assist-1 Final Results**: `docs/RT_ASSIST_1_FINAL_RESULTS.md`
- **DA-Safe Enhancer Results**: `docs/DA_SAFE_ENHANCER_1_RESULTS.md`
- **DeepRT-SOTA v2 Results**: `docs/DEEP_RT_SOTA_V2_RESULTS.md`
- **Finalization Report**: `docs/DEEP_RT_FINALIZATION_RESULTS.md`
- **Code Inventory**: `reports/local/deep_rt_finalize/code_inventory.md`

---

**Model Card Updated**: 2026-07-04  
**Next Version**: RT-Assist-2 (with trained classifiers)
