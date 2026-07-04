# DeepRT-SOTA-3C: Conditional Regime Specialist + Kill-Switch

## 最终报告

### 1. 执行总结

| 阶段 | 状态 | 说明 |
|------|------|------|
| Phase A: DA Weakness Audit | ✅ KEEP | 找到27个high-priority regimes |
| Phase B: Trigger Model | ✅ KEEP | 所有threshold都有模型KEEP |
| Phase C: Conditional Specialist | ✅ KEEP | 平均改进0.73pp，11/18个月有改进 |
| Phase D: Deep Model | ⏭ 跳过 | 先评估多月验证再决定 |
| Phase E: Multi-month Validation | ❌ NO_GO | 2026-02灾难性失败，平均改进-10.80pp |
| Phase F: Probabilistic Forecasting | ⏭ 跳过 | sMAPE方向失败，不继续 |
| Phase G: Final Report | ✅ 完成 | 本文档 |

**最终裁决**: **REGIME_SPECIALIST_NO_GO**

---

### 2. 为什么全局residual模型失败

（来自DeepRT-SOTA-3）
- corr(RT, DA) = 0.85（DA捕捉了RT变化的85%）
- corr(residual, residual_lag_24h) ≈ 0（residual无自相关性）
- 所有特征与residual相关性很低（|corr| < 0.15）

**结论**: residual (RT - DA) 从根本上不可预测。

---

### 3. DA Weakness Audit (Phase A)

**数据**: `shandong_pmos_hourly.csv` (2022-01 至 2026-06, 39168小时)

**Global DA sMAPE (day-level)**: 16.17

**Top 3 High-Priority Regimes**:
1. `online_period_da_combo=17_24_da_quantile_low`: DA sMAPE=91.16 (excess vs global=74.99pp)
2. `online_period_da_combo=9_16_da_quantile_low`: DA sMAPE=76.66 (excess=60.49pp)
3. `da_quantile_level=da_quantile_low`: DA sMAPE=75.50 (excess=59.34pp)

**发现**: DA在**低电价时段**和**特定小时（11-16）**误差最大。

**输出文件**:
- `regime_inventory.csv`
- `da_error_by_regime.csv`
- `regime_predictability.csv`
- `da_weakness_report.md`

---

### 4. Trigger Model (Phase B)

**方法**: 训练分类器预测"DA是否会大错"

**标签**: `large_da_error = abs(rt_actual - da_anchor) >= threshold`

**Thresholds**: 50, 100, 150, 200

**模型**: LogisticRegression, HGBClassifier, RandomForest_small

**KEEP条件**:
- `precision@top20% >= base_large_error_rate * 1.5`
- `recall@top20% >= 0.35`

**结果**:

| Threshold | Model | AUC | Precision@20% | Recall@20% | KEEP? |
|-----------|-------|-----|-------------------|----------------|--------|
| 50 | RandomForest_small | 0.75 | 0.68 | 0.39 | ✅ |
| 100 | HGBClassifier | 0.75 | 0.40 | 0.46 | ✅ |
| 150 | HGBClassifier | 0.74 | 0.31 | 0.49 | ✅ |
| 200 | HGBClassifier | 0.72 | 0.22 | 0.49 | ✅ |

**结论**: 所有threshold都有模型KEEP，trigger可以预测DA大错。

**输出文件**:
- `trigger_leaderboard.csv`
- `threshold_metrics.csv`
- `topk_metrics.csv`
- `calibration_report.csv`
- `trigger_report.md`

---

### 5. Conditional Specialist (Phase C)

**方法**: 只在trigger fire时应用residual correction

**形式**:
```
if trigger_score >= threshold:
    final_pred = da_anchor + alpha * clipped(specialist_residual_pred)
else:
    final_pred = da_anchor
```

**验证方法**: Walk-forward backtest (2024-05 至 2025-12)

**结果**:

| 月份 | Specialist sMAPE | DA sMAPE | 改进 |
|------|-------------------|------------|------|
| 2024-06 | 0.75 | 12.18 | +11.43 |
| 2024-07 | 15.12 | 17.58 | +2.46 |
| 2024-08 | 31.70 | 11.84 | -19.86 |
| ... | ... | ... | ... |
| **平均** | **12.31** | **13.04** | **+0.73** |

**月份有改进**: 11/18

**KEEP条件**: `global sMAPE improves vs DA by >= 0.3pp` ✅

**结论**: Phase C KEEP，条件专家模型可以改进sMAPE。

**输出文件**:
- `specialist_leaderboard.csv`
- `decision_log.csv`
- `predictions.csv`
- `bucket_metrics.csv`
- `period_metrics.csv`
- `specialist_report.md`

---

### 6. Multi-month Validation (Phase E)

**验证月份**: 2026-01, 2026-02, 2026-03, 2026-04, 2026-05

**结果**:

| 月份 | Model sMAPE | DA sMAPE | 改进 | Trigger Fire Rate |
|------|--------------|------------|------|--------------------|
| 2026-02 | 72.55 | 27.87 | **-44.68** | 27.53% |
| 2026-03 | 19.24 | 19.59 | +0.34 | 8.20% |
| 2026-04 | 14.92 | 15.43 | +0.52 | 1.11% |
| 2026-05 | 15.98 | 16.58 | +0.61 | 9.54% |
| **平均** | **30.67** | **19.87** | **-10.80** | - |

**问题**: 2026-02出现**灾难性失败**（Model sMAPE=72.55），原因：
1. Trigger在2026-02 fire rate太高（27.53%）
2. Specialist模型在2026-02做出大量错误修正
3. 2026-02的DA误差可能不可预测（distribution shift？）

**裁决**:
- ❌ REGIME_SPECIALIST_GO: 不满足（平均改进 < 0.3pp，且有月份比DA差>1.0pp）
- ❌ REGIME_SPECIALIST_AUX: 不满足（global degradation = -10.80pp >> 0.1pp）
- ✅ REGIME_SPECIALIST_NO_GO: 无有效改进

---

### 7. 哪些方向被舍弃

1. **全局residual学习** (DeepRT-SOTA-2B): 失败，residual不可预测
2. **直接预测RT价格** (实验A/B/C): 失败，无法超越DA-only
3. **深度模型** (Phase D): 跳过，因为多月验证失败
4. **概率预测** (Phase F): 跳过，因为sMAPE方向失败

---

### 8. 下一步建议

1. **分析2026-02失败原因**:
   - 检查2026-02数据是否有异常
   - 分析trigger为什么在2026-02 fire rate太高
   - 考虑使用更严格的trigger threshold

2. **尝试其他条件修正策略**:
   - 不使用trigger，直接在特定regime（如`da_quantile_low`）应用修正
   - 使用分位数回归（预测RT价格的分位数，而不是point forecast）

3. **接受DA-only baseline**:
   - DA-only是一个非常强的baseline（day-level sMAPE ≈ 16-19）
   - 可能RT价格本质上不可预测（除了DA）
   - 转而改进其他方面（如特征解释性、不确定性估计等）

4. **收集更多数据或特征**:
   - 当前特征与residual相关性很低（|corr| < 0.15）
   - 可能需要其他类型的特征（如天气、经济指标等）

---

### 9. 不允许伪造指标

所有实验都使用：
- **Walk-forward backtest**（避免数据泄漏）
- **Day-level sMAPE**（正确的评估指标）
- **DA-only baseline**（公平的比较基准）
- **Kill-Switch**（3个连续KILL停止）

**没有**:
- Oracle baseline
- Test actual选择策略
- 伪造指标

---

### 10. 修改文件列表

**新增脚本**:
1. `scripts/audit_da_weakness_regimes.py` (Phase A)
2. `scripts/train_da_error_trigger.py` (Phase B)
3. `scripts/train_conditional_residual_specialist.py` (Phase C)
4. `scripts/train_regime_specific_deep_model.py` (Phase D, 未运行)
5. `scripts/run_multi_month_backtest.py` (Phase E)

**新增报告**:
1. `reports/local/deep_rt_sota/da_weakness_regime_audit_2026_01_05/da_weakness_report.md`
2. `reports/local/deep_rt_sota/da_error_trigger_2026_01_05/trigger_leaderboard.csv`
3. `reports/local/deep_rt_sota/conditional_specialist_2026_02/specialist_leaderboard.csv`
4. `reports/local/deep_rt_sota/regime_specialist_backtest_2026_01_05/monthly_metrics.csv`
5. `docs/DEEP_RT_SOTA_3C_REGIME_SPECIALIST_RESULTS.md` (本文档)

**修改文件**: 无

---

### 11. Pytest结果

（待运行）

---

### 12. Commit Hash

（待提交）

---

## 附录：详细结果表格

### A. Phase A: Top 10 High-Priority Regimes

| Regime | Regime Value | N Rows | Coverage | DA sMAPE | Excess vs Global |
|---------|---------------|---------|----------|----------|-------------------|
| online_period_da_combo | 17_24_da_quantile_low | 1041 | 2.66% | 91.16 | 74.99 |
| online_period_da_combo | 9_16_da_quantile_low | 6729 | 17.18% | 76.66 | 60.49 |
| da_quantile_level | da_quantile_low | 9786 | 24.98% | 75.50 | 59.34 |
| online_low_da | True | 9786 | 24.98% | 75.50 | 59.34 |
| da_negative | True | 4426 | 11.30% | 68.58 | 52.41 |
| ... | ... | ... | ... | ... | ... |

### B. Phase B: Trigger Model Performance

（见`trigger_leaderboard.csv`）

### C. Phase C: Conditional Specialist Performance

（见`specialist_leaderboard.csv`）

### D. Phase E: Multi-month Validation

| Month | Model sMAPE | DA sMAPE | Improvement |
|-------|--------------|------------|-------------|
| 2026-02 | 72.55 | 27.87 | -44.68 |
| 2026-03 | 19.24 | 19.59 | +0.34 |
| 2026-04 | 14.92 | 15.43 | +0.52 |
| 2026-05 | 15.98 | 16.58 | +0.61 |

**Average Improvement**: -10.80pp

---

**结束**
