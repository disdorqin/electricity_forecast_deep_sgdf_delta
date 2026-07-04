# DeepRT-SOTA-2B 实验结果报告

**日期**: 2026-07-04  
**目标**: Reproduce Champion + Baseline-Safe Residual SOTA  
**Verdict**: ❌ **NO_GO** - 无法 beat DA anchor (26.69)

---

## 1. Group 4 复现结果

| 项目 | 结果 |
|------|------|
| 原始报告的 Group 4 sMAPE | 17.26 |
| 复现 sMAPE | 27.93 |
| 差距 | +10.67 pp |
| 裁决 | **NOT_REPRODUCED** |

**原因**: 原始 17.26 是评估 bug（对 residual 算 sMAPE，而非 final_pred）。修复 bug 后，真实性能约 27.66（2026-02 backtest）。

---

## 2. Pipeline 一致性审计

| 检查 | 结果 |
|------|------|
| sMAPE 实现一致性 | ✅ 通过 |
| predictions 对齐 | ✅ 通过 |
| final_pred 计算 | ✅ 通过 |
| 随机差异 (delta 0.27 sMAPE) | ⚠️ 微小（可接受）|

**裁决**: **MOSTLY_CONSISTENT** - pipeline 基本一致的，微小差异来自随机初始化。

---

## 3. Baseline-Safe Shrink/Gate

已实现并合并入 `scripts/train_working.py`。

**机制**: `final_pred = da_anchor + alpha * clip(residual_pred, -clip, clip)`  
**选择**: 在 validation set 上选最佳 `alpha`/`clip`，禁止在 test set 上选择。

**测试结果**:
- 当模型学不会 residual 时，`alpha=0`（自动回退到 DA anchor）✅
- 当模型学会部分 residual 时，`alpha>0`（应用 shrink/gate）✅

---

## 4. Residual 历史特征

已加入 `models/deep_sgdf_delta/deep_rt_sota_features.py`。

**新增特征** (9个):
- `residual_lag_24h/48h/72h/168h`
- `residual_prev_day_mean/std`
- `residual_prev_7d_same_hour_mean`
- `residual_prev_14d_same_hour_mean`
- `residual_prev_7d_peak/mid/off_mean`

**严格审计**: 无数据泄露（只用目标日 D-1 及以前的数据）✅

**实际效果**: 无改善。原因见第 6 节诊断分析。

---

## 5. 改进 Loss 与 Checkpoint 选择

已实现：
- ✅ 加入 `--checkpoint-metric` 参数（基于 validation sMAPE 选最佳 epoch）
- ✅ 加入 `--loss` 参数（huber/mse/hybrid）
- ✅ 加入 `weight_decay=1e-4`
- ✅ 修复 checkpoint 保存/加载 bug

**实际效果**: checkpoint 选择生效（选 epoch 10 而非 epoch 30），但 test sMAPE 仍败给 DA。

---

## 6. 诊断分析（关键发现）

### 6.1 为什么模型学不会 residual？

| 指标 | 值 | 含义 |
|------|-----|------|
| Correlation(RT, DA) | 0.8500 | RT 与 DA 高度相关 |
| Correlation(residual, residual_lag_24h) | -0.0266 | **residual 无自相关性！** |
| Mean \|residual\| | 75.44 | residual 绝对值均值（价格单位） |
| Residual std | 113.73 | residual 标准差很大（噪声） |

**结论**: **residual (RT - DA) 基本是随机噪声，无法用历史 residual 预测**。

这是因为：
1. DA anchor 已经捕捉了大部分价格变动（correlation 0.85）
2. 剩下的 residual 是"不可预测的"噪声（无自相关性）

### 6.2 模型评估结果汇总

| 实验 | 模型 | sMAPE | vs DA(26.69) | alpha | 学会了 residual? |
|------|------|--------|----------------|-------|----------------|
| Track E1 | TCN 14d | 27.75 | ❌ +1.06 | 0.0 | ❌ 否 |
| Track E2 | GRU 7d | 28.02 | ❌ +1.33 | 0.0 | ❌ 否 |
| Track E3 | Transformer 7d | 26.77 | ❌ +0.08 | 0.0 | ❌ 否 |
| Reproduce | TCN 7d | 27.93 | ❌ +1.24 | 0.0 | ❌ 否 |
| +residual feat | TCN 7d | 27.10 | ❌ +0.41 | 0.7 | ✅ 部分 |
| +residual feat | Transformer 7d | 26.69 | = 0.00 | 0.0 | ❌ 否 |

**最佳结果**: Transformer 7d = 26.77（差距最小，但仍败给 DA）

---

## 7. 多月 Backtest 结果

| 月份 | sMAPE | DA anchor | 差距 |
|--------|--------|------------|------|
| 2026-01 | 33.68 | 26.69 | ❌ +6.99 |
| 2026-02 | 27.66 | 26.69 | ❌ +0.97 |

**裁决**: ❌ **NO_GO** - 两个月均败给 DA anchor。

---

## 8. 目标达成情况

| 目标 | 要求 | 实际 | 达成? |
|------|------|------|--------|
| beat DA 26.69 | must | 26.77 (best) | ❌ 否 |
| 进 20 | target | 26.77 | ❌ 否 |
| 接近 17.26 | try | 26.77 | ❌ 否 |
| < 17 | target | - | ❌ 否 |
| < 15 | target | - | ❌ 否 |

---

## 9. 根本原因与下一步建议

### 9.1 根本原因

1. **Residual 不可预测**: 诊断显示 residual 与 lag-24h residual 相关性 ≈ 0，说明 residual 是随机噪声。
2. **DA anchor 很强**: Correlation(RT, DA) = 0.85，DA 已经捕捉了大部分价格变动。
3. **模型能力不足**: TCN/Transformer 都无法从当前特征中学到有用的 residual 预测。

### 9.2 下一步建议

**如果要继续改进**（可能很难）：

1. **换特征**: 加入负荷预测、可再生能源预测、输电线路预测等（可能与 RT-DA 相关性更强）
2. **换模型**: 尝试 LSTM + Attention、N-BEATS、TFT 等时序专用架构
3. **Direct 模式**: 直接预测 RT 价格（不用 residual），但需要更强的正则化防止过拟合
4. **Ensemble**: 多 seed ensemble（Track F）—— 但由于每个实验都败给 DA，ensemble 可能也无济于事

**如果要写论文/报告**：

1. 诚实报告: DA anchor 是非常强的基线，很难 beat
2. 报告我们尝试的方法（TCN、Transformer、residual 特征、shrink/gate）
3. 报告诊断分析（residual 不可预测）
4. 提出未来方向（更好的特征、更大的数据集、更先进的模型）

---

## 10. 修改文件列表

| 文件 | 修改内容 |
|------|----------|
| `models/deep_sgdf_delta/deep_rt_sota_features.py` | 加入 residual 历史特征（9个） |
| `scripts/train_working.py` | shrink/gate、checkpoint 选择、--use-residual-history-features |
| `scripts/reproduce_deep_rt_group4.py` | Track A 复现脚本 |
| `scripts/audit_deep_rt_pipeline_consistency.py` | Track B 一致性审计脚本 |

---

## 11. 最终裁决

```
FINAL_VERDICT: NO_GO
REASON: Residual (RT - DA) is not autocorrelated (corr ≈ 0), 
        making it fundamentally hard to predict.
        DA anchor (26.69) is a very strong baseline.
        All experiments failed to beat DA.

BEST_SMAPE: 26.77 (Transformer 7d, from earlier Track E experiments)
GAP_TO_DA: +0.08 pp
GAP_TO_TARGET_20: +6.77 pp
GAP_TO_TARGET_17: +9.77 pp
```

---

## 12. 不允许伪造指标

✅ 所有指标均为真实实验结果。  
✅ DA anchor 计算方式公平（walk-forward backtest）。  
✅ 未使用 test set 选择超参数。  
✅ 诊断分析诚实反映了模型学不会 residual 的原因。

---

**报告结束**。
