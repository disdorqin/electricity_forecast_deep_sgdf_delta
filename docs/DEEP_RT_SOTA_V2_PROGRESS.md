# DeepRT-SOTA v2 进度总结

**日期**: 2026-07-04

**任务**: 构建独立的深度学习实时电价预测模型（DeepRT-SOTA v2）

---

## 已完成

### Phase A: 范围定义 ✅
- 创建 `docs/DEEP_RT_SOTA_V2_SCOPE.md`
- 明确定位：独立 realtime price deep model，不负责日前/融合/风险修正

### Phase B: 数据集模块 ✅
- 创建 `models/deep_sgdf_delta/deep_rt_sota_dataset.py`
- 创建 `tests/test_deep_rt_sota_dataset.py`
- 测试: 15/15 通过
- **问题**: 数据集模块较简化，实际需要更完整的实现

### Phase C: 特征工程 ✅
- 创建 `models/deep_sgdf_delta/deep_rt_sota_features.py`
- 创建 `tests/test_deep_rt_sota_features.py`
- 测试: 11/11 通过
- 实现特征组：
  - Price history features (8个)
  - Anchor/forecast features (3个)
  - Calendar features (8个)
  - Risk features (可选，2个)

### Phase D: 模型架构 ✅
- 创建 `models/deep_sgdf_delta/deep_rt_sota_model.py`
- 创建 `tests/test_deep_rt_sota_model.py`
- 测试: 19/19 通过
- 实现 4 种模型：
  - `deep_rt_mlp`
  - `deep_rt_tcn`
  - `deep_rt_gru`
  - `deep_rt_transformer`

### Phase E: 训练脚本 🔄
- 创建多个训练脚本：
  - `scripts/train_deep_rt_sota_minimal.py` (MLP, 逐小时)
  - `scripts/train_deep_rt_sota_tcn.py` (TCN, 按日预测)
  - `scripts/train_residual.py` (residual-to-DA, 有 bug)
  - `scripts/quick_test.py` (MLP 快速测试)
  - `scripts/quick_test_tcn.py` (TCN 快速测试, **运行中**)

---

## 当前指标结果

### 基线对比 (2026-02)

| 模型/Baseline | sMAPE_floor50 | MAE | 状态 |
|---------------|---------------|-----|------|
| DA anchor | **26.70** | 75.45 | 最强基线 |
| Naive previous day | 51.33 | 148.83 | 弱基线 |
| DeepRT-SOTA (TCN, day-level) | 42.76 | 111.74 | ❌ 未超越 DA |
| DeepRT-SOTA (MLP, hourly) | 87.09 | 228.48 | ❌ 未超越 DA |

### sMAPE 计算修复 ✅
- **问题**: 之前使用错误的 sMAPE 公式
- **修复**:  now using canonical `smape_floor50` (floor=50)
- **影响**: 所有指标已重新计算

---

## 关键问题

### 1. 模型性能不佳 ❌
- **目标**: sMAPE_floor50 < 20
- **当前最佳**: 42.76 (TCN)
- **差距**: 22.76 pp

### 2. 测试样本太少 ⚠️
- TCN (day-level) 只有 6 天测试样本
- 可能导致评估不准确

### 3. 特征工程不完整 ⚠️
- 风险特征当前是合成的（非真实）
- 需要真实风险特征或禁用

### 4. 训练脚本不稳定 ⚠️
- `train_residual.py` 有 bug (X shape (0,))
- 需要修复和测试

---

## 下一步计划

### 立即执行 (等待 TCN 训练完成)
1. 检查 TCN 快速测试结果
2. 如果 TCN 优于 MLP，继续优化 TCN
3. 如果 TCN 仍不佳，尝试 GRU/Transformer

### 短期目标 (1-2小时)
1. 修复 `train_residual.py` bug
2. 运行 Phase H 小炮实验 (至少 2-3 个模型)
3. 获取稳定的指标结果

### 中期目标 (今天内)
1. 完成 Phase F (预测脚本)
2. 完成 Phase G (评估脚本)
3. 如果模型性能达标 (sMAPE < 26.70)，进入 Phase I (多月 backtest)

---

## 文件清单

### 新增文件
```
docs/
  DEEP_RT_SOTA_V2_SCOPE.md
  DEEP_RT_SOTA_V2_PROGRESS.md (当前文件)

models/deep_sgdf_delta/
  deep_rt_sota_dataset.py
  deep_rt_sota_features.py
  deep_rt_sota_model.py

scripts/
  train_deep_rt_sota_minimal.py
  train_deep_rt_sota_tcn.py
  train_residual.py (有 bug)
  quick_test.py
  quick_test_tcn.py (运行中)
  compute_baseline.py
  generate_synthetic_risk_features.py

tests/
  test_deep_rt_sota_dataset.py
  test_deep_rt_sota_features.py
  test_deep_rt_sota_model.py
  test_deep_rt_sota_e2e.py

artifacts/deep_rt_sota/
  minimal_exp/results.json
  tcn_exp/results.json
  synthetic_risk_features.csv
  baseline_comparison.json
```

---

## 待完成 Phase

- [x] Phase A: 范围定义
- [x] Phase B: 数据集
- [x] Phase C: 特征工程
- [x] Phase D: 模型架构
- [ ] Phase E: 训练脚本 (进行中)
- [ ] Phase F: 预测脚本
- [ ] Phase G: 评估脚本
- [ ] Phase H: 小炮实验
- [ ] Phase I: 多月 backtest
- [ ] Phase J: 导出模型包
- [ ] Phase K: 最终报告

---

## 内存记录

- 业务时间规则: 严格使用 `business_time.py`
- sMAPE 计算: 使用 `smape_floor50` (floor=50)
- 不允许伪造指标
- 不允许泄露 test actual
- 目标: 打赢所有 baseline (sMAPE < 26.70)
