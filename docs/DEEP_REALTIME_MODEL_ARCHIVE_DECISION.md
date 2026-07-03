# Deep Realtime Model — Archive Decision

**Date:** 2026-07-03
**Status:** ARCHIVED

---

## 1. 归档裁决

| 组件 | 裁决 |
|------|------|
| TrendKnightRT current architecture (TCN/GRU/Transformer, gated fusion) | **ARCHIVED** |
| FULL_DAY residual correction | **NO-GO** |
| Residual-only deep | **NOT_TRIGGERED** (deferred until HGB signal) |
| HGB residual baseline | **NO-GO** (27.56% vs DA anchor 26.69%) |
| SGDFNet + bias / Ridge / MLP | **NO-GO** (all worse than DA anchor) |

## 2. 归档原因

1. **TrendKnightRT 几乎完全复制 DA anchor/SGDFNet**
   - Corr(pred, DA anchor) = 0.9987
   - Corr(pred, SGDFNet) = 0.9974

2. **Residual signal 不存在**
   - residual_pred_std = 0.31
   - true residual_std = 113.38
   - residual_std_ratio = 0.27%

3. **所有 residual 模型均无法超越 DA anchor**
   - HGB residual: 27.56%
   - Ridge residual: 27.71%
   - MLP residual: 28.27%
   - Hour bias: 27.47%
   - DA anchor (baseline): **26.69%**

4. **训练动力学异常**
   - Best epoch = 1（第一轮后永不改善）
   - Val sMAPE 从 31.59 持续上升到 32.63
   - Train loss 几乎没有下降

## 3. 保留资产

以下资产已验证可用，**推荐保留供主系统调用或参考**：

### 特征工程
- `models/deep_sgdf_delta/realtime_feature_builder.py` — 完整特征管线（36 特征）
- `models/deep_sgdf_delta/realtime_column_mapping.py` — 中文→英文列映射
- `models/deep_sgdf_delta/realtime_feature_contract.py` — 特征合约
- `models/deep_sgdf_delta/business_time.py` — 山东市场业务时间对齐

### 数据加载
- `models/deep_sgdf_delta/sgdfnet_prediction_loader.py` — SGDFNet 预测加载 + coverage audit
- `models/deep_sgdf_delta/realtime_dataset_final.py` — PyTorch Dataset 实现

### 评估工具
- `scripts/evaluate_realtime_deep_model.py` — 评估入口
- `scripts/build_realtime_baseline_leaderboard.py` — Baseline 排行榜
- `scripts/run_realtime_model_backtest.py` — 多月回测
- `scripts/diagnose_trendknight_failure.py` — 预测行为诊断
- `scripts/run_residual_baseline_lab.py` — Residual baseline lab
- `scripts/audit_realtime_features.py` — 特征审计
- `scripts/audit_training_dynamics.py` — 训练动力学审计

### 监控工具
- `scripts/monitor_sgdfnet_protocol_b.py` — Protocol B 产物监控
- `scripts/consolidate_sgdfnet_predictions.py` — 预测文件合并
- `scripts/run_deepfinal3_when_ready.py` — 自动触发流水线

### 模型包骨架
- `models/deep_sgdf_delta/trendknight_rt.py` — 模型定义（可复用/residual head）
- `models/deep_sgdf_delta/residual_baselines.py` — Residual baseline 模型

### 文档
- `docs/SGDFNET_PREDICTION_INPUT_CONTRACT.md`
- `docs/DEEP_REALTIME_MODEL_FINAL_HANDOFF.md`
- `docs/DEEP_REALTIME_FINAL_RESULTS.md`
- `docs/DEEPFINAL_4_FAILURE_DIAGNOSIS_REPORT.md`

## 4. 不再继续

以下方向已无继续价值：

- trendknight_rt_tcn / gru / transformer 主力训练
- residual-only deep model
- 更大 backbone（更深/更宽网络）
- 盲目调参（LR、batch size、epoch）
- Gated fusion mode A/B/C
- 更多 calendar/lag 特征（已有 36 个，无 improvement）

## 5. 未来可重新打开的条件

只有当满足**任一**条件时，才允许多深度实时模型：

1. **新增明确强信号特征**
   - 真实供需缺口
   - 机组级/负荷级高质量 forecast
   - 市场行为特征（如报价曲线）
   - 任何可使 HGB residual < 23 的新特征

2. **HGB residual 在某个新特征集下降至 < 23**
   - 这是 residual signal 存在的最低门槛
   - HGB < 23 → residual-only deep 可尝试

3. **INTRADAY 模式需要深度模型**
   - 当前 lab 限定在 FULL_DAY 模式
   - INTRADAY 模式有更多可用信息（同日 earlier actuals）
   - 可能在 INTRADAY 下深度模型表现不同

4. **主系统明确需要 diversity candidate**
   - 如果主系统需要多个模型 vote/ensemble
   - TrendKnightRT 即使 26.61% 也可作为 weak candidate

## 6. 下一阶段推荐

```text
1. 产差模块 (production deviation)
2. 尖峰/极端高价模块 (spike)
3. 负价模块 (negative price)
4. ledger/dynamic fusion
5. intraday tracker（限定 INTRADAY 模式）
```
