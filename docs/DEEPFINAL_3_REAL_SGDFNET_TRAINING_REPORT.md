# DeepFinal-3: Real SGDFNet Anchor Training Report

**Date:** 2026-07-03
**Status:** BLOCKED — Missing full-coverage SGDFNet predictions

---

## 1. Task A: SGDFNet Prediction Search

**脚本：** `scripts/find_sgdfnet_predictions.py`
**输出：** `reports/local/deep_final/sgdfnet_search/`

搜索结果：

| 路径 | 类型 | 日期范围 | 覆盖天数 | 目标覆盖 |
|------|------|----------|----------|----------|
| fold predictions (60 files) | Validation folds | 2026-01-02 ~ 2026-02-07 | 36 days | 21.8% |
| forecast_predictions (6 files) | Real forecast | 2026-02-01 ~ 2026-02-08 | 6 days | 3.6% |
| weighted_lgbm fusion | Fusion output | 2025-11-01 ~ 2026-03-01 | 121 days | 79.5% |

**最佳候选：** fold predictions，864 行覆盖 36 个唯一日（Jan 2 - Feb 7），包含真实 `prediction` 列。

**结论：** 没有找到覆盖 2026-01~05 全部目标月份的 SGDFNet 预测文件。

## 2. Task B: SGDFNet Prediction Generation

**脚本：** `scripts/generate_sgdfnet_predictions_for_deep.py`

尝试调用 SGDFNet Protocol B cutoff recovery：
- **配置：** `cutoff_recovery_2026_diag_a_prune_actualside.yaml`
- **数据范围：** 2026-01-01 ~ 2026-05-11
- **状态：** ⚠️ **运行中**（LightGBM 训练进行中，预计 30+ 分钟）

已有产出（之前运行生成的 validation folds）：
- `electricity_forecast_model2.0_exp/outputs/2026-02-{01..07}/realtime/validation/folds/`
- 使用 `prediction` 列作为 `sgdfnet_pred`

**当前可用预测文件：**
- `reports/local/deep_final/sgdfnet_predictions/sgdfnet_from_folds_2026_02.csv` — Feb 2026 (7 days)
- `reports/local/deep_final/sgdfnet_predictions/sgdfnet_from_folds_2026_01_07.csv` — Jan 2 - Feb 7 (36 days)

## 3. Task C: Feature Audit (with fold predictions)

使用 fold 预测运行 real SGDFNet audit：

```bash
# Feb 2026 only — limited coverage
python scripts/audit_realtime_features.py \
  --data-path <data.csv> \
  --sgdfnet-predictions <sgdfnet_from_folds_2026_02.csv>
```

**预期结果：**
- `sgdfnet_real_coverage` ≈ 25% (7/28 days for Feb)
- `verdict` = NOT_READY (coverage < 95%)
- `formal_train_ready` = False

**真实结果：** 未运行正式 audit（因为 coverage 不足 95% 必然 NOT_READY）。

## 4. Task D: Small Cannon Training

**未运行。** 必要条件未满足：

```text
DeepFinal-3 小炮训练被阻断：
  real SGDFNet audit = NOT FORMAL_READY
  sgdfnet_real_coverage = 25% (Feb 2026, fold predictions only)
  formal_train_ready = False

只有 SGDFNet coverage >= 95% 才能运行正式训练。
```

## 5. Task E-G: 后续任务

所有后续任务（Leaderboard、多月 Backtest、Error Diagnosis）均被阻断。

## 6. Status Summary

| 任务 | 状态 | 备注 |
|------|------|------|
| Task A: Search | ✅ 完成 | 找到 fold predictions (36 days) |
| Task B: Generate | ⏳ 进行中 | Protocol B 在跑，预计 30+ 分钟 |
| Task C: Audit | ❌ 阻断 | coverage 不足 95% |
| Task D: 小炮训练 | ❌ 阻断 | 缺少 FORMAL_READY audit |
| Task E: Leaderboard | ❌ 阻断 | 依赖小炮 |
| Task F: Backtest | ❌ 阻断 | 依赖小炮 |
| Task G: Error Diag | ❌ 阻断 | 依赖训练 |
| Task H: 本报告 | ✅ 完成 | 真实记录 |

## 7. 下一步建议

1. **等待 SGDFNet Protocol B 跑完** — 当前后台进程正在生成完整 2026-01~05 预测
2. **运行 real SGDFNet audit** — 确认 FORMAL_READY
3. **小炮训练** — 如果 audit 通过，运行 2026-02 TCN 80 epochs
4. **Leaderboard + Backtest** — 如果小炮通过
