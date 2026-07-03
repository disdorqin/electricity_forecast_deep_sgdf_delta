# DeepFinal-3a: Feature Hardening Report

**Date:** 2026-07-03
**Phase:** DeepFinal-3a (Feature Hardening + Audit Fixes)

---

## 1. P0 Issues Fixed

### P0-1: FORMAL_READY 误判 ✅

**问题：** `audit_feature_coverage()` 在 `sgdfnet_fallback_used=True` 时仍然可能输出 `FORMAL_READY` 裁决，因为未区分 real coverage 和 effective coverage。

**修复：**
- 新增 `sgdfnet_real_coverage`（仅计算来自真实预测文件的 coverage）
- 新增 `sgdfnet_effective_coverage`（含 fallback 的 coverage）
- 新增裁决 `FALLBACK_READY`：当 effective coverage >= 95% 但 fallback 被使用
- 裁决规则：

| 裁决 | n_features | real_coverage | fallback | leakage_ok |
|------|-----------|---------------|----------|------------|
| FORMAL_READY | >= 25 | >= 95% | False | True |
| FALLBACK_READY | >= 25 | 任意 | True | True |
| PARTIAL_READY | >= 15 | >= 80% | False | 任意 |
| NOT_READY | 其他 | — | — | — |

- `formal_train_ready` 只对 `FORMAL_READY` 返回 `True`

### P0-2: FULL_DAY Lag / Residual 泄露 ✅

**问题 1 — `rt_mean_24h` / `rt_std_24h` 使用 `.shift(1).rolling(24)`：**
- 在 D 日 hour 16 时，`.shift(1).rolling(24)` 会包含 D 日 hour 1~15 的 actual
- FULL_DAY 模式下这些数据不可见 → 泄露风险

**修复：** 改为 `groupby("business_day") → agg mean/std → shift(1) business_day`：
- `rt_mean_24h` = D-1 全天的 rt_actual 均值
- `rt_std_24h` = D-1 全天的 rt_actual 标准差
- 新增 `previous_day_delta_mean_24h` / `previous_day_delta_std_24h`

**问题 2 — `sgdfnet_residual_lag_1h` 使用 `.shift(1)`：**
- FULL_DAY 模式下 D 日 earlier actual 不可见

**修复：** FULL_DAY 模式下置为 0，INTRADAY 模式才启用。

**问题 3 — `sgdfnet_residual_mean_7d` 使用 `.shift(1).rolling(168)`：**
- 会包含 D 日 earlier 小时的 residual

**修复：** 改为按 `hour_business` 分组，对 `business_day` 序列 rolling(7) 且 shift(1)，严格跨天计算。

### 验证测试

新增 9 个泄漏测试：
- `test_rt_mean_24h_not_affected_by_current_day_change`
- `test_sgdfnet_residual_mean_7d_not_affected_by_current_day`
- `test_sgdfnet_residual_lag_1h_is_zero_in_full_day`
- `test_rt_lag_24h_is_d_minus_1_same_hour`
- `test_check_leakage_on_residual_features`
- `test_previous_day_delta_mean_24h_is_d_minus_1_daily_mean`
- `test_intraday_has_same_day_actuals`
- `test_sgdfnet_residual_lag_1h_nonzero_in_intraday`
- `test_intraday_requires_explicit_mode`

---

## 2. Task A: SGDFNet Prediction Loader ✅

新增 `models/deep_sgdf_delta/sgdfnet_prediction_loader.py`：
- 自动识别时间戳列（ds / timestamp / time / 时刻）
- 自动识别预测列（sgdfnet_pred / pred / prediction / y_pred / rt_pred）
- 自动添加 business_day / hour_business
- 按 `(business_day, hour_business)` 去重
- 输出 coverage report（JSON + MD）
- Coverage < 95% → formal training fail

新增文档 `docs/SGDFNET_PREDICTION_INPUT_CONTRACT.md`

---

## 3. Task B: 训练脚本阻断 fallback 正式指标 ✅

- `--feature-mode full` 且无 `--sgdfnet-predictions` 且无 `--allow-sgdfnet-fallback` → 直接 fail
- Fallback 训练的 manifest 标记 `metric_status = SMOKE_ONLY`, `formal_metric = false`

---

## 4. Task C: Fallback Audit 结果

### Fallback Audit

```text
$ python scripts/audit_realtime_features.py --allow-sgdfnet-fallback

Verdict:           FALLBACK_READY
n_features:        36
Required missing:  0
SGDFNet real cov:  0.0%
SGDFNet eff cov:   99.9%
Fallback used:     True
Formal ready:      False
Lag coverage:      100%
Leakage OK:        True
```

### Real SGDFNet Audit

没有真实 SGDFNet 预测文件可用。

```text
NO_REAL_SGDFNET_AVAILABLE — 无法执行 real SGDFNet audit。
```

---

## 5. Task D: DeepFinal-3 是否训练

```text
DeepFinal-3 blocked: missing real SGDFNet predictions.

必要条件不满足：
  1. real SGDFNet audit != FORMAL_READY
  2. sgdfnet_real_coverage == 0%
  3. 没有真实的 SGDFNet 预测文件

只有真实 SGDFNet 文件 coverage >= 95% 才能：
  - 运行正式训练
  - 产出正式指标
  - 进入 DeepFinal-3 评估
```

---

## 6. DeepFinal-2 结果复盘

| 项目 | DeepFinal-2 (之前) | DeepFinal-2 (修正后) |
|------|-------------------|---------------------|
| n_features | 2 → 34 | 2 → **36** |
| Feature verdict | FORMAL_READY (误判) | **FALLBACK_READY** ✅ |
| **formal_train_ready** | True (误判) | **False** ✅ |
| FULL_DAY leakage | 有 (rolling leak) | **无** ✅ |
| SGDFNet real coverage | 0% | **0%** |
| SGDFNet effective coverage | 99.9% (fallback) | **99.9%** (fallback) |
| Test sMAPE (Feb 2026) | 26.69% | **26.69%** (SMOKE_ONLY) |
| Metric status | 未标记 | **SMOKE_ONLY** ✅ |
| 测试总数 | 152 | **186+** ✅ |

---

## 7. 修改文件清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `models/deep_sgdf_delta/sgdfnet_prediction_loader.py` | SGDFNet 预测加载器 |
| `docs/SGDFNET_PREDICTION_INPUT_CONTRACT.md` | 预测输入合约 |
| `docs/DEEPFINAL_3A_FEATURE_HARDENING_REPORT.md` | 本文档 |
| `tests/test_realtime_feature_builder_no_leak_full_day.py` | FULL_DAY 泄露测试 (9) |
| `tests/test_sgdfnet_prediction_loader.py` | 预测加载器测试 (12) |
| `tests/test_audit_formal_ready_requires_real_sgdfnet.py` | 正式裁决测试 (5) |
| `tests/test_train_realtime_blocks_fallback_formal_metrics.py` | 训练阻断测试 (2) |

### 修改文件

| 文件 | 变更 |
|------|------|
| `models/deep_sgdf_delta/realtime_feature_builder.py` | 修复 verdict 逻辑 + FULL_DAY 泄露 |
| `models/deep_sgdf_delta/realtime_feature_contract.py` | 新增 `previous_day_delta_mean/std` |
| `scripts/train_realtime_deep_model.py` | 新增 fallback 阻断 + SMOKE_ONLY 标记 |
| `scripts/audit_realtime_features.py` | 更新 audit 输出区分 real/effective coverage |
| `tests/test_realtime_feature_builder.py` | 更新 audit test 断言 |

---

## 8. 结论

- ✅ **FORMAL_READY 误判已修复** — 现在正确输出 FALLBACK_READY
- ✅ **FULL_DAY leakage 已修复** — 所有 lag/residual 特征使用 business_day 对齐
- ✅ **训练阻断逻辑已生效** — fallback 训练标记 SMOKE_ONLY
- ✅ **SGDFNet Prediction Loader 可用** — 等待真实预测文件
- ✅ **186+ 测试全部通过**（不含 SGDFNet bridge 依赖缺失导致的 11 个预存失败）

**DeepFinal-3 正式训练 blocker：缺少真实 SGDFNet 预测文件。**
