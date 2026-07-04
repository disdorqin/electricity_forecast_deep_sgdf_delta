# DeepRT-SOTA v2 Scope

**Status:** ACTIVE
**Date:** 2026-07-04
**Phase:** DeepRT-SOTA-v2 Phase A
**Model:** DeepRT-SOTA v2 (独立 realtime price deep model)

---

## 1. 本模型负责 (In Scope)

本模型是一个**独立的深度学习实时电价预测模型**，负责以下功能：

- **standalone realtime price deep learning prediction** — 直接预测实时电价（rt_actual / realtime_price）
- **direct rt price prediction** — 输出 `rt_pred`（24小时完整向量或逐小时预测）
- **residual-to-anchor prediction** — 可选模式：预测 `residual = rt_actual - da_anchor` 或 `residual = rt_actual - base_pred`
- **multi-month walk-forward evaluation** — 支持多月滚动回测
- **model pack export** — 导出可部署的模型包（model.pt + config + schema）
- **script callable inference** — 可通过脚本独立调用训练/预测/评估/导出

**输出字段（必须）**：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `business_day` | date | 业务日期（严格使用 business_time.py） |
| `hour_business` | int | 业务小时 1-24（严格使用 business_time.py） |
| `ds` | datetime | 原始时间戳 |
| `rt_pred` | float | 实时电价预测值 |
| `confidence` | float | 预测置信度 [0, 1] |
| `model_version` | str | 模型版本标识 |

**输出字段（可选）**：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `base_anchor` | float | 基准锚点（da_anchor 或 base_pred） |
| `residual_pred` | float | 残差预测（如果 target_mode=residual） |
| `negative_risk_score` | float | 负价风险分数（如果 risk_features=on） |
| `spike_risk_score` | float | 尖峰风险分数（如果 risk_features=on） |
| `uncertainty` | float | 预测不确定性估计 |

---

## 2. 本模型不负责 (Out of Scope)

以下功能**不在本模型中实现**：

- **day-ahead price prediction** — 日前电价预测由其他模型负责
- **final risk fusion** — 最终风险融合由主系统负责
- **ledger dynamic fusion** — 账本动态融合由主系统负责
- **production deployment** — 生产部署由主系统负责
- **manual correction** — 人工修正
- **final spike/negative postprocess** — 最终尖峰/负价后处理（可以使用风险特征作为输入，但模型输出必须是 realtime price prediction）
- **TrendKnightRT legacy code** — 旧模型已归档，不作为基础

---

## 3. 数据 Contract

### 3.1 时间对齐规则

**严格复用** `models/deep_sgdf_delta/business_time.py`：

```python
from models.deep_sgdf_delta.business_time import add_business_time_columns

df = add_business_time_columns(df, timestamp_col="ds")
```

规则：
- `timestamp D 00:00:00` → `business_day = D-1, hour_business = 24`
- `timestamp D 01:00~23:00` → `business_day = D, hour_business = 1~23`

**禁止**：
- 使用 naive `ds.date()` / `normalize()` 乱对齐
- 手动计算 business_day / hour_business

### 3.2 泄露检查

**FULL_DAY 模式**（当前默认）：

目标日 D 的预测只能使用：
- ✅ 目标日 forecast-side features
- ✅ D-1 及以前 actual / realtime history
- ✅ calendar features
- ✅ previously generated model/risk features that are available before prediction

**禁止**：
- ❌ 目标日 realtime actual
- ❌ 目标日 earlier actual
- ❌ test actual
- ❌ future month information

### 3.3 输入数据

**必须**：
- raw hourly data（rt_actual / realtime_price）

**可选**：
- risk feature pack（negative_prob, spike_prob, etc.）
- SGDFNet / baseline prediction
- forecast-side features（load_forecast, renewable_forecast, etc.）

---

## 4. 特征设计

### 4.1 Price history features

| 特征名 | 说明 |
|--------|------|
| `rt_lag_24h` | 昨日同时刻实时电价 |
| `rt_lag_48h` | 前日同时刻实时电价 |
| `rt_lag_72h` | 大前日同时刻实时电价 |
| `rt_lag_168h` | 上周同时刻实时电价 |
| `previous_day_rt_mean` | 昨日实时电价均值 |
| `previous_day_rt_std` | 昨日实时电价标准差 |
| `previous_7d_same_hour_mean` | 过去7天同时刻实时电价均值 |
| `previous_7d_same_hour_std` | 过去7天同时刻实时电价标准差 |

### 4.2 Anchor / forecast features

| 特征名 | 说明 |
|--------|------|
| `da_anchor` | 日前电价锚点 |
| `forecast_price` | 预测电价 |
| `sgdfnet_pred` | SGDFNet 预测（可选，如果没有真实预测，记录 missing） |
| `available_base_pred` | 可用基准预测（可选） |
| `anchor_spread` | 锚点价差 |

**重要**：
- 如果没有真实 `sgdfnet_pred`，**不要 fallback 成 fake 强特征**
- 必须记录 missing

### 4.3 Calendar features

| 特征名 | 说明 |
|--------|------|
| `hour_sin` | 小时正弦编码 |
| `hour_cos` | 小时余弦编码 |
| `dow_sin` | 星期正弦编码 |
| `dow_cos` | 星期余弦编码 |
| `month_sin` | 月份正弦编码 |
| `month_cos` | 月份余弦编码 |
| `is_weekend` | 是否周末 |
| `period_id` | 时段 ID（1_8=0, 9_16=1, 17_24=2） |

### 4.4 Risk features（可选）

可以复用已有风险特征：

| 特征名 | 说明 |
|--------|------|
| `negative_prob` | 负价概率 |
| `negative_risk_score` | 负价风险分数 |
| `spike_prob` | 尖峰概率 |
| `spike_risk_score` | 尖峰风险分数 |
| `deviation_down_prob` | 向下偏离概率 |
| `deviation_up_prob` | 向上偏离概率 |
| `deviation_risk_score` | 偏离风险分数 |
| `relative_spike_prob` | 相对尖峰概率 |
| `relative_down_prob` | 相对向下概率 |

**说明**：
- risk features are model-generated features, not test actual
- 如果使用 risk features，必须在 feature_manifest 中说明

### 4.5 Forecast-side power market features（可选）

尽量自动识别：

| 特征名 | 说明 |
|--------|------|
| `load_forecast` | 负荷预测 |
| `renewable_forecast` | 可再生能源预测 |
| `wind_forecast` | 风电预测 |
| `solar_forecast` | 光伏预测 |
| `tie_line_forecast` | 联络线预测 |
| `bidding_space_forecast` | 竞价空间预测 |
| `forecast_net_load` | 净负荷预测 |
| `forecast_renewable_share` | 可再生能源占比预测 |
| `forecast_supply_demand_gap` | 供需缺口预测 |
| `forecast_bidding_pressure` | 竞价压力预测 |

缺失可以接受，但必须在 feature_manifest 中记录。

---

## 5. 模型架构候选

实现至少 4 类模型 profile：

| 模型名 | 说明 |
|--------|------|
| `deep_rt_mlp` | 多层感知机 |
| `deep_rt_tcn` | 时间卷积网络 |
| `deep_rt_gru` | 门控循环单元 |
| `deep_rt_transformer` | Transformer 编码器 |

可选：
| 模型名 | 说明 |
|--------|------|
| `deep_rt_nbeats` | N-BEATS（如果实现） |
| `deep_rt_patchtst_lite` | PatchTST 简化版（如果实现） |

### 5.1 推荐结构

**shared input**：
- sequence input: past 7/14/30 days hourly sequence
- static/current input: target day forecast/calendar/risk features

**output**：
- 24-hour vector prediction per business_day（优先推荐）
- 或 hourly prediction

**输出**：
- `rt_pred[24]`
- `confidence[24]`

### 5.2 Loss

支持以下 loss：
- Huber loss
- MAE loss
- sMAPE_floor50 aligned loss
- period weighted loss
- negative/spike bucket weighted loss

### 5.3 配置项

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `--loss` | `huber`, `mae`, `smape_floor50`, `hybrid` | Loss 函数 |
| `--seq-len-days` | `7`, `14`, `30` | 序列长度（天） |
| `--model-profile` | `deep_rt_tcn`, `deep_rt_gru`, `deep_rt_transformer`, `deep_rt_mlp` | 模型架构 |
| `--target-mode` | `direct`, `residual_to_da`, `residual_to_base` | 目标模式 |
| `--risk-features` | `on`, `off` | 是否使用风险特征 |
| `--forecast-features` | `on`, `off` | 是否使用预测侧特征 |

---

## 6. 训练脚本

**脚本**：`scripts/train_deep_realtime_sota.py`

**命令示例**：

```bash
python scripts/train_deep_realtime_sota.py \
  --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \
  --target-month 2026-02 \
  --model-profile deep_rt_tcn \
  --target-mode direct \
  --seq-len-days 14 \
  --risk-features on \
  --forecast-features on \
  --loss hybrid \
  --epochs 80 \
  --batch-size 64 \
  --lr 0.001 \
  --out-dir artifacts/deep_rt_sota/exp_tcn_2026_02
```

**训练 split**：
- train: target_month 之前所有可用数据
- val: train 最后 30 天
- test: target_month

**必须保存**：
- `model.pt`
- `config.yaml`
- `feature_manifest.json`
- `train_manifest.json`
- `metrics_summary.json`
- `predictions.csv`
- `training_curves.csv`

**manifest 必须记录**：
- `target_month`
- `model_profile`
- `target_mode`
- `seq_len_days`
- `feature_columns`
- `risk_features_used`
- `forecast_features_used`
- `n_train_days`
- `n_val_days`
- `n_test_days`
- `leakage_check`
- `metric_status`
- `created_at`

---

## 7. 预测脚本

**脚本**：`scripts/predict_deep_realtime_sota.py`

**命令示例**：

```bash
python scripts/predict_deep_realtime_sota.py \
  --model-dir artifacts/deep_rt_sota/exp_tcn_2026_02 \
  --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \
  --start 2026-02-01 \
  --end 2026-02-28 \
  --out predictions/deep_rt_sota_2026_02.csv
```

**输出**：
- `business_day`
- `hour_business`
- `ds`
- `rt_pred`
- `confidence`
- `model_version`

**不要输出 y_true unless `--eval-mode`**.

---

## 8. 评估脚本

**脚本**：`scripts/evaluate_deep_realtime_sota.py`

**输入**：
- `predictions.csv`
- ground truth data

**输出**：
- `metrics_summary.json`
- `monthly_metrics.csv`
- `period_metrics.csv`
- `bucket_metrics.csv`
- `hourly_metrics.csv`
- `baseline_comparison.csv`
- `go_nogo.md`

**必须评估**：
- overall sMAPE_floor50
- MAE
- RMSE
- 1_8
- 9_16
- 17_24
- negative bucket
- spike bucket
- normal bucket
- monthly stability

**baseline comparison 至少包括**：
- DA anchor if valid
- naive previous day
- previous 7d same hour mean
- available SGDFNet / base model if provided

---

## 9. 实验计划

### 9.1 小炮实验 2026-02

**目标月**：`2026-02`

**实验矩阵**：

| 参数 | 可选值 |
|------|--------|
| `model_profile` | `deep_rt_mlp`, `deep_rt_tcn`, `deep_rt_gru`, `deep_rt_transformer` |
| `target_mode` | `direct`, `residual_to_da` |
| `seq_len_days` | `7`, `14` |
| `risk_features` | `off`, `on` |

最多 16 个组合。

**裁决**：
- `PASS_FAST`: best model beats naive previous-day and DA/available anchor by >=0.5pp
- `STRONG_FAST`: best overall <20
- `SOTA_FAST`: best overall <17
- `NO_GO_FAST`: no model beats simple baseline

### 9.2 多月 backtest

如果 2026-02 小炮至少 `PASS_FAST`，运行多月回测：

```bash
python scripts/run_deep_realtime_sota_backtest.py \
  --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \
  --target-months 2026-01,2026-02,2026-03,2026-04,2026-05 \
  --out-dir reports/local/deep_rt_sota/backtest_2026_01_05
```

**裁决**：
- `SOTA_CANDIDATE`: beats all available standalone baselines
- `STRONG_SOTA`: mean monthly sMAPE_floor50 <17
- `TARGET_SOTA`: mean monthly sMAPE_floor50 <15
- `NO_GO`: does not beat simple baselines

---

## 10. 导出模型包

**脚本**：`scripts/export_deep_realtime_sota_pack.py`

**导出内容**：
- `model.pt`
- `config.yaml`
- `feature_manifest.json`
- `model_card.md`
- `predict_schema.json`
- `metrics_summary.json`

**要求**：
可以通过脚本调用：

```bash
python scripts/predict_deep_realtime_sota.py \
  --model-dir exported_models/deep_rt_sota_champion \
  --data-path <data> \
  --start <date> \
  --end <date> \
  --out <csv>
```

---

## 11. 报告

**报告**：`docs/DEEP_RT_SOTA_1_RESULTS.md`

**必须包含**：
1. 为什么重启 DeepRT-SOTA v2
2. 旧 TrendKnightRT 失败原因
3. 新模型范围
4. 数据集与特征
5. 小炮实验结果
6. 是否进入多月 backtest
7. 多月结果（如果运行）
8. best model
9. 是否达到：beats baselines / <20 / <17 / <15
10. 模型包是否导出
11. 下一步建议
12. **不允许伪造指标**

---

## 12. 测试要求

**新增测试**：

```text
tests/test_deep_rt_sota_dataset.py
tests/test_deep_rt_sota_features.py
tests/test_deep_rt_sota_model.py
tests/test_train_deep_realtime_sota.py
tests/test_predict_deep_realtime_sota.py
tests/test_evaluate_deep_realtime_sota.py
tests/test_run_deep_realtime_sota_backtest.py
tests/test_export_deep_realtime_sota_pack.py
```

**至少运行**：

```bash
pytest tests/test_deep_rt_sota_dataset.py \
       tests/test_deep_rt_sota_features.py \
       tests/test_deep_rt_sota_model.py \
       tests/test_train_deep_realtime_sota.py \
       tests/test_predict_deep_realtime_sota.py \
       tests/test_evaluate_deep_realtime_sota.py \
       tests/test_export_deep_realtime_sota_pack.py -q
```

---

## 13. 严禁事项

1. **不允许用 test actual 生成预测**
2. **不允许伪造指标**
3. **不允许泄露**（目标日 realtime actual，未来信息）
4. **不允许手动计算 business_day / hour_business**（必须使用 `business_time.py`）
5. **不允许在没有真实 `sgdfnet_pred` 时 fallback 成 fake 强特征**

---

*本文件定义 DeepRT-SOTA v2 范围。任何范围变更需在新 Phase 中提出。*

---

## 附录：旧 TrendKnightRT 归档原因

**状态**：`ENGINEERING_COMPLETE, MODEL_NO_GO, ARCHIVE_AS_MAIN_REALTIME_MODEL`

**旧结果**：
- DeepFinal-3 real SGDFNet + 36 features: sMAPE_floor50 ≈ 26.61
- Residual baseline lab:
  - DA anchor ≈ 26.69
  - HGB residual ≈ 27.56
  - Ridge / MLP / bias 全部未超越 DA anchor

**旧结论**：
- TrendKnightRT 当前架构不适合作为主力 realtime deep model

**新方向**：
- 重启第二代独立深度模型：DeepRT-SOTA v2
- 目标：构建可以通过脚本调用的独立 realtime price deep model
- 不是最终 fusion 模块，不是风险模块，不是 ledger，不是日前电价模型
