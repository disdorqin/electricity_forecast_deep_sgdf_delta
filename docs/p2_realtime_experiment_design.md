# P2 Realtime — 实验设计文档（Experiment Design）

> 阶段 B 产物。描述统一实时实验框架的设计：目标、cutoff、数据、指标、模型注册表、walk-forward 协议、输出 schema、对比协议、安全。

## 1. 目标
在 `electricity_forecast_deep_sgdf_delta` 中系统复现/比较/调优**实时电价趋势预测**候选模型，找出是否优于 2.5 实时融合（sMAPE_floor50 ≈ 23）的 candidate。只输出 trend/delta，不覆盖 3.0 最终实时预测。

## 2. Cutoff 口径（铁律：D14）
- 预测目标日 T，决策日 = T-1，可见信息截止 **T-1 14:00**。
- 允许：business_day < T-1 全量历史实时 actual；T-1 日 1..14 点实时 actual；DA 锚点（T 日日前电价，预测值）；**预测侧**负荷/新能源预报（T 日"预测值"列）；日历/时段特征。
- 禁止：T-1 日 15..24 点实时 actual；目标日 actual；任何由未来 actual 派生的在线特征；用 y_true 派生的 spike/negative label 作在线特征。
- 实现：`p2_common.assemble_day_features` 对 lag-24 在 h>14 时强制置 0（mask），其余 lag 取历史日同小时；审计记录每个目标日 `max_visible_realtime_timestamp = T-1 14:00`。
- 现有 deep_sgdf_delta 配置 `decision_hour=15`（D15）**已弃用**，本框架强制 D14。

## 3. 数据
- 源：`electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.xlsx`（39168×23，2022-01 起）。
- 列：时刻 / 日前电价(DA) / 实时电价(RT) / 负荷·新能源（预测值 + 实际值两列族）。
- 列族约定：**预测值** = 预测侧（cutoff 前已发布，允许）；**实际值** = 实际侧（仅 pre-cutoff lag/可见历史允许，且经 `use_visible_actual_history` 处理）。
- 缺失月份/日期：自动跳过（iter_target_days 仅取 24 行完整的 business_day），并在报告中说明，不伪造。

## 4. 统一指标（复用 SGDFNet `capped_smape(floor=50)`）
- 主指标：sMAPE_floor50（趋势预测 vs y_true）。
- 辅助：MAE、RMSE。
- 分段：period 1_8 / 9_16 / 17_24 的 sMAPE_floor50。
- 极端：尖峰（|y_true|>500）、负价（y_true<0）的 sMAPE_floor50；正常段 degradation（vs DA anchor）。
- 工程：训练耗时、推理耗时、failed days、NaN 数、missing hour 数。
- cutoff 审计：每目标日 max_visible ≤ T-1 14:00。

## 5. 模型注册表（`p2_common.MODEL_REGISTRY` + baselines）
| 模型 | 类型 | 目标 | 说明 |
|---|---|---|---|
| da_anchor | baseline | — | trend_pred = DA，强基线 |
| sgdfnet_d14 | baseline (bridge) | delta | 复用 SGDFNet Protocol B，decision_hour=14 |
| tcn_day | deep | delta/abs | 按日 24h TCN decoder |
| gru_day | deep | delta/abs | 按日 24h GRU decoder |
| dlinear_day | deep | delta/abs | 分解线性 |
| linear_day | deep | delta/abs | 逐时线性（对照） |

所有 deep 模型：输入 [24, F]（F≈28+lags），输出 [24] delta 或 abs；trend = da + delta（delta 模式）或直接 abs。

## 6. Walk-forward 协议（monthly）
- 对每个测试月 M：仅用 business_day < M_start - val_days 的数据训练（train_horizon 滑动窗口，默认 180 天），用 [M_start-val_days, M_start) 作 early-stopping 验证；预测 M 月全部目标日。
- 特征/目标标准化（训练统计量），推理时反标准化。
- 严格时序：模型永远看不到目标日及之后的 actual。

## 7. 输出 schema（统一，符合 3.0 candidate contract）
`predictions.csv` 列：business_day, ds, hour_business, period, da_anchor, delta_pred, trend_pred, model_name, model_version, confidence, run_id, y_true, spike_pred, negative_pred, residual_for_spike_module, residual_for_negative_module, is_spike, is_negative。
目录：`outputs/p2_realtime/{run_id}/{predictions,metrics,reports,audit}/`。

## 8. 对比协议（compare_p2_realtime_candidates.py）
- 聚合多个 run_id 的 metrics.json + per_month.csv。
- 对比对象：da_anchor、sgdfnet_d14、各 deep 候选；引用 2.5 实时融合 ≈23 作为外部基线（明确标注"引用值，未在本仓复现融合"）。
- 输出：realtime_candidate_comparison_report.md + metrics.json + per-month breakdown + 2025/2026 月度表。
- 通过标准（来自任务）：多月份平均 sMAPE_floor50 优于/接近 2.5 实时基线；period 17_24 不恶化；尖峰/负价不恶化；24 行完整；无 NaN/leakage；D14-safe；有完整报告。

## 9. 安全边界（与任务一致）
不修改 model2.5 / 3.0 正式链路 / submission_ready.csv；不提交 data/models/outputs 大文件；不伪造 actual；不跳过 NaN/失败检查；只报完整月份；不删除失败案例；不把 fallback 冒充正常预测。
