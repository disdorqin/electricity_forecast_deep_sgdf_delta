# P2 Realtime — 2.5 实时基线理解文档

> 阶段 A 产物。目的：厘清 2.5 实时模型入口、SGDFNet/RT916/TimeMixer 经验、DeepSGDFDelta 现状、cutoff-safe 口径，并标记可复用 / 不建议改动 / 可替换模块。
> 仓库：`electricity_forecast_deep_sgdf_delta`（工作仓，仅此处修改）
> 只读参考：`electricity_forecast_model2.5` ≡ 本地 `efm3.0`；`electricity_forecast_model2.0_exp/SGDFNet`（bridge 依赖，永不复制源码）
> 生成日期：2026-07-06

---

## 0. 统一口径与硬边界（先看这个）

| 项 | 结论 |
|---|---|
| 任务强制 cutoff | **D 日 14:00（D14）**：预测 D+1 实时电价，只允许使用 D-1 14:00 之前可见信息 |
| 2.5 实际运行时 cutoff | **D14**：`SGDFNet/pipeline.py:59` 传 `decision_hour=realtime_cutoff_hour` 默认 14；`TimeMixer/pipeline.py` `cutoff_hour_rt` 默认 14；`RT916/pipeline.py` `asof_hour` 默认 14；timesfm 靠 `skip_style="gap"` 天然截断在 14:00 |
| 2.5 业务文档口径 | ⚠️ `efm3.0/docs/项目执行逻辑与陪跑步骤对齐.md` 写的是 **"D 日 15:00 截止"**，与 CLI 默认 14 **自相矛盾**；以代码实际运行的 **D14** 为准（任务也要求 D14） |
| DeepSGDFDelta 当前 cutoff | ❌ **D15**：`models/deep_sgdf_delta/config.yaml:14` `decision_hour: 15`；`protocol_b_cutoff.py:47/79` 默认 15；`protocol_tag="B_D15_cutoff_walk_forward"` |
| **不一致裁决** | 现有 D15 结果**不能直接进入 3.0 shadow**，也不能直接与 2.5 (D14) baseline 比较。本任务**新增 D14 mode（`decision_hour=14`）**，所有产出/对比/审计一律以 D14 为准 |

**D15 与 D14 的泄漏差**：`_build_protocol_b_visible_frame` 中 `blocked_same_day_mask = (business_day==decision_day) & (timestamp > decision_ts)`。D15 下仅 `>15:00` 的实时 actual 被 DA 替换，**14:00–15:00 这一小时实时 actual 仍可见** → 相对 D14 多泄漏 1 小时，属任务禁止项。D14 mode 必须把 `decision_hour` 设为 14，使物理 14:00 之后的实时 actual 全部不可见。

---

## 1. 2.5 实时模型入口与经验

### 1.1 timesfm（TimesFMBackend）
- 入口：`TimesFMBackend/infer.py` → `price_forecast_copy_分时段预测.py:forecast_next_day`
- 预测方式：**绝对价格预测**，非 delta；日前电价作为外生特征（target="realtime" 用日前电价列）
- cutoff：无 `realtime_cutoff_hour` 参数，靠 `skip_style="gap"` 在 14:00 截断空缺，数据天然不越界
- 输出：`时刻`+`预测值`，24 行
- 经验：**稳定、轻量、CPU 可跑**；是 2.5 实时融合的重要组成

### 1.2 sgdfnet（SGDFNet，A 仓副本）
- 入口：`SGDFNet/pipeline.py:ModelPipeline.predict_range` → 调 `run_protocol_b_cutoff_experiment`
- 预测方式：**delta learning**，`rt_hat = da_anchor + delta_hat`（`protocol_b_cutoff.py:319/334`）
- da_anchor = `日前电价`（DA_COL）
- cutoff：`decision_hour = realtime_cutoff_hour` 默认 **14**（`pipeline.py:59`）；同日 post-cutoff 用 DA 填充
- 输出：`rt_hat` 列，24 小时（hour24 = 次日 00:00）
- 经验：**本任务最相关的单模型基线**（deep_sgdf_delta 即在其协议上生长）

### 1.3 timemixer（TimeMixer）
- 入口：`TimeMixer/pipeline.py:predict_range` → `repro_pipeline.py:run_monthly_reproduction`
- 预测方式：RT 行以 `pred_day_ahead_price` 作 DA 注入特征，模型**直接输出绝对 RT**（`y_pred`）
- cutoff：`cutoff_hour_rt` 默认 **14**；非 14 会告警
- 输出：`时刻`+`y_pred`，校验 24 行（D 01:00 ~ D+1 00:00）
- 经验：GPU 模型，需要 Torch 环境

### 1.4 rt916（RT916_SpikeFusionNet）
- 入口：`RT916_SpikeFusionNet/pipeline.py:predict_range` → `core.run_joint_da_rt_daily_backtest`
- 预测方式：**anchor/delta**：先产 DA 再注入 RT，`annual_model.py:106` `pred = editable_mask*pred + (1-mask)*anchor_pred`
- da_anchor = 自身 DA 预测
- cutoff：`asof_hour` 默认 **14**（realtime）/ 24（dayahead）
- 输出：`预测实时电价`，24 行
- 经验：GPU 模型，融合 spike 处理

**2.5 实时融合参考**：sMAPE_floor50 ≈ **23**（任务给定，作为本任务重要 baseline；不直接复现融合，但用作对照锚点）。

---

## 2. SGDFNet 关键经验（来自 model2.0_exp/SGDFNet）

- **数据契约** `data_contract.py`：
  - `RT_COL="实时电价"`、`DA_COL="日前电价"`、`TIMESTAMP_COL="时刻"`
  - `ACTUAL_COLS`：10 个“…实际值”列（负荷/新能源实际）
  - `da_anchor = DA_COL`；`delta_target = rt_actual - da_anchor`
  - `business_day`：物理 00:00 归前一日；`target_hour`：0→24（hour_business 1..24）
  - `segment`：`_segment_from_hour` → `1_8 / 9_16 / 17_24`
- **指标** `metrics.py`：`sMAPE_floor50 = capped_smape(..., floor=50.0)`；口径：`y_true/y_pred` 各自 floor 到 50 后算 `200*|Δ|/(|y_t|+|y_p|)` 均值。本任务统一复用此口径。
- **协议** `protocol_b.py` / `protocol_b_cutoff.py`：`decision_hour` 默认 **15（D15）**；`_build_protocol_b_visible_frame` 实现 cutoff-safe 可见帧（post-cutoff 实时 actual 用 DA 替换）；`_build_train_val_by_decision_day` 做 walk-forward（val_days 窗口）。

---

## 3. RT916 关键经验
- 联合 DA/RT 每日回测；anchor+delta 结构；`asof_hour=14` cutoff；残差/尖峰融合在 `RT916_SpikeFusionNet/core`。
- 复用点：anchor 注入思路、DA→RT 的协变量传递。

## 4. TimeMixer 关键经验
- 月度复现脚本 `repro_pipeline.py`；RT 直接预测 + DA 注入；`cutoff_hour_rt=14`。
- 复用点：DA 注入特征构造、24 行完整性校验。

---

## 5. DeepSGDFDelta 当前结构（工作仓现状）

- **版本**：V1（逐小时 per-hour TCN/GRU）、V2（按日 24h decoder + hour/segment embedding）、V3（多尺度 TCN/GRU + teacher distillation/moe）。
- **配置**：`models/deep_sgdf_delta/config.yaml`（默认 tcn, hidden 64, blend=deep_only, decision_hour=15）；`configs/trendknight_x_profiles.yaml`（debug_cpu / v3_fast_tcn / v3_fast_gru / v3_multiscale_tcn / v3_teacher_residual / v3_teacher_moe）。
- **已有脚本**：`scripts/run_phase2_monthly_backtest.py`、`evaluate_phase2_trendknight.py`、`search_phase2_champion.py`、`p0_reproduce_sgdfnet_baseline.py`、大量 `audit_*`。
- **已有测试**：`tests/` 含 cutoff 安全、business_day/hour24、blend、go/no-go 等。
- **当前最佳候选**：`TrendKnightRT(DeepFinal-2)` 被归档为 MAIN_REALTIME_MODEL，但判定 **MODEL_NO_GO**。
  - sMAPE_floor50：DeepFinal-2 ≈ **26.69%**（34 feat fallback）/ **26.61%**（real SGDFNet, DeepFinal-3B）
  - `DEEP_RT_SOTA_CHAMPION_SEED.md` 报 17.26 但被证为**评估 bug**（真实 ≈27.66），且 `pred_std/target_std=0.28` 疑 collapse。

### 5.1 ⚠️ 根因级发现（决定实验方向）
所有 prior 深度候选 NO_GO 的共同根因（见 `DEEP_RT_SOTA_2B_RESULTS.md` / `DEEP_REALTIME_MODEL_FINAL_HANDOFF.md`）：
- **RT−DA 残差自相关 ≈ 0，几乎不可预测**；
- **DA anchor 与 RT 相关 ≈ 0.85，已是强基线**；
- 因此所有“残差修正”都劣于直接用 DA anchor。

**对本任务实验策略的启示**：
1. 不能只押注“RT−DA 残差学习”；纯残差模型预期弱。
2. 真正的可预测信号在 **D14 cutoff 前可见的当日实时轨迹**（D-1 1..14 点的 RT 相对 DA 的偏离形态）+ DA anchor + 日历/时段 + **预测侧**（非实际侧）负荷/新能源预报。
3. 目标不是“击败 DA anchor 本身”（DA 已很强），而是**击败 2.5 实时融合 ≈23** 这一更弱的目标——通过更好利用可见实时轨迹与多模型互补。
4. prior 工作还受“测试样本过少（6 天）”“评估 bug”拖累；本任务框架必须保证 **walk-forward 多月、24 行完整、统一指标、cutoff 审计**。

---

## 6. cutoff-safe 检查结果

| 检查点 | 结果 |
|---|---|
| 2.5 模型运行口径 | D14（代码实际）✅ |
| SGDFNet protocol_b 默认 | D15 ❌ → 本任务强制改 D14 |
| deep_sgdf_delta config | D15 ❌ → 新增 D14 mode |
| 是否使用 D+1 actual | 禁止，所有 walk-forward 以 target_day 为测试、train/val 严格早于 decision_day |
| 是否使用 post-D14 实时 actual | D15 下 14–15 点可见 → 通过 D14 mode 消除 |
| 是否用未来 spike/negative label 作在线特征 | 禁止；`residual_for_spike/negative_module` 仅评估期用 y_true 派生，不进训练特征 |
| 预测侧特征 | 允许（负荷/新能源“预测值”在 cutoff 前已发布） |
| 实际侧特征 | 仅允许 pre-cutoff 的 lag/可见 actual history（`use_visible_actual_history=true`） |

**结论**：deep_sgdf_delta 现有结果为 D15，**不符合进入 3.0 shadow 的 D14 要求**。阶段 B 框架将 `decision_hour` 设为可配（默认 14），所有产出走 D14。

---

## 7. 模块处置建议

### 7.1 可复用（直接复用，不重写）
- `sgdfnet_bridge.py`：定位并 import SGDFNet 的 data_contract / protocol_b_cutoff / metrics（**永不复制源码**）。
- `metrics.py`：`capped_smape` / `smape_floor50` / MAE / RMSE / 分段指标 —— 统一指标口径。
- `output_contract.py`：标准输出 schema（`trend_pred`, `delta_pred`, `residual_for_spike_module`, `residual_for_negative_module`）。
- `protocol_b_cutoff._build_protocol_b_visible_frame` / `_build_train_val_by_decision_day`：walk-forward + cutoff-safe 可见帧逻辑（仅把 `decision_hour` 改为 14）。
- `business_time.py` / `add_business_time_columns`：business_day / hour_business / segment 对齐。
- `tests/test_*`：cutoff 安全、hour24、go/no-go 测试范式。

### 7.2 不建议改动（保持与 2.5 / SGDFNet 一致）
- SGDFNet / 2.5 任何源码（只读参考，绝不修改）。
- `sMAPE_floor50` 指标定义（保持口径可比）。
- business_day/hour_business 映射（保持 24 行完整性与段划分 1_8/9_16/17_24）。

### 7.3 可替换 / 新增（本任务主战场）
- **新增 D14 mode**：在框架与所有脚本中把 cutoff 显式设为 14。
- **新增统一实验框架**（阶段 B）：`run_realtime_p2_walkforward.py` / `compare_p2_realtime_candidates.py` / `audit_p2_cutoff_safety.py`，支持 2025/2026 多月、统一指标、cutoff 审计、与 2.5 baseline 对比。
- **替换/增强模型主体**：在 D14-cutoff-safe 协议上替换或增强模型（TCN/GRU/Transformer/DLinear/TimesNet 等），不推翻数据契约与指标。
- **产出 3.0 candidate package**（阶段 F）：`exports/efm3_candidates/realtime_trend/{run_id}/`。

---

## 8. 待阶段 B/C 验证的开放问题
- 数据实际覆盖的 2025/2026 月份上限（框架自动探测，缺失月份跳过并说明）。
- SGDFNet baseline（D14）在本数据的真实 sMAPE_floor50（作为单模型对照）。
- DA-anchor baseline 的 sMAPE_floor50（强参考）。
- 哪些深度候选在 D14 下能逼近/优于 2.5 融合 ≈23。
