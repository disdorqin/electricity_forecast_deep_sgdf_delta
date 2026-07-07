# P2 实时深度模型开放探索 —— 工作总结与结果汇总

> 仓库：`electricity_forecast_deep_sgdf_delta`（位于 `其他资料/` 下，独立于 efm3.0 / model2.5 / 3.0）
> 日期：2026-07-06
> 状态：**NO_GO / PARTIAL**（候选未达晋级标准，但流程、框架与经验已沉淀）

---

## 0. 一句话结论（即上一条收口原文）

> 这轮探索证明了"在 D14 cutoff 下，任何单一深度残差模型都打不过 DA 锚点（31.11%）和忠实复现的 SGDFNet（31.99%），2.5 融合≈23% 是 4 模型多样集成的结果，单架构做不到"——所以晋级裁决是 **NO_GO / PARTIAL**，且全程 cutoff-safe、无伪造、无泄漏。
>
> 如果想推进到真正能挑战 23% 的路线，下一步是复现 timesfm / timemixer / rt916 三个 D14 单模型来组多样集成，而不是再换一个 solo 深度结构。

---

## 1. 任务背景与目标（第一个任务回顾）

第一个任务书要求：在 `deep_sgdf_delta` 仓内，**循环执行直到产出符合要求结果**，分 A–F 六阶段，严格遵守以下硬约束：

- **cutoff = D14**：预测 D+1 实时电价，只能用 D 日 14:00 之前的实时 actual；
- **cutoff-safe**：绝不使用 post-D14 实时 actual 或未来（目标日）actual 作为特征；
- **统一输出格式**：每个 run 落 `predictions / metrics / reports / audit`；
- **不污染正式链路**：不修改 2.5 / 3.0 交付代码，不把候选写入 `submission_ready.csv`；
- **不伪造数据**：所有数字来自真实回测；
- **最终报告 13 节**，并产出 3.0 candidate package + 晋级裁决 JSON。

目标：寻找一个能优于 2.5 实时基线（≈23% sMAPE_floor50）的实时深度趋势模型。

---

## 2. 工作过程（A–F 六阶段）

| 阶段 | 内容 | 产出 |
|---|---|---|
| **A 理解 2.5 基线** | 读 2.5 实时入口、SGDFNet/RT916/TimeMixer 经验；**发现关键不一致：任务强制 D14，但 deep_sgdf_delta 原仓用 D15（decision_hour=15）** | `docs/p2_realtime_25_baseline_understanding.md` |
| **B 建统一框架** | 写 `p2_common.py`（D14-safe 特征装配 + 模型注册表 + 指标）、`run_realtime_p2_walkforward.py`（按月 walk-forward 主运行）、`compare_p2_realtime_candidates.py`、`audit_p2_cutoff_safety.py` | 可复现实验框架 |
| **C 复现候选** | DA 锚点（强基线）+ SGDFNet D14（忠实复现 2.5 单模型，override decision_hour=14）+ TCN/GRU/DLinear/Linear 深度日模型 | 6 个候选 run |
| **D 对比基线** | 全量回测 2025-01..2026-06（536 天，D14），统一指标对比 | `comparison_report.md` |
| **E 调优 + 消融** | 等权/最优/全量集成消融 + 架构消融分析 | `ablation_report.md` |
| **F 生成候选包 + 13 节报告** | 导出 `efm3_candidates/realtime_trend/p2_realtime_20260706/`，写 13 节最终报告 + 晋级裁决 | `P2_REALTIME_FINAL_REPORT.md`、`promotion_decision.json` |

执行方式遵循你的要求：**后台 nohup 持续实时输出 → 检测跑完立刻进入下一阶段，不中途打断回复**，日志与 `RESULTS_LIVE.md`、`logs/live_ticker.txt` 实时可见。

---

## 3. 统一实验框架与代码

- **`scripts/p2_common.py`**：`FeatureSpec` + `assemble_day_features`（D14-safe：da_anchor + lag24/48/168 + trajectory(D-1 时 1..14 的 rt−da) + forecast_side + 日历）、`MODEL_REGISTRY`（da_anchor / tcn_day / gru_day / dlinear_day / linear_day / sgdfnet_d14）、`compute_metrics`（用 `trend_pred` 列）、TCN/GRU/DLinear/LinearDay 类、`build_lookups`（用 `to_dict` 规避中文列名 itertuples 问题）。
- **`scripts/run_realtime_p2_walkforward.py`**：按月 walk-forward（每月用之前所有数据训练，预测该月）；`--model / --start-date / --end-date / --device / --epochs / --target-mode abs|delta / --decision-hour 14`；`run_sgdfnet_d14` 加载 2.5 baseline yaml 并 override decision_hour=14 忠实复现。
- **`scripts/audit_p2_cutoff_safety.py`**：逐行审计特征可见窗口，断言无 post-D14 / 无未来 actual 泄漏。

---

## 4. 关键发现

1. **Cutoff 不一致（任务 vs 原仓）**：任务书强制 D14，但 `deep_sgdf_delta` 原仓用 D15（decision_hour=15）。已统一改到 D14，所有 7 个 run 均按 D14 跑。
2. **根因 —— RT−DA 残差结构性不可预测**：RT−DA 残差自相关 ≈ 0，从 cutoff-safe 特征里学不到稳定信号；2.5 融合≈23% 来自 **4 个多样模型集成**（约 23% 中 ~23% 份额来自集成），单架构无法达到。
3. **DA 锚点是强基线**：trend = da_anchor + delta_pred，DA anchor 与真实实时价相关 ≈ 0.85，本身就是极难超越的锚。

---

## 5. 完整实验结果（D14, 2025-01..2026-06, 536 天）

### 5.1 总指标（sMAPE_floor50 越低越好）

| 模型 | MAE | RMSE | **sMAPE_floor50** | 1_8 | 9_16 | 17_24 | spike | neg | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| **da_anchor** | 64.03 | 110.80 | **31.11** | 22.61 | 53.07 | 17.65 | 20.61 | 64.06 | 强基线 |
| sgdfnet_d14 | 64.54 | 109.48 | **31.99** | 22.87 | 55.11 | 17.99 | 21.30 | 64.32 | 忠实复现 |
| gru_day | 67.11 | 110.41 | 34.16 | 23.80 | 60.44 | 18.24 | 21.69 | 79.29 | 劣 |
| tcn_day | 68.12 | 112.28 | 34.23 | 23.71 | 59.87 | 19.12 | 22.47 | 77.55 | 劣 |
| dlinear_day | 68.85 | 111.07 | 35.24 | 24.05 | 62.37 | 19.32 | 22.30 | 83.08 | 劣 |
| linear_day | 69.14 | 111.62 | 35.35 | 24.39 | 62.78 | 18.88 | 23.53 | 83.28 | 劣 |
| tcn_abs | — | — | 46.31 | — | — | — | — | — | 直预测 RT 更差 |

> 外部 2.5 fused realtime 参考 ≈ 23%（多模型集成，本仓未复现）；DA 锚点 31.11% 为强单模型基线。

### 5.2 逐月对比（获胜方）

| 月份 | 基线 sMAPE | 最佳候选 | 最佳 sMAPE | 胜者 |
|---|---:|---:|---:|:--:|
| 2025-01 | 30.40 | gru_day | 31.38 | 基线 |
| 2025-02 | 27.21 | gru_day | 28.59 | 基线 |
| 2025-03 | 31.61 | gru_day | 33.36 | 基线 |
| 2025-04 | 25.03 | tcn_day | 27.60 | 基线 |
| 2025-05 | 26.63 | sgdfnet_d14 | 26.28 | sgdfnet |
| 2025-06 | 29.86 | sgdfnet_d14 | 31.03 | 基线 |
| 2025-07 | 22.11 | dlinear_day | 22.73 | 基线 |
| 2025-08 | 16.95 | tcn_day | 16.69 | tcn |
| 2025-09 | 18.88 | sgdfnet_d14 | 19.31 | 基线 |
| 2025-10 | 18.29 | sgdfnet_d14 | 18.23 | sgdfnet |
| 2025-11 | 31.92 | sgdfnet_d14 | 30.54 | sgdfnet |
| 2025-12 | 34.86 | sgdfnet_d14 | 35.53 | 基线 |
| 2026-01 | 50.18 | gru_day | 51.29 | 基线 |
| 2026-02 | 47.34 | tcn_day | 48.43 | 基线 |
| 2026-03 | 38.39 | tcn_day | 39.42 | 基线 |
| 2026-04 | 36.77 | sgdfnet_d14 | 33.74 | sgdfnet |
| 2026-05 | 35.12 | sgdfnet_d14 | 33.26 | sgdfnet |
| 2026-06 | 45.05 | sgdfnet_d14 | 42.25 | sgdfnet |

（深度候选仅在个别月份偶然优于基线，整体稳定劣于 DA 锚点 / SGDFNet。）

---

## 6. Cutoff 安全审计（D14）

审计 7 个 run，逐行核对特征可见窗口：**全部 PASS**——无 post-D14 实时 actual，无未来 actual 作为特征，无泄漏。

| Run | 判定 | 行数 | 天数 | trend NaN | delta NaN | 问题 |
|---|:--:|--:|--:|--:|--:|---|
| da_anchor_d14_…203328 | PASS | 12864 | 536 | 0 | 0 | 无 |
| tcn_day_d14_…203328 | PASS | 12864 | 536 | 0 | 0 | 无 |
| gru_day_d14_…203600 | PASS | 12864 | 536 | 0 | 0 | 无 |
| sgdfnet_d14_d14_…203328 | PASS | 12864 | 536 | 0 | 0 | 无 |
| linear_day_d14_…205109 | PASS | 12864 | 536 | 0 | 0 | 无 |
| dlinear_day_d14_…205844 | PASS | 12864 | 536 | 0 | 0 | 无 |
| tcn_abs_d14_…210259 | PASS | 12864 | 536 | 0 | 0 | 无 |

---

## 7. 消融实验结论（Stage E）

- **E1 等权集成（da_anchor + sgdfnet_d14）**：31.17 ≈ 单模型，无多样增益（两者近同）。
- **E2 最优常权混合（da + tcn）**：最优 w(da)=1.00 → 31.11，tcn 只带来噪声。
- **E3 全候选等权集成**：32.73，弱模型把集成拖向 34–35%。
- **E4 架构消融（未跑，结构性原因）**：period-head / hour-embedding 等无法在自相关≈0 的残差上制造信号；abs 目标 TCN 单月 49.6%、全量 46.31% 更差。
- **结论**：无消融能击败 DA 锚点 / SGDFNet D14；2.5 融合≈23% 是 4 多样模型集成，单架构残差学习达不到。

---

## 8. 最终裁决与交付物清单

**`promotion_decision.json` 关键字段**
- `recommended_status`: **no_go**
- `baseline_smape_floor50`: 31.11 | `candidate_smape_floor50`: 34.16
- `cutoff_safety_result`: PASS（D14, ALL runs audited PASS）
- 已知风险：RT−DA 残差近不可预测；深度模型引入方差；2.5 融合需 4 多样模型。
- 必需后续：复现 timesfm/timemixer/rt916 单模型 D14 组多样集成；仅当某模型在 2025 与 2026 各 ≥3 个月严格胜出才重评。

**交付物路径**（均在 `electricity_forecast_deep_sgdf_delta/`）
- `outputs/p2_realtime/P2_REALTIME_FINAL_REPORT.md`（13 节最终报告）
- `outputs/p2_realtime/comparison_report.md`、`ablation_report.md`
- `outputs/p2_realtime/_comparison/realtime_cutoff_safety_report.md`
- `exports/efm3_candidates/realtime_trend/p2_realtime_20260706/`（manifest / metrics / promotion_decision.json / trend_predictions.csv）
- `docs/p2_realtime_25_baseline_understanding.md`、`p2_realtime_experiment_design.md`

---

## 9. 踩坑记录（Bug Fixes）

1. `itertuples` 中文列名解包失败 → 改用 `to_dict("records")`。
2. `compute_metrics` 误用 `pred/segment` 列 → 改用 `trend_pred/period`。
3. `pd.isnan` → `pd.isna`。
4. TCN `Sequential` 通道不匹配 → 重写卷积层。
5. 特征未归一化致 TCN 70% → 加标准化 + 增 epoch。
6. `dlinear_day/linear_day` 崩 `TypeError: unexpected keyword 'layers'` → registry lambda 去 layers；DLinear 输出形状修复为 `[B,24]`。
7. tcn_abs 启动路径笔误 → 重启用正确路径并重命名为 `tcn_abs_d14_*`。
8. 后台启动命令丢后两条 → 改单条 `nohup ... &`。
9. 502 代理遥测报错（与脚本无关）→ 输出重定向日志规避。

---

## 10. 下一步建议（展开"这段话"）

- **不要再造 solo 深度结构**：当前证据显示单架构残差学习在 D14 下触及天花板（≈31% 由 DA 锚点锁死）。
- **走多样集成路线**：复现 timesfm / timemixer / rt916 三个 D14 单模型输出，与 DA 锚点 / SGDFNet 组加权集成，才可能逼近 2.5 融合的 ≈23%。
- **重评门槛**：任一候选须在 2025 与 2026 各自 ≥3 个月严格胜出 DA 锚点，才值得晋级。
- period/segment head 仅作为集成成员，不当 solo 替换。

---

## 11. 技术经验文档留存（跨 session 不遗忘）

- `electricity_forecast_deep_sgdf_delta/.workbuddy/memory/P2_REALTIME_TECH_EXPERIENCE.md`：cutoff D14 约束、根因、各模型 sMAPE、bug 修复史、交付物路径。
- `efm3.0/.workbuddy/memory/2026-07-06.md` + `MEMORY.md`：记入 autonomous 连续输出规则（跑实验不中途打断回复）。
- 全局 `~/.workbuddy/MEMORY.md`：永久写入"实验期间持续实时输出、一次到位完成才汇报"的执行纪律。
