# P2 Realtime Deep Exploration — Live Results (D14 cutoff, 2025-01..2026-06, 536 days)

> 最后更新：2026-07-06 21:10 — 全部阶段 A–F 完成。

## 最终结论：REALTIME_P2_RECOMMENDATION = NO_GO；P2_REALTIME_RESULT = PARTIAL

## 全量结果（sMAPE_floor50，越低越好）

| 模型 | sMAPE_floor50 | MAE | RMSE | 9_16 | 17_24 | spike | neg | cutoff |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| da_anchor (DA 锚点强基线) | **31.11** | 64.03 | 110.80 | 53.07 | 17.65 | 20.61 | 64.06 | D14 |
| sgdfnet_d14 (忠实复现 2.5 单模型) | 31.99 | 64.54 | 109.48 | 55.11 | 17.99 | 21.30 | 64.32 | D14 |
| gru_day (深度 delta, 最佳新模型) | 34.16 | 67.11 | 110.41 | 60.44 | 18.24 | 21.69 | 79.29 | D14 |
| tcn_day (深度 delta) | 34.23 | 68.12 | 112.28 | 59.87 | 19.12 | 22.47 | 77.55 | D14 |
| dlinear_day | 35.24 | 68.85 | 111.07 | 62.37 | 19.32 | 22.30 | 83.08 | D14 |
| linear_day | 35.35 | 69.14 | 111.62 | 62.78 | 18.88 | 23.53 | 83.28 | D14 |
| tcn_abs (直接预测 RT) | 46.31 | 97.09 | 134.83 | 78.33 | 24.94 | 34.30 | 122.62 | D14 |

## 关键发现
- 深度 delta 模型（tcn/gru/linear/dlinear）全部 34–35%，未超越 DA 锚点（31.11）与忠实复现的 SGDFNet D14（31.99）。
- 根因：RT−DA 残差自相关≈0，从 cutoff-safe 特征几乎不可预测；DA anchor（相关0.85）已是强基线。
- 2.5 融合实时 ≈23% 来自 4 个**多样**模型集成；单架构/单模型无法达到。
- cutoff 审计：全部 7 个 run ALL PASS（D14，无泄漏，无 NaN，24h 完整）。
- 实验过程靠 live ticker 抓到并修复 3 个 bug：dlinear/linear `layers` 参数、dlinear 输出形状 (64 vs 24)、tcn_abs 启动路径笔误。

## 产物
- 阶段 D：outputs/p2_realtime/comparison_report.md + comparison_metrics.json
- 阶段 E：outputs/p2_realtime/ablation_report.md
- cutoff：outputs/p2_realtime/_comparison/realtime_cutoff_safety_report.md
- 阶段 F 包：exports/efm3_candidates/realtime_trend/p2_realtime_20260706/（trend_predictions/metrics/comparison/cutoff/ablation/manifest/promotion_decision）
- 最终报告：outputs/p2_realtime/P2_REALTIME_FINAL_REPORT.md（13 节）
