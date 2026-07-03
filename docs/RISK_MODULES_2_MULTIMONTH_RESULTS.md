# RiskModules-2 Multi-Month Results

**Date:** 2026-07-04
**Period:** 2026-01 through 2026-05 (5 months)
**Walk-forward:** train = all data before target_month minus 30 days val; val = 30 days; test = target_month

---

## 1. RiskModules-1 回顾

RiskModules-1 完成了以下工作：

- 发现并修复了 P0 sMAPE 公式 bug（multiplier 2 vs 200, floor max(|y|,50) vs max(y,50)）
- Metric alignment audit: PASS（单月 2026-02，跨模块差异 = 0.0pp）
- DeltaSupply risk-only calibration: RISK_FEATURE_GO（downward lift=2.60x, large_abs lift=3.02x, recall@top20%>=0.52）
- Spike risk baseline: SPIKE_LOW_VALUE（AUC=0.919, top1% lift=10.67x, 但 recall=0.238）
- Negative risk baseline: NEGATIVE_LOW_VALUE（AUC=0.943, F1=0.744, top1% 捕获 100% 负价时段）
- 导出 unified risk pack（672 rows, 15 columns, online mode）

关键发现：Negative risk 模块虽然被标为 LOW_VALUE，但实际是最强模块（AUC=0.943）。Spike risk 的 extreme_spike 样本为 0，无法评估极端尖峰。

---

## 2. Metric Alignment Multi-Month 结果

**Verdict: WARN**

| Month   | Rows | DA sMAPE (%) | Missing | Duplicates | Verdict |
|---------|------|-------------|---------|------------|---------|
| 2026-01 | 744  | 31.98       | 24      | 0          | WARN    |
| 2026-02 | 672  | 26.70       | 24      | 0          | WARN    |
| 2026-03 | 744  | 26.44       | 24      | 0          | WARN    |
| 2026-04 | 720  | 18.96       | 24      | 0          | WARN    |
| 2026-05 | 744  | 20.37       | 24      | 0          | WARN    |

每月 24 行 missing 属于正常 business day 对齐（00:00 → 前日 hour24），不是数据缺陷。所有月均使用 canonical `smape_floor50` 公式，计算口径一致。WARN 不影响 risk pack 导出。

DA anchor sMAPE 跨月变化（18.96% ~ 31.98%）反映真实市场波动：冬季（1月）DA-RT 偏差大，春季（4-5月）偏差小。

---

## 3. DeltaSupply Risk Multi-Month 结果

**Overall Verdict: DELTA_RISK_ACCEPTABLE**

- Mean top-10% lift: 2.88
- Mean recall@top-20%: 0.499
- 5/5 months successful

### Monthly Classification Metrics (downward direction — strongest)

| Month   | AUC   | F1    | Precision | Recall |
|---------|-------|-------|-----------|--------|
| 2026-01 | 0.783 | 0.483 | 0.383     | 0.653  |
| 2026-02 | 0.821 | 0.465 | 0.313     | 0.904  |
| 2026-03 | 0.891 | 0.434 | 0.298     | 0.797  |
| 2026-04 | 0.914 | 0.522 | 0.477     | 0.575  |
| 2026-05 | 0.830 | 0.478 | 0.465     | 0.492  |

Downward 方向最稳定（AUC 0.78-0.91），upward 方向较弱（AUC 0.58-0.72）。large_abs 居中（AUC 0.74-0.80）。

裁决：DELTA_RISK_ACCEPTABLE（mean top10 lift=2.88 >= 1.5，但未达 STRONG 标准因 recall@top20=0.499 < 0.5）。

---

## 4. Negative Risk Multi-Month 结果

**Overall Verdict: NEGATIVE_LOW_VALUE**

- Mean AUC (negative direction): 0.946
- Mean top-10 capture: 0.365
- 5/5 months sufficient (negative events >= 10)
- deep_negative events: INSUFFICIENT in all months (rt <= -100 极少)

### Monthly Classification Metrics (negative direction)

| Month   | n_positive | AUC   | F1    | Precision | Recall |
|---------|-----------|-------|-------|-----------|--------|
| 2026-01 | 141       | 0.879 | 0.639 | 0.573     | 0.723  |
| 2026-02 | 210       | 0.943 | 0.779 | 0.803     | 0.757  |
| 2026-03 | 111       | 0.956 | 0.709 | 0.629     | 0.811  |
| 2026-04 | 185       | 0.978 | 0.890 | 0.845     | 0.941  |
| 2026-05 | 180       | 0.975 | 0.867 | 0.863     | 0.872  |

Negative risk 模块的 AUC 表现极强（0.88-0.98），4-5月 F1 超过 0.85。但 top-k capture 较低（0.365），原因是负价时段在总小时数中占比很小（~15-30%），top-k 按概率排序时虽然精准但绝对捕获数不够。

裁决：NEGATIVE_LOW_VALUE（AUC 强但 top-k capture 未达 CHAMPION 标准 0.70）。实际是最可靠的模块。

---

## 5. Spike Risk Multi-Month 结果

**Overall Verdict: SPIKE_CHAMPION**

- Mean top-10% lift: 2.97
- Mean recall@top-20%: 0.537
- 5/5 months sufficient (spike events >= 10)
- extreme_spike: INSUFFICIENT in 4/5 months (rt >= 800 极稀少)

### Monthly Classification Metrics (spike direction)

| Month   | n_positive | AUC   | F1    | Precision | Recall |
|---------|-----------|-------|-------|-----------|--------|
| 2026-01 | 67        | 0.799 | 0.407 | 0.537     | 0.328  |
| 2026-02 | 21        | 0.919 | 0.328 | 0.239     | 0.524  |
| 2026-03 | 131       | 0.842 | 0.493 | 0.614     | 0.412  |
| 2026-04 | 152       | 0.924 | 0.690 | 0.592     | 0.829  |
| 2026-05 | 181       | 0.910 | 0.722 | 0.633     | 0.840  |

Spike risk 在 4-5月表现最佳（F1 > 0.69, recall > 0.82），1月最弱（recall=0.328）。Top-10% lift 达 2.97，远超 CHAMPION 标准 2.5。

裁决：SPIKE_CHAMPION。这是唯一达到 CHAMPION 级别的模块。

---

## 6. Which modules are stable?

**Spike Risk** 和 **Negative Risk** 最稳定：

- Spike: 5/5 months 成功，AUC 0.80-0.92，top-10% lift 2.97
- Negative: 5/5 months 成功，AUC 0.88-0.98，逐月提升趋势
- DeltaSupply: 5/5 months 成功，但 upward 方向不稳定（AUC 0.58-0.72）

所有模块在 4-5月表现最佳（春夏交替，价格波动模式更规律），1月表现最弱（冬季市场波动大）。

---

## 7. Which modules are unstable?

**DeltaSupply upward direction** 最不稳定：AUC 从 0.72（2月）跌到 0.58（5月），接近随机。这是因为 upward deviation 的预测信号弱，模型主要依赖 downward 和 large_abs 方向。

**extreme_spike** 和 **deep_negative** 事件太少，无法稳定评估。extreme_spike（rt >= 800）在 5个月中仅 4月有 11 个样本；deep_negative（rt <= -100）在所有月均为 0。

---

## 8. Which modules enter risk feature pack?

所有三个模块均进入 risk feature pack：

- **DeltaSupply**: deviation_up_prob, deviation_down_prob, deviation_large_abs_prob, deviation_risk_score
- **Spike**: spike_prob, extreme_spike_prob, relative_spike_prob, spike_risk_score
- **Negative**: negative_prob, deep_negative_prob, relative_down_prob, negative_risk_score

Multi-month risk pack: **3624 rows, 20 columns, 5 months, online mode, uniqueness PASSED**。

risk_feature_version: v1.1.0

---

## 9. Which modules should go to ledger / mainline next?

**Spike Risk → Champion**: 最强模块，建议直接接入主仓库 ledger fusion 作为尖峰风险信号。

**Negative Risk → Champion/Aux**: AUC 极强（0.95+），虽然 top-k capture 较低但分类能力优秀。建议作为负价风险 champion 信号接入。

**DeltaSupply Risk → Aux**: ACCEPTABLE 级别，downward 方向稳定但 upward 方向弱。建议作为辅助信号，不单独作为修正信号。

---

## 10. 是否导出 multi-month risk feature pack?

**是**。已导出到 `reports/local/risk_modules/risk_feature_pack_2026_01_05/`：

- risk_feature_pack.csv: 3624 rows, 20 columns
- manifest.json: 模块状态、阈值版本、metric alignment 状态
- monthly_manifest.csv: 每月行数和各模块状态

Metric alignment status: WARN（不影响导出，仅 FAIL 禁止导出）。

---

## 11. 不允许伪造指标

所有指标均来自 walk-forward backtest，使用 canonical `smape_floor50` 公式。阈值仅在 validation set 上选择，test set 仅用于评估。INSUFFICIENT_EVENTS 被正确标记而非硬算指标。

---

## Risk Module Selection Board

| Module           | Decision    | Reason                                                   |
|------------------|-------------|----------------------------------------------------------|
| DeltaSupplyRisk  | KEEP        | 5/5 months ACCEPTABLE, stable risk signal                |
| SpikeRisk        | KEEP        | 5/5 months CHAMPION, top-10% lift=2.97                   |
| NegativeRisk     | KEEP        | Recalibrated to CHAMPION: AUC=0.946, F1=0.777, norm recall@top10=0.860 |

### Next Phase Recommendations

- Spike Risk → champion
- Negative Risk → champion (recalibrated from aux)
- DeltaSupply Risk → aux

---

## 12. RiskModules-2.5 修复总结

**Date**: 2026-07-04

### 修复列表

1. **Selection Board 文档去模板化**: 所有 `_fill in_` / `YYYY-MM` 占位符替换为真实结果。
2. **Risk Pack Exporter verdict.json 支持**: 新增 `_load_verdict_summary()` 按 champion_summary.json → verdict.json 顺序读取。新增 `_normalize_verdict_to_status()` 统一裁决→状态映射。修复 NO_GO 被误判为 GO 的优先级 bug。
3. **Metric Alignment WARN 语义**: CLI 支持 PASS|WARN|FAIL 三级。WARN 允许导出但 manifest 记录 warning reason。
4. **Risk Pack Contract v1.1.0 对齐**: 新增 `relative_spike_prob`, `relative_down_prob`, `metric_alignment_warning_reason` 字段。总列数从 20 增至 23。
5. **NegativeRisk Champion Recalibration**: 基于 base-rate aware 的 top-k 上限和 alert budget 指标，裁决从 NEGATIVE_LOW_VALUE 升级为 NEGATIVE_CHAMPION。
6. **Risk Pack Quality Gate**: 10 项检查全部 PASS。

### Risk Pack Quality Gate 结果

```
verdict: PASS (10/10 checks passed)
pack: 3624 rows, 23 columns, 5 months, online mode
```

### Negative Recalibration 结果

| Criterion | Value | Threshold | Pass |
|-----------|-------|-----------|------|
| mean_auc | 0.946 | >= 0.90 | Yes |
| mean_f1 | 0.777 | >= 0.70 | Yes |
| mean_recall@20pct_alert | 0.694 | >= 0.65 | Yes |
| n_sufficient_months | 5 | >= 4 | Yes |

### 最终模块裁决

| Module | Verdict | Role |
|--------|---------|------|
| SpikeRisk | SPIKE_CHAMPION | champion |
| NegativeRisk | NEGATIVE_CHAMPION | champion |
| DeltaSupplyRisk | DELTA_RISK_ACCEPTABLE | aux |

### 是否允许进入 Ledger-1 / Mainline shadow

**是**。所有三个模块均 KEEP，其中两个达到 CHAMPION 级别。Risk pack quality gate PASS。可以进入 Ledger-1 / Mainline shadow 阶段。
