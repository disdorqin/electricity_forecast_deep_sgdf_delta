# DeepFinal-4: Failure Diagnosis Report

**Date:** 2026-07-03
**Status:** COMPLETE — NO_RESIDUAL_SIGNAL — RECOMMEND ARCHIVE

---

## 1. DeepFinal-3B 结果复盘

| 模型 | 特征数 | SGDFNet | Test sMAPE | Best val sMAPE |
|------|--------|---------|-----------|----------------|
| DA anchor only | 0 | — | 26.69% | — |
| DeepFinal-1 (2 feat, fallback) | 2 | fallback | 26.76% | 31.11% |
| DeepFinal-2 (34 feat, fallback) | 34 | fallback | 26.69% | 31.18% |
| DeepFinal-3B (36 feat, real SGDFNet) | 36 | **real** | **26.61%** | **31.59%** |

即使接入真实 SGDFNet 预测，模型表现未改善。

## 2. 预测行为诊断结果

**诊断工具：** `scripts/diagnose_trendknight_failure.py`

| 诊断项 | 值 | 结论 |
|--------|-----|------|
| Corr(pred, da_anchor) | **0.9987** | 几乎完全复制 DA anchor |
| Corr(pred, SGDFNet) | **0.9974** | 几乎完全复制 SGDFNet |
| Residual true std | **113.38** | 真实 residual 标准差大 |
| Residual pred std | **0.31** | 预测 residual 标准差极小 |
| Residual std ratio | **0.27%** | 无 residual signal |
| Gate mean | N/A | gate 未追踪 |

**诊断裁决：**

```text
COPY_SGDFNET         ✅ — corr(pred, SGDFNet)=0.9974 > 0.98
NO_RESIDUAL_SIGNAL   ✅ — residual_pred_std / residual_true_std = 0.27% << 10%
NO_BETTER_THAN_ANCHOR ✅ — 26.61% vs 26.69% (仅好 0.08pp)
```

## 3. 是否 collapse/copy anchor

**是。** `corr(pred, da_anchor)=0.9987`，模型输出几乎等于 DA anchor。

根因：
- SGDFNet 的 `rt_hat` 预测与 `da_anchor` 高度相关（均为日前价格信号）
- 两个 head（delta head 预测 `rt-da_anchor`，residual head 预测 `rt-sgdfnet_pred`）目标几乎相同
- 模型学到的最优解就是将 delta/residual 都预测为 0，从而复制 anchor

## 4. Training Dynamics 诊断结果

**诊断工具：** `scripts/audit_training_dynamics.py`

| 诊断项 | 值 |
|--------|-----|
| Epochs run | 11 |
| Best epoch | **1** |
| Train loss trend | flat (44.60 → 44.38) |
| Val sMAPE trend | **up** (31.59 → 32.63) |
| Overfitting | 否（train loss 也没降）|

**裁决：NO_DEEP_SIGNAL**

从 epoch 1 开始 val sMAPE 就持续变差，说明模型没有学到任何有效模式。

## 5. Residual Baseline Lab 结果

**脚本：** `scripts/run_residual_baseline_lab.py`
**数据：** 2026-02，35088 train rows / 672 test rows

| 排名 | 模型 | Overall | 1_8 | 9_16 | 17_24 |
|------|------|---------|-----|------|-------|
| 1 | **DA_anchor** | **26.69%** | 26.39% | 28.72% | 25.09% |
| 2 | Mean_bias | 26.87% | — | — | — |
| 3 | SGDFNet | 26.88% | 26.42% | 29.11% | 25.23% |
| 4 | Period_bias | 26.95% | — | — | — |
| 5 | Hour_bias | 27.47% | — | — | — |
| 6 | HGB | 27.56% | — | — | — |
| 7 | Ridge | 27.71% | — | — | — |
| 8 | MLP | 28.27% | — | — | — |

**Verdict: NO_RESIDUAL_SIGNAL**

- 最好的模型是 DA anchor (26.69%)
- 没有任何 residual 模型超过 DA anchor
- HGB (27.56%) 比 SGDFNet (26.88%) 更差
- MLP (28.27%) 最差

## 6. 最终结论

```text
residual_baselines.py bug:     已修复 (baseline_hour_bias, baseline_period_bias)
run_residual_baseline_lab.py:  已完成并运行
Residual baseline leaderboard: DA_anchor (26.69%) 最高
HGB residual overall:          27.56% (比 DA anchor 差 0.87pp)
HGB < 23:                      NO (27.56%)
HGB < 20:                      NO (27.56%)
Residual signal exists:        NO
继续 residual-only deep:       NO — 标记为 DEFERRED_UNTIL_HGB_SIGNAL (永不触发)
趋势：                        ARCHIVE_DEEP_MODEL
```

建议：
1. **归档 TrendKnightRT 当前架构** — 在 36 个特征 + 真实 SGDFNet 下无法学到 residual
2. **不继续 residual-only deep** — 连 HGB 都比 DA anchor 差，residual signal 不存在
3. **主系统直接使用 DA anchor 作为实时预测** — 这是最准确的 baseline (26.69%)

## 6. 下一步建议

```text
1. 运行 residual baseline lab (scripts/run_residual_baseline_lab.py)
   比较 HGB / Ridge / MLP / simple bias

2. 如果 HGB residual overall < 23:
   residual signal exists，考虑 lightweight residual model
   替代 TrendKnightRT

3. 如果 HGB residual < 20:
   立刻把 HGB 作为 champion candidate

4. 如果所有 baseline >= 25:
   确认 residual signal 弱，归档深度模型部件
```
