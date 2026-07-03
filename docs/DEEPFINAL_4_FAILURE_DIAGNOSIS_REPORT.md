# DeepFinal-4: Failure Diagnosis Report

**Date:** 2026-07-03
**Status:** COMPLETE — NO_DEEP_SIGNAL

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

## 5. 最终结论

```text
TrendKnightRT 在当前配置下不值得继续优化。

原因：
1. 残差信号不存在 — residual_pred_std 仅为 true residual 的 0.27%
2. SGDFNet rt_hat 与 da_anchor 高度相关，模型无法从中提取新信息
3. 训练从 epoch 1 就无进展，不是超参问题
4. 36 个特征 + 真实 SGDFNet 仍只能达到 26.61%

建议：
├── USE_HGB_RESIDUAL     — 轻量模型可能足够
├── CONTINUE_DEEP_RESIDUAL — 仅在 HGB < 20% 时考虑
└── ARCHIVE_DEEP_MODEL   — 如果 HGB 也无法改善
```

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
