# Deep Realtime Model Scope — TrendKnightRT / DeepRealtimeTrendModel

**Status:** FROZEN
**Date:** 2026-07-03
**Phase:** DeepFinal-1

---

## 1. 本模型负责 (In Scope)

本模型是一个独立的深度学习实时电价趋势预测部件，负责以下功能：

- **realtime trend prediction** — 预测 realtime 电价主体趋势（24h 完整输出）
- **realtime delta prediction** — 预测 `delta = rt_actual - da_anchor`
- **SGDFNet residual correction** — 预测 `residual = rt_actual - sgdfnet_pred`，供后续融合使用
- **confidence score** — 每小时的预测置信度 [0, 1]
- **period-aware output** — 感知 1_8 / 9_16 / 17_24 时段差异

## 2. 本模型不负责 (Out of Scope)

以下功能由其他模块或系统负责，**不在本模型中实现**：

- **day-ahead prediction** — 日前预测由 LightGBM 日前模型负责
- **产差最终修正** — 产差模块 (production difference module) 是后续独立模块
- **spike/extreme price final correction** — 尖峰模块是后续独立模块
- **negative price final correction** — 负价模块是后续独立模块
- **ledger fusion** — 账本融合由主系统负责
- **mainline deployment** — 主链路部署由主系统负责
- **IntradayTracker** — 日内追踪器是主系统模块
- **RT916 / TimeMixer / TimesFM** — 这些是教师模型或主系统组件

## 3. 本模型必须输出给后续模块的字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `trend_rt_pred` | float[24] | 实时电价趋势预测（主输出） |
| `trend_delta_pred` | float[24] | delta 预测 = rt_pred - da_anchor |
| `residual_to_sgdfnet` | float[24] | SGDFNet 残差修正 = rt_pred - sgdfnet_pred |
| `confidence` | float[24] | 预测置信度 [0, 1] |
| `period` | str[24] | 时段标签 (1_8 / 9_16 / 17_24) |
| `residual_for_spike_module` | float[24] | 供尖峰模块使用的残差（= rt_actual - rt_pred，仅评估时有） |
| `residual_for_negative_module` | float[24] | 供负价模块使用的残差（= rt_actual - rt_pred，仅评估时有） |

**在线模式（无 y_true 时）：** `residual_for_spike_module` 和 `residual_for_negative_module` 不包含在输出中。这两个字段仅在评估/回测模式下可用。

## 4. 融合模式

本模型支持三种融合模式，可通过配置选择：

```
Mode A (delta mode):
  rt_pred = da_anchor + trend_delta_pred

Mode B (residual mode):
  rt_pred = sgdfnet_pred + residual_to_sgdfnet

Mode C (gated mode):
  rt_pred = learned_gate * modeB + (1 - learned_gate) * modeA
```

默认使用 Mode C（gated fusion），gate 由模型学习。

## 5. 后续模块接口

产差、尖峰、负价、ledger 模块将通过以下方式与本模型交互：

1. **读取预测输出** — 通过 `scripts/predict_realtime_deep_model.py` 生成的 CSV
2. **读取 model pack** — 通过 `scripts/export_realtime_model_pack.py` 导出的模型包
3. **使用残差** — 尖峰/负价模块使用 `residual_for_spike_module` / `residual_for_negative_module`
4. **使用置信度** — 后续模块可根据 `confidence` 决定是否覆盖预测

## 6. 数据 Contract

- **时间对齐**：严格使用 business_day / hour_business
- **Cutoff-safe**：不使用目标小时 realtime actual，不使用 cutoff 后不可见信息
- **输入维度**：24h day-level feature sequence（每行 24 个时间步）
- **输出维度**：24 个完整小时预测

## 7. 目标指标

| 级别 | 条件 |
|------|------|
| PASS | overall sMAPE_floor50 < 15 |
| STRONG | overall < 17 AND 9_16 < 22 |
| ACCEPTABLE | overall < 20 |
| NO-GO | overall >= 20 |

**可信参考**：SGDFNet corrected realtime overall capped RT sMAPE ≈ 16.59, 9_16 ≈ 21.19

---

*本文件冻结于 Phase DeepFinal-1。任何范围变更需在新 Phase 中提出。*
