# Phase 10: Intraday Tracker Production Handoff

**Date:** 2026-07-03
**Status:** GO — 建议主融合接入

## 1. Phase 9 结果摘要

Phase 9 构建了 Intraday Adaptive Residual Tracker，使用同一天已观测到的 9_16 段真实实时电价残差来修正后续小时的 SGDFNet 预测。

核心结论：
- 仅适用于 INTRADAY 模式，禁止 FULL_DAY / day-ahead
- Overall: baseline 42.01 → corrected 41.43，改善 +0.59
- Cutoff 分组：cutoff 12: +0.92, cutoff 14: +1.14
- Bucket: normal +0.70, negative +0.49
- Verdict: LOW-WEIGHT

## 2. Phase 10 修正点

### Task A: Correction Pipeline 重构
修复了 `intraday_raw_correction` 命名/逻辑歧义。原来 `raw_correction` 已经乘过一次 model_weight，然后 guardrail 又乘一次 weight，导致双重加权。

新管线：
```
intraday_base_correction = 0.40 × mean_residual + 0.35 × ewm_residual + 0.25 × last_residual
intraday_model_weight = confidence × exp(-distance_decay × distance) × std_penalty
intraday_pre_guardrail_correction = clip(base × model_weight, ±max_abs_correction)
intraday_guardrail_weight = past_hour_weight × negative_price_weight × confidence_floor
intraday_final_correction = clip(pre_guardrail × guardrail_weight, ±max_abs_correction)
intraday_corrected_pred = sgdfnet_pred + intraday_final_correction
```

### Task B: Policy Gating
新增 `intraday_tracker_policy.py`，实现 cutoff gating 策略：

| 条件 | 决策 | Fusion Weight |
|------|------|---------------|
| mode != INTRADAY | DISABLED | 0 |
| n_observed < 3 | DISABLED | 0 |
| cutoff_hour < 10 | DISABLED | 0 |
| cutoff_hour < 12 | SHADOW_ONLY | 0 |
| confidence < 0.35 | SHADOW_ONLY | 0 |
| residual_std > 180 | SHADOW_ONLY | 0 |
| negative price risk | LOW_WEIGHT | 0.08 |
| confidence >= 0.55 AND cutoff >= 14 | HIGH_WEIGHT | 0.22 |
| default | LOW_WEIGHT | 0.12 |

## 3. Policy Gating 规则

Policy 按优先级顺序评估：
1. **Mode check**: 非 INTRADAY → DISABLED
2. **Observation check**: n_observed < 3 → DISABLED
3. **Cutoff floor**: cutoff < 10 → DISABLED; cutoff < 12 → SHADOW_ONLY
4. **Confidence floor**: confidence < 0.35 → SHADOW_ONLY
5. **Stability check**: residual_std > 180 → SHADOW_ONLY
6. **Negative risk**: da_anchor < 0 → LOW_WEIGHT (0.08)
7. **High confidence**: confidence >= 0.55 AND cutoff >= 14 → HIGH_WEIGHT (0.22)
8. **Default**: LOW_WEIGHT (0.12)

## 4. Fusion Weight 建议

| 阶段 | 决策 | Fusion Weight | 说明 |
|------|------|---------------|------|
| Shadow | SHADOW_ONLY | 0 | 仅记录，不实际修正 |
| Low-weight | LOW_WEIGHT | 0.12 | 保守接入 |
| Negative risk | LOW_WEIGHT | 0.08 | 负电价风险降低权重 |
| High-weight | HIGH_WEIGHT | 0.22 | 高置信度高cutoff |

## 5. Shadow-Only 上线阶段

第一阶段建议以 SHADOW_ONLY 模式运行 1-2 周：
- Tracker 正常运行并输出 correction
- 但不实际应用到融合预测
- 收集 shadow 数据用于验证 policy 决策准确性
- 确认 DISABLED/SHADOW_ONLY 的触发率合理

## 6. Low-Weight 上线阶段

Shadow 验证通过后，进入 LOW_WEIGHT 阶段：
- cutoff >= 12 且 confidence >= 0.35 的日子以 fusion_weight=0.12 接入
- 监控实际改善是否与回测一致
- 重点关注 negative bucket 是否稳定改善

## 7. Rollback 条件

以下任一条件触发 rollback 到 SHADOW_ONLY 或 DISABLED：
- 连续 3 天 intraday_gain_vs_baseline < -1.0
- negative_bucket_delta < -2.0（negative bucket 恶化超过 2.0）
- correction_clip_rate > 30%（修正被频繁截断说明模型不稳定）
- policy_disabled_rate > 50%（超过一半日子被 policy 禁用）

## 8. 不适用场景

- **FULL_DAY 模式**: 无当天观测数据
- **Day-ahead 预测**: 无当天实际电价
- **No observed actuals**: 无真实实时电价可用
- **Cutoff < 12**: 观测小时太少，修正不可靠
- **残差非平稳**: 如果 base model bias 剧烈变化，同一天内的残差模式可能不延续

## 9. 监控指标

| 指标 | 说明 | 健康范围 |
|------|------|----------|
| intraday_gain_vs_baseline | 修正后 vs 基线的 sMAPE 改善 | > 0.5 |
| negative_bucket_delta | 负电价 bucket 改善 | > -1.0 |
| correction_abs_mean | 修正绝对值均值 | 10~40 |
| correction_clip_rate | 修正被截断的比例 | < 20% |
| policy_disabled_rate | 被 policy 禁用的比例 | < 40% |
| shadow_hit_rate | SHADOW_ONLY 触发率 | < 30% |
| high_weight_rate | HIGH_WEIGHT 触发率 | 5~20% |
