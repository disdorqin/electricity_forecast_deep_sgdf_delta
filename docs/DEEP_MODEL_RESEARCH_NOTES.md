# Deep Model Research Notes -- TrendKnight-X

> 7 research directions evaluated for the electricity price forecasting project.
> Goal: sMAPE_floor50 < 15, with particular improvement on the 9_16 period.
> All decisions must be cutoff-safe (no future data leakage past D-1).

---

## 1. TimeMixer / Multiscale Mixing

**Paper:** "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting" (Wang et al., 2024)

### 核心思想

TimeMixer 的核心是在不同时间尺度上分别建模，然后通过混合机制融合。它将时间序列分解为多个尺度（如小时级、天级、周级），每个尺度用独立的子模型处理，最后通过可学习的混合权重组合。

关键创新：
- ** multiscale decomposition **：通过下采样（average pooling）生成多个尺度的序列
- ** PDM (Period Decomposition Module) **：在频域做周期分解
- ** 混合方式 **：从粗到细逐层融合，而非简单拼接

### 适合本项目的部分

- **电价具有天然的多尺度结构**：小时级波动（供需瞬时变化）、日级模式（工作日/周末）、周级趋势（季节性）。TimeMixer 的架构天然匹配这种结构。
- **9_16 时段的改善潜力**：光伏出力在 9-16 时段产生剧烈的日内波动，单一尺度模型难以同时捕捉日内尖峰和日间趋势。多尺度分解可以让模型在不同分辨率下分别处理。
- **已实现**：`model_v3.py` 中的 `MultiscaleDecomposer` 已经实现了 trend/seasonal/shock 三分解，这是 TimeMixer 思想的简化版。

### 不适合本项目的部分

- **PDM 模块过于复杂**：频域分解需要 FFT，对于 24 小时序列（只有 24 个点）频域分辨率太低，不如直接在时域做分解。
- **计算开销**：完整的 TimeMixer 有多个尺度的独立子模型，参数量容易超过 500k 限制。
- **数据量不足**：山东电价数据只有约 2 年，多尺度模型容易在小数据集上过拟合。

### 是否进入 TrendKnight-X

**YES -- 简化版已实现。** `MultiscaleDecomposer` 在 `model_v3.py` 中实现了 3 尺度分解（trend/seasonal/shock），通过可学习的权重组合。这比完整 TimeMixer 轻量得多，但保留了核心思想。

### 如何实现

```python
# model_v3.py: MultiscaleDecomposer
class MultiscaleDecomposer(nn.Module):
    def __init__(self, hidden_dim):
        # 3 个 1D 卷积核，不同 kernel_size 对应不同尺度
        self.trend_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3)
        self.seasonal_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.shock_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        # 可学习混合权重
        self.mix_weights = nn.Parameter(torch.ones(3) / 3.0)

    def forward(self, h):
        trend = self.trend_conv(h)      # 大核 -> 长期趋势
        seasonal = self.seasonal_conv(h) # 中核 -> 周期波动
        shock = self.shock_conv(h)       # 小核 -> 瞬时冲击
        w = F.softmax(self.mix_weights, dim=0)
        return w[0]*trend + w[1]*seasonal + w[2]*shock
```

**Ablation 验证**：`run_trendknight_x_ablation.py` 中 `v3_fast_tcn`（无 multiscale）vs `v3_multiscale_tcn`（有 multiscale）直接对比。

---

## 2. TimesNet / 2D Temporal Variation

**Paper:** "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis" (Wu et al., 2023)

### 核心思想

TimesNet 将 1D 时间序列转换为 2D 张量，利用不同周期将序列折叠成不同形状的 2D 矩阵，然后用 2D 卷积捕捉周期内和周期间的交互模式。

关键步骤：
1. **FFT 找主周期**：通过快速傅里叶变换找到 top-k 显著周期
2. **1D -> 2D 折叠**：按每个周期将序列折叠成 2D 矩阵
3. **2D 卷积**：用 Inception 块在 2D 空间提取特征
4. **堆叠多层**：多层 TimesBlock 逐层提取更复杂的时序模式

### 适合本项目的部分

- **电价有强周期性**：24 小时日周期、7 天周周期、365 天年周期。TimesNet 的 FFT 找周期步骤可以自动发现这些。
- **日内模式可视化**：将 24 小时折叠成 2D 后，行内卷积捕捉相邻小时的关联（如 9-10 点的光伏爬坡），行间卷积捕捉日与日的差异。

### 不适合本项目的部分

- **序列太短**：我们的输入窗口是 7 天 x 24 小时 = 168 个点。FFT 在这么短的序列上分辨率很低，可能只能分辨出 24 小时周期，更长的周期（周、月）需要更长的窗口。
- **2D 折叠的边界效应**：168 个点按 24 折叠成 7x24 矩阵，只有 7 行，2D 卷积的感受野在行方向上几乎没用。
- **参数量爆炸**：Inception 块有多条并行卷积分支，容易超出 500k 参数限制。
- **cutoff-safe 风险**：FFT 需要完整周期的数据，如果 D-1 的数据不完整（例如缺少最后几小时），FFT 会产生虚假周期。

### 是否进入 TrendKnight-X

**NO -- 不适合本项目数据规模。** 但 TimesNet 的 "按周期折叠" 思想有一个简化版可以借鉴：在 `dataset_v3.py` 中，我们将 24 小时按 period 分成 3 段（1_8, 9_16, 17_24），这本质上是一种粗粒度的 "周期折叠"。

### 如果未来数据量增大

如果积累 5+ 年数据，可以重新评估 TimesNet。届时：
- 输入窗口可以扩大到 30 天，FFT 分辨率足够
- 按 7 天折叠成 week x day 矩阵，2D 卷积可以捕捉周间模式
- 需要实现 cutoff-safe FFT：只用 D-1 及之前的数据

---

## 3. PatchTST / Patch-Based Efficient Transformer

**Paper:** "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (Nie et al., 2023)

### 核心思想

PatchTST 将时间序列切分成固定大小的 patch（如每 16 个点一个 patch），每个 patch 作为一个 token 输入 Transformer。这比逐点输入高效得多，同时保留了局部时序模式。

关键设计：
- **Patch 大小**：通常 16 或 8，trade-off  between 局部精度和序列长度
- **Channel-independence**：每个变量独立处理，不建模变量间交叉
- **Masked autoencoder pretraining**：用 masking 做自监督预训练

### 适合本项目的部分

- **24 小时 -> patch 的降采样**：如果 patch_size=8，24 小时变成 3 个 patch，正好对应 3 个 period（1_8, 9_16, 17_24）。这跟我们的 period-aware 设计天然吻合。
- **效率**：Transformer 的 O(n^2) 复杂度在 24 点上不是问题，但 patch 化后可以处理更长的 lookback 窗口（如 30 天 = 720 点 -> 90 个 patch）。
- **Channel-independence 合理**：我们的输入特征已经通过 feature engineering 编码了变量间关系，不需要 Transformer 再学一遍。

### 不适合本项目的部分

- **序列太短，Transformer 优势不明显**：24 个点上 TCN 的因果卷积已经足够，Transformer 的 self-attention 在短序列上没有效率优势。
- **预训练不现实**：PatchTST 的核心优势来自大规模预训练，我们没有足够的数据做有意义的 pretraining。
- **可解释性差**：Transformer 的 attention 权重难以解释，而业务方需要理解 "为什么预测这个价格"。

### 是否进入 TrendKnight-X

**PARTIAL -- Patch 思想已融入。** `dataset_v3.py` 中的 24h 序列本质上就是 3 个 patch（每段 8 小时）。但我们用的是 TCN 而非 Transformer 做 backbone，因为 TCN 在短序列上更高效且更容易控制因果性。

### 如何实现

当前 `model_v3.py` 的 `PeriodBranch` 可以看作一种 patch-based 处理：
```python
# 将 24h 特征按 period 分段处理
# 每段 = 一个 "patch"
for period_id in [0, 1, 2]:  # 1_8, 9_16, 17_24
    period_features = features_24h[:, period_start:period_end, :]
    period_embedding = self.period_embed(period_id)
    period_repr = self.backbone(period_features) + period_embedding
```

如果未来需要处理更长的序列（如 7 天 x 24 小时），可以考虑真正的 PatchTST：
- patch_size=24（每天一个 patch）
- 7 天 -> 7 个 patch
- 用 lightweight Transformer 处理 7 个 patch

---

## 4. iTransformer / Variable-Centric Attention

**Paper:** "iTransformer: Transformers are the Better Choice for Effective Multivariate Time Series Forecasting" (Liu et al., 2024)

### 核心思想

传统 Transformer 对时间维度做 attention（每个时间步是一个 token），iTransformer 反过来对变量维度做 attention（每个变量是一个 token）。这使得模型可以更好地捕捉变量间的依赖关系。

关键设计：
- **变量 = token**：每个变量的整个时间序列作为一个 token
- **Feed-forward = 时间建模**：每个变量的时序模式由独立的 FFN 处理
- **Attention = 变量交互**：self-attention 建模变量间的依赖

### 适合本项目的部分

- **多节点电价的空间相关性**：如果未来扩展到多节点预测，iTransformer 的变量级 attention 可以自动学习节点间的电价传导关系。
- **特征交互**：我们的输入有 ~40 个特征（da_anchor, 历史电价, 负荷预测, 新能源出力...），变量级 attention 可以自动发现哪些特征对当前预测最重要。

### 不适合本项目的部分

- **当前是单节点预测**：我们只预测山东节点电价，不存在多变量间的依赖需要建模。特征工程已经编码了必要的信息。
- **40 个特征太多**：如果每个特征是一个 token，40 个 token 的 attention 矩阵是 40x40，但其中大部分关系是噪声。特征选择比 attention 更有效。
- **破坏时序因果性**：变量级 attention 不区分时间先后，需要额外的因果掩码。这增加了复杂度但没有明显收益。

### 是否进入 TrendKnight-X

**NO -- 不适合当前单节点场景。** 但如果未来扩展到多节点联合预测（如山东 + 山西 + 广东），iTransformer 的架构值得认真考虑。

### 如果未来多节点扩展

```python
# 每个节点的特征序列作为一个 token
class iTransformerNode(nn.Module):
    def __init__(self, n_nodes, hidden_dim):
        self.node_ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(T, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, T))
            for _ in range(n_nodes)
        ])
        self.cross_node_attention = nn.MultiheadAttention(hidden_dim, nhead=4)

    def forward(self, x):
        # x: [B, n_nodes, T]
        # 1. Per-node FFN (时间建模)
        node_reprs = [self.node_ffn[i](x[:, i, :]) for i in range(n_nodes)]
        # 2. Cross-node attention (空间交互)
        node_tokens = torch.stack(node_reprs, dim=1)  # [B, n_nodes, hidden]
        attn_out, _ = self.cross_node_attention(node_tokens, node_tokens, node_tokens)
        return attn_out
```

---

## 5. N-HiTS / Hierarchical Interpolation

**Paper:** "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting" (Challu et al., 2023)

### 核心思想

N-HiTS 用多层 block，每层在不同的时间粒度上预测，然后通过分层插值组合。低层处理高频细节（小时级），高层处理低频趋势（天级/周级）。

关键设计：
- **多速率采样**：每层用不同的 pooling 步长处理输入
- **分层插值输出**：每层输出一个低维预测，通过插值扩展到目标长度
- **Backcast + Forecast**：每层同时输出 "已解释部分"（backcast）和 "预测部分"（forecast），下一层只处理残差

### 适合本项目的部分

- **24 小时输出的天然匹配**：N-HiTS 的分层插值非常适合 "输入 7 天 -> 输出 24 小时" 的场景。高层捕捉日级趋势，低层捕捉小时级细节。
- **Backcast 机制**：每层解释一部分信号，下一层处理残差。这跟我们的 residual distillation（`v3_teacher_residual`）思想一致。
- **参数高效**：每层输出低维向量再插值，比直接输出 24 维的参数少得多。

### 不适合本项目的部分

- **层次结构已隐含在 V3 中**：`MultiscaleDecomposer` 已经实现了类似的多尺度分解。N-HiTS 的 hierarchical interpolation 在数学上等价于我们的 trend + seasonal + shock 分解。
- **插值引入平滑**：分层插值倾向于产生平滑的预测，但电价在 9_16 时段有剧烈的非平滑波动（光伏骤降），平滑预测反而有害。
- **实现复杂度**：完整的 N-HiTS 有多层 block，每层有不同的采样率和插值方式，代码复杂度高且调试困难。

### 是否进入 TrendKnight-X

**PARTIAL -- 分层思想已融入。** `DayDecoderHead` 在 `model_v3.py` 中将 backbone 输出解码为 24 小时预测，本质上是一种 "从低维表示插值到完整输出" 的操作。但我们用的是线性解码器而非 N-HiTS 的分层插值，因为后者在 24 小时输出上没有显著优势。

### 如何实现

当前 `DayDecoderHead` 的简化实现：
```python
class DayDecoderHead(nn.Module):
    def __init__(self, hidden_dim, output_hours=24):
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_hours),  # 直接输出 24h
        )

    def forward(self, h):
        return self.decoder(h)  # [B, 24]
```

如果未来需要更好的 24h 输出建模，可以考虑 N-HiTS 风格的分层解码：
```python
# 高层：日级趋势 (1 个值 -> 24h 常数)
# 中层：周期模式 (3 个值 -> 8h 分段常数)
# 低层：小时细节 (24 个值 -> 逐小时修正)
```

---

## 6. Mixture-of-Experts / Gating

**Paper:** "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (Shazeer et al., 2017)
**Related:** "Gating Networks" in various transformer MoE variants

### 核心思想

MoE 将模型分成多个 "expert"，每个 expert 是一个独立的子网络。一个 gating network 根据输入决定激活哪些 expert。这样可以在不增加推理计算量的情况下大幅增加模型容量。

关键设计：
- **Top-k gating**：每个输入只激活 top-k 个 expert（通常 k=1 或 2）
- **Load balancing loss**：防止所有输入都涌向同一个 expert
- **Expert specialization**：不同 expert 自然学会处理不同类型的输入

### 适合本项目的部分

- **电价分布的多模态性**：电价在不同时段、不同季节有截然不同的分布。MoE 可以让不同的 expert 分别处理 1_8（低谷）、9_16（光伏高峰）、17_24（晚高峰）。
- **Teacher fusion 的天然框架**：`v3_teacher_moe` 已经实现了这个想法。每个 teacher 可以看作一个 expert，gating network 根据时段和置信度选择最合适的 teacher。
- **参数效率**：3 个 expert 各 32k 参数 = 96k 总参数，但每次推理只用 32k。比单个 96k 参数的模型更灵活。

### 不适合本项目的部分

- **数据量不足以训练多个 expert**：每个 expert 需要足够的数据来学习自己的子分布。如果 9_16 时段的 spike 样本很少（可能只占 5%），对应的 expert 可能训练不充分。
- **Load balancing 困难**：如果 3 个时段的数据量差异很大（1_8 和 17_24 各 8 小时，9_16 也是 8 小时但波动更大），gating network 可能学不好。
- **推理延迟**：虽然理论上只激活 1 个 expert，但实际实现中需要计算所有 expert 的 gating 权重，增加了推理时间。

### 是否进入 TrendKnight-X

**YES -- 已实现为 v3_teacher_moe。** `model_v3.py` 中的 `TeacherFusionGate` 实现了 MoE 风格的 teacher 融合。gating network 根据 backbone 特征和 teacher 可用性计算权重。

### 如何实现

```python
class TeacherFusionGate(nn.Module):
    def __init__(self, hidden_dim, teacher_input_dim):
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + teacher_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, teacher_input_dim),
        )
        self.teacher_mask = None  # 动态 masking

    def forward(self, h_bb, teacher_preds, teacher_mask):
        # h_bb: [B, hidden_dim]
        # teacher_preds: [B, T, 24]
        # teacher_mask: [B, T] (1=available, 0=unavailable)

        gate_input = torch.cat([h_bb, teacher_preds.mean(dim=-1)], dim=-1)
        gate_logits = self.gate(gate_input)  # [B, T]

        # Mask unavailable teachers
        gate_logits = gate_logits.masked_fill(~teacher_mask.bool(), -1e9)
        gate_weights = F.softmax(gate_logits, dim=-1)  # [B, T]

        # Weighted combination
        fused = torch.einsum('bt,bth->bh', gate_weights, teacher_preds)  # [B, 24]
        return fused
```

**Ablation 验证**：`v3_teacher_residual`（简单残差融合）vs `v3_teacher_moe`（MoE 门控融合）直接对比。

---

## 7. Electricity Price Spike / Volatility Modeling

**Context:** 这不是单篇论文，而是电力市场价格预测领域的核心挑战。参考：
- "A review and discussion of decomposition-based hybrid methods for time series forecasting"
- "Deep learning for electricity price forecasting: A review"
- 本项目内的 spike detector 模块设计文档

### 核心思想

电价尖峰（spike）是指价格突然飙升到正常水平的 3-10 倍（如从 300 元/MWh 飙升到 3000+ 元/MWh）。尖峰的成因包括：
- 发电机组突然停机
- 输电线路故障
- 极端天气导致新能源出力骤降
- 需求侧突发事件

尖峰建模的关键挑战：
1. **频率低**：尖峰只占全部小时的 1-3%，模型容易忽略
2. **幅度大**：尖峰的价格偏差是正常小时的 10 倍以上
3. **不可预测性**：很多尖峰由突发事件引起，历史数据中没有前兆信号
4. **sMAPE 影响**：由于 sMAPE_floor50 的分母有 floor=50 的保护，尖峰对 sMAPE 的影响被部分抑制，但仍然显著

### 适合本项目的部分

- **直接关系 sMAPE_floor50 < 15 的目标**：如果尖峰处理不好，即使正常小时预测很准，整体 sMAPE 也会被拉高。
- **已有 spike detector 模块**：项目内的 spike detector 作为独立模块处理尖峰，TrendKnight-X 只需要输出 `residual_for_spike` 供其使用。
- **shock_sensitivity 头**：`model_v3.py` 的 `ShockSensitivityHead` 已经实现了对波动性的显式建模。

### 不适合本项目的部分

- **端到端学习尖峰不现实**：由于尖峰太稀少，让 TrendKnight-X 自己学会预测尖峰几乎不可能。更好的策略是让趋势模型预测 "正常价格"，然后由专门的 spike detector 处理残差中的尖峰。
- **负价和尖峰是相反方向**：尖峰是价格异常高，负价是价格低于零。一个模型很难同时处理两个方向。这也是为什么我们有独立的 `spike_module` 和 `negative_module`。

### 是否进入 TrendKnight-X

**YES -- 通过辅助头和残差输出。** TrendKnight-X 不直接预测尖峰，而是：
1. `ShockSensitivityHead` 输出波动性评分，供 spike detector 使用
2. 输出 `residual_for_spike`（在 eval pack 中），供 spike module 训练
3. `trend_confidence` 低时，spike detector 获得更高优先级

### 如何实现

```python
# model_v3.py: ShockSensitivityHead
class ShockSensitivityHead(nn.Module):
    def __init__(self, hidden_dim):
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # 输出 [0, 1]
        )

    def forward(self, h):
        return self.head(h).squeeze(-1)  # [B]

# 训练时的辅助 loss
# shock_sensitivity 应该在高波动小时接近 1，在平稳小时接近 0
shock_target = (y_true.abs() > y_true.median() * 2).float()
shock_loss = F.binary_cross_entropy(shock_pred, shock_target)
```

**Spike module 接口**：
```python
# TrendKnight-X 输出
fusion_row = {
    "trend_pred": 320.5,
    "shock_sensitivity": 0.82,  # 高波动信号
    "trend_confidence": 0.45,   # 低置信度
}

# Spike detector 使用
if fusion_row["shock_sensitivity"] > 0.7:
    spike_correction = SpikeModule.predict(features, residual_for_spike)
    final_price = fusion_row["trend_pred"] + spike_correction
```

---

## Summary: What's in TrendKnight-X

| Direction | Status | Implementation | Ablation Candidate |
|-----------|--------|----------------|--------------------|
| TimeMixer / Multiscale | ADOPTED | `MultiscaleDecomposer` in `model_v3.py` | `v3_fast_tcn` vs `v3_multiscale_tcn` |
| TimesNet / 2D Variation | REJECTED | Too complex for 24h sequences | N/A |
| PatchTST / Patch-based | PARTIAL | Period-based segmentation in `dataset_v3.py` | Implicit in all v3 candidates |
| iTransformer / Variable-centric | REJECTED | Single-node prediction, no variable interaction needed | N/A |
| N-HiTS / Hierarchical | PARTIAL | `DayDecoderHead` as simplified hierarchical decode | Implicit in all v3 candidates |
| MoE / Gating | ADOPTED | `TeacherFusionGate` in `model_v3.py` | `v3_teacher_residual` vs `v3_teacher_moe` |
| Spike / Volatility | ADOPTED | `ShockSensitivityHead` + residual output | Evaluated via bucket_metrics.csv |

## Key Engineering Decisions

1. **TCN over Transformer**: For 24h sequences, TCN's causal convolution is more efficient and easier to make cutoff-safe than Transformer's self-attention.

2. **Simplified multiscale over full TimeMixer**: 3-scale decomposition (trend/seasonal/shock) captures the essential multi-resolution structure without the complexity of FFT-based period detection.

3. **Separate spike module over end-to-end spike prediction**: Spikes are too rare and too large for the deep model to learn. Better to output clean trend + residual, let a specialized module handle the tail.

4. **Teacher residual distillation over direct teacher copying**: The student learns to correct the teacher's mistakes, not to replicate the teacher. This is more robust when the teacher is wrong.

5. **Period-aware design over uniform treatment**: The 9_16 period has fundamentally different dynamics (solar peak, high volatility) than 1_8 or 17_24. Explicit period segmentation helps the model allocate capacity where it's needed most.
