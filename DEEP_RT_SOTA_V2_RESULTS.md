# DeepRT-SOTA v2 训练结果

## 日期：2026-07-04

### 实验结果

#### 实验 1: MLP (逐小时预测)
- **模型**: deep_rt_mlp
- **预测模式**: 逐小时 (hourly)
- **目标月**: 2026-02
- **指标**:
  - MAE: 191.14
  - RMSE: 242.37
  - sMAPE: 94.44
- **裁决**: NO_GO (sMAPE >> 20)

#### 实验 2: TCN (按日预测 24小时向量)
- **模型**: deep_rt_tcn
- **预测模式**: 按日 (day-level, 24-hour vector)
- **序列长度**: 14 天
- **目标月**: 2026-02
- **特征数**: 19
- **训练样本**: 1552
- **测试样本**: 6 (⚠️ 太少，需要修复)
- **指标**:
  - MAE: 111.74
  - RMSE: 139.49
  - sMAPE: 56.14
- **裁决**: NO_GO (sMAPE > 20, 但比 MLP 好很多)

### 改进方向

1. **增加测试样本**: 当前只有 6 个测试样本（2026-02），需要修复 NaN 处理
2. **添加风险特征**: 当前未使用风险特征
3. **添加预测侧特征**: 当前未使用 forecast-side features
4. **调参**: 调整 learning rate, hidden dim, num layers
5. **改进损失函数**: 使用 sMAPE loss 而非 Huber loss
6. **尝试其他模型**: GRU, Transformer
7. **增加训练数据**: 使用更多历史数据

### 下一步

1. 修复测试样本过少问题
2. 运行小炮实验矩阵（16个组合）
3. 如果 PASS_FAST，运行多月 backtest
4. 实现预测脚本、评估脚本、导出脚本
5. 生成报告

### 文件

- `scripts/train_deep_rt_sota_minimal.py`: MLP 训练脚本
- `scripts/train_deep_rt_sota_tcn.py`: TCN 训练脚本
- `artifacts/deep_rt_sota/minimal_exp/results.json`: MLP 结果
- `artifacts/deep_rt_sota/tcn_exp/results.json`: TCN 结果
