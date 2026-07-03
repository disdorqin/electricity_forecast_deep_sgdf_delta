# SGDFNet Prediction Input Contract

## 用途

定义 SGDFNet 实时预测文件与 TrendKnightRT 之间的数据合约。

## 文件格式

- **格式**: CSV
- **编码**: UTF-8 或 GBK
- **必需列**: 时间戳列 + 预测列（见下方映射）

## 列名自动识别

### 时间戳列（任选其一）

| 列名 | 说明 |
|------|------|
| `ds` | 标准名称 |
| `timestamp` | 常见替代 |
| `time` | 通用名称 |
| `时刻` | 山东数据中文名 |

### 预测列（任选其一）

| 列名 | 说明 |
|------|------|
| `sgdfnet_pred` | 标准名称 |
| `pred` | 通用名称 |
| `prediction` | 通用名称 |
| `y_pred` | ML 通用名称 |
| `rt_pred` | 实时预测名称 |

### 可选列

| 列名 | 说明 |
|------|------|
| `business_day` | 业务日对齐（如缺失则自动计算） |
| `hour_business` | 业务小时 1-24（如缺失则自动计算） |

## 输出

Loader 输出 DataFrame 包含：

```
ds, business_day, hour_business, sgdfnet_pred
```

## Coverage 要求

| 场景 | 最低 Coverage | 行为 |
|------|---------------|------|
| 正式训练 | >= 95% | 允许训练 |
| 预测/推理 | >= 90% | 允许预测（缺失行使用 da_anchor fallback）|
| 冒烟测试 | 无要求 | 允许 fallback |

- Coverage < 95% 时正式训练直接 fail
- 不允许将 da_anchor 当作真实 SGDFNet 预测

## 去重

按 `(business_day, hour_business)` 去重，保留第一条记录。

## 输出文件

```text
sgdfnet_prediction_coverage.json
sgdfnet_prediction_coverage.md
```
