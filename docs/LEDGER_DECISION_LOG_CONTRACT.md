# Ledger Decision Log Contract

**Version**: 1.0  
**Date**: 2026-07-04  
**Status**: Draft

---

## 1. Purpose

This document defines the **contract** for the **ledger decision log**, which records every risk-aware guardrail action taken during shadow replay or production.

The decision log is the **single source of truth** for:
- Why a guardrail action was taken
- What action was taken
- How much adjustment was applied
- What the policy configuration was

---

## 2. Decision Log Schema

### 2.1 Core Fields (Required)

| Field | Type | Description | Online Mode | Eval Mode |
|-------|------|-------------|-------------|-----------|
| `business_day` | `datetime64[ns]` | Business day (see `business_time.py`) | ✅ | ✅ |
| `hour_business` | `int64` (1-24) | Business hour (see `business_time.py`) | ✅ | ✅ |
| `target_month` | `str` ("YYYY-MM") | Target month for prediction | ✅ | ✅ |
| `base_pred` | `float64` | Base prediction (DA anchor, SGDFNet, etc.) | ✅ | ✅ |
| `risk_adjusted_pred` | `float64` | Prediction after guardrail adjustment | ✅ | ✅ |
| `negative_prob` | `float64` [0, 1] | Negative risk probability | ✅ | ✅ |
| `negative_risk_score` | `float64` [0, 1] | Negative risk score | ✅ | ✅ |
| `spike_prob` | `float64` [0, 1] | Spike risk probability | ✅ | ✅ |
| `spike_risk_score` | `float64` [0, 1] | Spike risk score | ✅ | ✅ |
| `deviation_down_prob` | `float64` [0, 1] | Delta supply down probability | ✅ | ✅ |
| `deviation_up_prob` | `float64` [0, 1] | Delta supply up probability | ✅ | ✅ |
| `negative_triggered` | `bool` | Whether negative guardrail triggered | ✅ | ✅ |
| `spike_triggered` | `bool` | Whether spike guardrail triggered | ✅ | ✅ |
| `delta_supply_triggered` | `bool` | Whether delta supply guardrail triggered | ✅ | ✅ |
| `policy_id` | `str` | Policy configuration ID (e.g., "negative_thresh06_spike_thresh07_blend02") | ✅ | ✅ |
| `action_taken` | `str` | Action taken (`none`, `alert_only`, `soft_negative_blend`, `soft_spike_blend`, `weight_adjust`) | ✅ | ✅ |
| `adjustment_amount` | `float64` | Amount of adjustment applied (`risk_adjusted_pred - base_pred`) | ✅ | ✅ |
| `reason_codes` | `str` | Pipe-separated reason codes (e.g., "NEGATIVE_HIGH_RISK\|SPIKE_HIGH_RISK") | ✅ | ✅ |

### 2.2 Optional Fields (Eval Mode Only)

| Field | Type | Description | Online Mode | Eval Mode |
|-------|------|-------------|-------------|-----------|
| `y_true` | `float64` | Actual price (for evaluation **only**) | ❌ **Forbidden** | ✅ |
| `error` | `float64` | Prediction error (`y_true - risk_adjusted_pred`) | ❌ **Forbidden** | ✅ |
| `base_error` | `float64` | Base prediction error (`y_true - base_pred`) | ❌ **Forbidden** | ✅ |
| `error_reduction` | `float64` | Error reduction from guardrail (`base_error - error`) | ❌ **Forbidden** | ✅ |

---

## 3. Key Constraints

### 3.1 Key Uniqueness

The combination of `(business_day, hour_business, target_month)` **MUST** be **unique**.

**Rationale**:
- Each hour in each target month has exactly one base prediction.
- Each hour can have exactly one guardrail action per policy.
- Duplicate keys indicate a bug in data loading or policy application.

**Validation**:
```python
key_cols = ["business_day", "hour_business", "target_month"]
assert not df[key_cols].duplicated().any(), "Duplicate keys found in decision log"
```

---

### 3.2 Online Mode vs Eval Mode

| Constraint | Online Mode | Eval Mode |
|-----------|-------------|-----------|
| `y_true` allowed? | ❌ **No** | ✅ Yes |
| `error` allowed? | ❌ **No** | ✅ Yes |
| `base_error` allowed? | ❌ **No** | ✅ Yes |
| `error_reduction` allowed? | ❌ **No** | ✅ Yes |
| Purpose | Real-time guardrail application | Backtest / shadow replay evaluation |

**Online mode** = Production or real-time prediction.  
**Eval mode** = Backtest, shadow replay, or offline evaluation.

**Critical**: `y_true` **MUST NOT** be used in guardrail decision-making. It is only for evaluation.

---

### 3.3 Reason Codes (Required)

Every guardrail action **MUST** have at least one reason code.

**Valid reason codes**:
- `NEGATIVE_HIGH_RISK`: Negative risk score or probability >= threshold.
- `SPIKE_HIGH_RISK`: Spike risk score or probability >= threshold.
- `DELTA_SUPPLY_DOWN_RISK`: Delta supply down probability >= threshold.
- `DELTA_SUPPLY_UP_RISK`: Delta supply up probability >= threshold.
- `NO_TRIGGER`: No risk triggered, no action taken.

**Format**: Pipe-separated string (e.g., `"NEGATIVE_HIGH_RISK|SPIKE_HIGH_RISK"`).

**Validation**:
```python
assert "reason_codes" in df.columns, "Missing reason_codes column"
assert df["reason_codes"].notna().all(), "Found NaN in reason_codes"
```

---

### 3.4 Policy Version Recording

The `policy_id` field **MUST** record the exact policy configuration used.

**Recommended format**:
```
{negative_action}_{negative_threshold}_{spike_action}_{spike_threshold}_{blend_weight}
```

**Example**:
```
soft_negative_blend_06_soft_spike_blend_07_02
```

**Purpose**:
- Reproducibility: Exact policy can be reconstructed from `policy_id`.
- Comparison: Different policies can be compared using `policy_id`.
- Debugging: If a guardrail action seems wrong, check `policy_id`.

---

## 4. Decision Log Example

### 4.1 Online Mode (No `y_true`)

```csv
business_day,hour_business,target_month,base_pred,risk_adjusted_pred,negative_prob,negative_risk_score,spike_prob,spike_risk_score,deviation_down_prob,deviation_up_prob,negative_triggered,spike_triggered,delta_supply_triggered,policy_id,action_taken,adjustment_amount,reason_codes
2026-01-15,14,2026-01,285.50,285.50,0.12,0.15,0.08,0.10,0.05,0.03,False,False,False,soft_negative_blend_06_soft_spike_blend_07_02,none,0.0,NO_TRIGGER
2026-01-15,18,2026-01,320.40,320.40,0.25,0.28,0.75,0.78,0.15,0.10,False,True,False,soft_negative_blend_06_soft_spike_blend_07_02,soft_spike_blend,36.08,SPIKE_HIGH_RISK
2026-01-20,03,2026-01,150.20,120.16,0.68,0.72,0.12,0.15,0.08,0.05,True,False,False,soft_negative_blend_06_soft_spike_blend_07_02,soft_negative_blend,-30.04,NEGATIVE_HIGH_RISK
```

---

### 4.2 Eval Mode (With `y_true`)

```csv
business_day,hour_business,target_month,base_pred,risk_adjusted_pred,negative_prob,negative_risk_score,spike_prob,spike_risk_score,deviation_down_prob,deviation_up_prob,negative_triggered,spike_triggered,delta_supply_triggered,policy_id,action_taken,adjustment_amount,reason_codes,y_true,error,base_error,error_reduction
2026-01-15,14,2026-01,285.50,285.50,0.12,0.15,0.08,0.10,0.05,0.03,False,False,False,soft_negative_blend_06_soft_spike_blend_07_02,none,0.0,NO_TRIGGER,288.30,2.80,2.80,0.0
2026-01-15,18,2026-01,320.40,356.48,0.25,0.28,0.75,0.78,0.15,0.10,False,True,False,soft_negative_blend_06_soft_spike_blend_07_02,soft_spike_blend,36.08,SPIKE_HIGH_RISK,410.20,53.72,-89.80,36.08
2026-01-20,03,2026-01,150.20,120.16,0.68,0.72,0.12,0.15,0.08,0.05,True,False,False,soft_negative_blend_06_soft_spike_blend_07_02,soft_negative_blend,-30.04,NEGATIVE_HIGH_RISK,85.40,-34.76,64.80,30.04
```

---

## 5. Export Format

### 5.1 CSV Export

**File name**: `decision_log.csv`

**Required columns** (online mode):
- All fields in Section 2.1 (no `y_true`, `error`, etc.)

**Optional columns** (eval mode only):
- `y_true`, `error`, `base_error`, `error_reduction`

**Encoding**: UTF-8  
**Separator**: `,` (comma)  
**Index**: No index column

---

### 5.2 JSON Export (Metadata)

**File name**: `decision_log_metadata.json`

**Content**:
```json
{
  "n_rows": 744,
  "target_months": ["2026-01", "2026-02"],
  "policy_id": "soft_negative_blend_06_soft_spike_blend_07_02",
  "online_mode": false,
  "has_y_true": true,
  "export_timestamp": "2026-07-04T12:00:00Z",
  "columns": ["business_day", "hour_business", ..., "error_reduction"]
}
```

---

## 6. Validation Rules

| Rule | Validation |
|-------|-------------|
| **Key uniqueness** | `(business_day, hour_business, target_month)` must be unique |
| **No `y_true` in online mode** | If `online_mode=True`, `y_true` column must not exist |
| **Reason codes required** | `reason_codes` must not be null or empty |
| **Probability in [0, 1]** | `negative_prob`, `spike_prob`, `deviation_down_prob`, `deviation_up_prob` must be in [0, 1] or NaN |
| **Adjustment consistency** | `risk_adjusted_pred` must equal `base_pred + adjustment_amount` (within floating-point tolerance) |
| **Action matches trigger** | If `negative_triggered=True`, `reason_codes` must contain `NEGATIVE_HIGH_RISK` |

---

## 7. Anti-Patterns (Forbidden)

❌ **DO NOT**:
1. Use `y_true` in guardrail decision-making.
2. Hard-code corrections based on `y_true`.
3. Omit `reason_codes`.
4. Have duplicate keys.
5. Use unbounded arbitrary corrections (must be based on `floor_value` or `spike_floor`).
6. Modify `y_true` or actuals in any way.

---

## 8. Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-07-04 | Initial draft. |

---

## 9. References

- `docs/LEDGER_1_SCOPE.md`: Ledger-1 scope and responsibilities.
- `models/deep_sgdf_delta/risk_guardrail_policy.py`: Guardrail policy implementation.
- `models/deep_sgdf_delta/business_time.py`: Business time alignment.
- `docs/RISK_MODULE_SELECTION_CRITERIA.md`: Risk module selection criteria.
