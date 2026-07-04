# Ledger-1 Scope

## Phase Overview

**Phase**: Ledger-1: Risk-aware Guardrail / Dynamic Fusion Shadow Replay  
**Repository**: `disdorqin/electricity_forecast_deep_sgdf_delta`  
**Goal**: Validate whether risk pack can improve final prediction quality in shadow replay, especially for negative bucket, spike bucket, and high-risk hours.

---

## What This Phase Covers

### Core Responsibilities

| Component | Description |
|-----------|-------------|
| Risk pack ingestion | Load and validate risk feature pack (v1.1.0) |
| Risk-aware guardrail simulation | Apply guardrail policies to adjust base predictions |
| Negative guardrail shadow replay | Test negative price guardrail |
| Spike guardrail shadow replay | Test spike price guardrail |
| Fusion weight adjustment simulation | Simulate dynamic fusion weight adjustment |
| Alert-only evaluation | Evaluate risk alerts without modifying predictions |
| Bucket-level evaluation | Evaluate by negative/spike/normal buckets |
| Decision log / ledger output | Export hourly decision log |

### Key Constraints

1. **No TrendKnightRT**: Deep realtime model is archived (MODEL_NO_GO).
2. **No new deep model training**: Only use existing risk modules.
3. **No production pipeline modification**: This is shadow replay only.
4. **No real deployment**: Simulation and evaluation only.
5. **No test actual in decisions**: Test actual can only be used for evaluation.
6. **No fake metrics**: All metrics must be real and reproducible.

---

## What This Phase Does NOT Cover

| Out of Scope | Reason |
|--------------|--------|
| Production deployment | Shadow replay only |
| Main repo pipeline modification | Isolated in this repo |
| Training new price models | Use existing risk modules |
| Manual correction using test actual | Test actual for evaluation only |
| Final trading decision | Research/validation phase |

---

## Risk Pack Used

| Attribute | Value |
|-----------|-------|
| Source | `reports/local/risk_modules/risk_feature_pack_2026_01_05/risk_feature_pack.csv` |
| Version | v1.1.0 |
| Mode | online |
| Columns | 23 |
| Quality gate | PASS |
| Metric alignment | WARN (allowed) |

---

## Base Prediction Strategy

1. **Primary**: Load base prediction file (SGDFNet / fusion / etc.)
2. **Fallback**: DA anchor baseline (must be marked as sensitivity test, not production baseline)
3. **Requirement**: `business_day + hour_business + target_month` must be unique

---

## Guardrail Policy

### Supported Actions

| Action | Description |
|--------|-------------|
| `none` | No adjustment |
| `alert_only` | Generate alert, no prediction adjustment |
| `soft_negative_blend` | Blend prediction toward negative floor |
| `soft_spike_blend` | Blend prediction toward spike floor |
| `weight_adjust` | Adjust fusion weights based on risk |

### Prohibited Actions

| Action | Reason |
|--------|--------|
| Hard set to actual | Uses test actual in decision |
| Using y_true in decision | Test actual for evaluation only |
| Unbounded arbitrary correction | Must have reasoned, bounded adjustment |

---

## Evaluation Metrics

### Overall Metrics

- sMAPE_floor50 (primary)
- sMAPE (secondary)

### Bucket Metrics

| Bucket | Definition |
|--------|------------|
| Negative | rt_actual < 0 |
| Spike | rt_actual >= 500 |
| Normal | 0 <= rt_actual < 500 |

### Alert Quality Metrics

- Precision / Recall / F1
- Top-k capture
- Lift
- Alert rate

---

## Success Criteria

| Verdict | Condition |
|---------|-----------|
| LEDGER_POLICY_GO | Overall improves >= 0.3pp, target buckets improve, no guardrail violation |
| LEDGER_POLICY_AUX | Bucket improves but overall neutral |
| LEDGER_POLICY_ALERT_ONLY | Correction worsens/neutral but alert metrics strong |
| LEDGER_POLICY_NO_GO | No stable improvement, weak trigger value |

---

## Deliverables

| Track | Deliverable | Status |
|-------|-------------|--------|
| Track 0 | Fix 12 risk module test failures | ✅ In Progress |
| Track A | docs/LEDGER_1_SCOPE.md | ✅ In Progress |
| Track B | models/deep_sgdf_delta/base_prediction_adapter.py | ⏳ Pending |
| Track C | models/deep_sgdf_delta/risk_pack_loader.py | ⏳ Pending |
| Track D | models/deep_sgdf_delta/risk_guardrail_policy.py | ⏳ Pending |
| Track E | models/deep_sgdf_delta/risk_shadow_replay.py | ⏳ Pending |
| Track F | Policy sweep implementation | ⏳ Pending |
| Track G | scripts/evaluate_risk_triggers.py | ⏳ Pending |
| Track H | docs/LEDGER_DECISION_LOG_CONTRACT.md | ⏳ Pending |
| Track I | docs/LEDGER_1_SHADOW_REPLAY_RESULTS.md | ⏳ Pending |
