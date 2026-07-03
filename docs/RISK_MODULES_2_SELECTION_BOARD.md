# RiskModules-2 Selection Board

> Multi-month backtest evaluation and selection decisions for risk modules.

## Overview

This document records the selection board results for all risk modules after multi-month walk-forward backtesting. Each module is evaluated across multiple months and receives a verdict that determines its role in the next phase of the pipeline.

**Date**: YYYY-MM-DD
**Backtest window**: YYYY-MM through YYYY-MM
**Metric alignment status**: PASS / FAIL

---

## Module Selection Summary

| Module | Multi-Month Verdict | Decision | Next Phase Role |
|--------|-------------------|----------|----------------|
| NegativeRisk | _see below_ | KEEP / KEEP_AS_AUX / DROP / NEEDS_MORE_DATA | champion / aux / drop |
| SpikeRisk | _see below_ | KEEP / KEEP_AS_AUX / DROP / NEEDS_MORE_DATA | champion / aux / needs more data |
| DeltaSupplyRisk | _see below_ | KEEP / KEEP_AS_AUX / DROP / NEEDS_MORE_DATA | aux / drop |

---

## Decision Rules

| Decision | Criteria |
|----------|----------|
| **KEEP** | Stable GO or ACCEPTABLE verdict across >= 2 months |
| **KEEP_AS_AUX** | LOW_VALUE overall but useful top-k discrimination (AUC >= 0.85) |
| **DROP** | NO-GO across all months, or overall verdict is NO-GO |
| **NEEDS_MORE_DATA** | Fewer than 2 successful months, or ambiguous verdict distribution |

---

## Module Details

### 1. NegativeRisk

**Module name**: NegativeRisk
**Backtest root**: `reports/local/risk_modules/negative_risk_backtest_YYYY_MM_DD`

#### Key Metrics Summary

| Metric | Value |
|--------|-------|
| Overall verdict | _fill in_ |
| Mean ROC-AUC | _fill in_ |
| Mean F1 | _fill in_ |
| Months evaluated | _fill in_ |
| GO months | _fill in_ |
| NO-GO months | _fill in_ |

#### Monthly Verdicts

| Month | Verdict | AUC | F1 | Notes |
|-------|---------|-----|----|-------|
| YYYY-MM | _fill in_ | _fill in_ | _fill in_ | |
| YYYY-MM | _fill in_ | _fill in_ | _fill in_ | |

#### Decision: _KEEP / KEEP_AS_AUX / DROP / NEEDS_MORE_DATA_

**Reason**: _fill in_

#### Recommendation for Next Phase

- **Role**: champion / aux / drop
- **Rationale**: _fill in_
- **Action items**:
  - [ ] _fill in_

---

### 2. SpikeRisk

**Module name**: SpikeRisk
**Backtest root**: `reports/local/risk_modules/spike_risk_backtest_YYYY_MM_DD`

#### Key Metrics Summary

| Metric | Value |
|--------|-------|
| Overall verdict | _fill in_ |
| Mean ROC-AUC | _fill in_ |
| Mean F1 | _fill in_ |
| Months evaluated | _fill in_ |
| GO months | _fill in_ |
| NO-GO months | _fill in_ |

#### Monthly Verdicts

| Month | Verdict | AUC | F1 | Notes |
|-------|---------|-----|----|-------|
| YYYY-MM | _fill in_ | _fill in_ | _fill in_ | |
| YYYY-MM | _fill in_ | _fill in_ | _fill in_ | |

#### Decision: _KEEP / KEEP_AS_AUX / DROP / NEEDS_MORE_DATA_

**Reason**: _fill in_

#### Recommendation for Next Phase

- **Role**: champion / aux / needs more data
- **Rationale**: _fill in_
- **Action items**:
  - [ ] _fill in_

---

### 3. DeltaSupplyRisk

**Module name**: DeltaSupplyRisk
**Backtest root**: `reports/local/risk_modules/delta_supply_risk_backtest_YYYY_MM_DD`

#### Key Metrics Summary

| Metric | Value |
|--------|-------|
| Overall verdict | _fill in_ |
| Mean ROC-AUC | _fill in_ |
| Mean F1 | _fill in_ |
| Months evaluated | _fill in_ |
| GO months | _fill in_ |
| NO-GO months | _fill in_ |

#### Monthly Verdicts

| Month | Verdict | AUC | F1 | Notes |
|-------|---------|-----|----|-------|
| YYYY-MM | _fill in_ | _fill in_ | _fill in_ | |
| YYYY-MM | _fill in_ | _fill in_ | _fill in_ | |

#### Decision: _KEEP / KEEP_AS_AUX / DROP / NEEDS_MORE_DATA_

**Reason**: _fill in_

#### Recommendation for Next Phase

- **Role**: aux / drop
- **Rationale**: _fill in_
- **Action items**:
  - [ ] _fill in_

---

## Next Phase Recommendations

| Module | Recommended Role | Confidence | Notes |
|--------|-----------------|------------|-------|
| NegativeRisk | champion / aux / drop | High / Medium / Low | _fill in_ |
| SpikeRisk | champion / aux / needs more data | High / Medium / Low | _fill in_ |
| DeltaSupplyRisk | aux / drop | High / Medium / Low | _fill in_ |

### Integration Plan

1. **Champion modules**: Integrate as primary risk signals in the fusion layer.
2. **Auxiliary modules**: Use as top-k auxiliary features for downstream decision support.
3. **Dropped modules**: Remove from the production pipeline. Archive backtest artifacts for reference.
4. **Needs more data**: Schedule additional backtest months before making a final decision.

---

## Risk Feature Pack Export

After selection, the multi-month risk feature pack is exported via:

```bash
python scripts/export_risk_feature_pack_multimonth.py \
  --delta-supply-root reports/local/risk_modules/delta_supply_risk_backtest_YYYY_MM_DD \
  --spike-root reports/local/risk_modules/spike_risk_backtest_YYYY_MM_DD \
  --negative-root reports/local/risk_modules/negative_risk_backtest_YYYY_MM_DD \
  --metric-alignment-status PASS \
  --out-dir reports/local/risk_modules/risk_feature_pack_YYYY_MM_DD \
  --mode online
```

**Schema version**: v1.1.0
**Threshold version**: v1.0.0

---

## Reproduction

```bash
# Run selection board
python scripts/select_risk_modules.py \
  --delta-supply-backtest reports/local/risk_modules/delta_supply_risk_backtest_YYYY_MM_DD \
  --negative-backtest reports/local/risk_modules/negative_risk_backtest_YYYY_MM_DD \
  --spike-backtest reports/local/risk_modules/spike_risk_backtest_YYYY_MM_DD \
  --out-dir reports/local/risk_modules/risk_module_selection
```

Output files:
- `risk_module_selection.json` -- machine-readable decisions
- `risk_module_selection.csv` -- human-readable summary table

---

## No Fabricated Metrics

All metrics in this document are from actual backtest runs. Fill in the template fields above with real data before finalizing. Do not fabricate or estimate metrics.
