# Metric Alignment Audit Report

**Date:** 2026-07-03 23:02
**Canonical formula:** `models.deep_sgdf_delta.metrics.smape_floor50` (multiplier=200, percent scale)

## Per-Source Statistics

| Source | Rows | Date Min | Date Max | DA mean | DA std | RT mean | RT std |
|--------|------|----------|----------|---------|--------|---------|--------|
| DeepFinal | 672 | 2026-02-01 01:00:00 | 2026-03-01 00:00:00 | 212.6279 | 193.4911 | None | None |
| DeltaSupply | 672 | 2026-02-01 00:00:00 | 2026-02-28 23:00:00 | 212.6401 | 193.5007 | 206.7426 | 215.084 |
| RawData | 39168 | 2022-01-01 01:00:00 | 2026-06-21 00:00:00 | 326.0921 | 199.0494 | 324.498 | 222.1887 |

## sMAPE_floor50 Comparison

| Source | Full sMAPE | Common Rows | Common sMAPE |
|--------|-----------|-------------|-------------|
| DeepFinal | N/A | 671 | N/A |
| DeltaSupply | N/A | 671 | 26.7203 |
| RawData | N/A | 671 | 26.7203 |

- Common sMAPE min: 26.7203
- Common sMAPE max: 26.7203
- Spread: 0.0000 pp

## Verdict: **PASS**

Spread threshold: PASS <= 0.1 pp, WARN <= 1.0 pp, FAIL > 1.0 pp

All modules are metric-aligned within 0.1 pp on the common intersection.