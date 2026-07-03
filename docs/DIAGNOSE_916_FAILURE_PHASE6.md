# 9_16 Segment Failure Diagnosis

**Date:** 2026-07-03 12:40
**Period:** 2026-02-01 to 2026-02-28
**9_16 rows:** 216

## Analysis 1: By Hour

| Hour | Count | RT Mean | Delta Std | SGDFNet sMAPE |
|------|-------|---------|-----------|---------------|
| 9 | 27 | 162.6 | 79.2 | 25.34 |
| 10 | 27 | 82.8 | 146.9 | 45.60 |
| 11 | 27 | 31.2 | 145.0 | 40.72 |
| 12 | 27 | 31.8 | 136.3 | 43.72 |
| 13 | 27 | 15.3 | 139.6 | 40.70 |
| 14 | 27 | 27.0 | 136.7 | 42.63 |
| 15 | 27 | 30.8 | 135.8 | 39.48 |
| 16 | 27 | 62.5 | 132.7 | 42.23 |

**Hardest hour:** 10 (sMAPE = 45.60)

## Analysis 2: By Bucket

| Bucket | Count | RT Mean | Delta Std | SGDFNet sMAPE |
|--------|-------|---------|-----------|---------------|
| normal | 75 | 296.1 | 172.0 | 61.59 |
| spike | 1 | 511.3 | nan | 41.53 |
| negative | 140 | -76.7 | 67.8 | 28.51 |

## Analysis 3: By Month

| Month | Count | RT Mean | Delta Std | SGDFNet sMAPE |
|-------|-------|---------|-----------|---------------|
| 2026-02 | 216 | 55.5 | 132.2 | 40.05 |

**Hardest month:** 2026-02 (sMAPE = 40.05)

## Analysis 4: Feature Correlation with Delta

| Feature | Correlation |
|---------|-------------|

## Conclusion

**RECOMMENDATION: Consider training a dedicated `TrendKnightSolar916` model.** Worst hour sMAPE = 45.60 > 40, indicating severe solar-volatility mismatch.