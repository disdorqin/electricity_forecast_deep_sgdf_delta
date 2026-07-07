# P2.9 Rule Gate Report

## Rule Gates Evaluated

| Gate | Rule | Overall |
|------|------|-------:|
| hour_period_rule | HourPeriodRule | 19.88 |
| month_regime_rule | MonthRegimeRule | 19.27 |
| volatility_rule | VolatilityRule | 19.8 |
| volatility_rule | VolatilityRule | 19.8 |
| conservative_gate | ConservativeGate | 19.23 |

**Best rule gate**: conservative_gate (19.23)

## Conservative Gate Rules
- Default: DA anchor (confidence 0.9)
- Switch to SGDFNet only when DA-SGD gap > 50 AND price is normal (0 < DA < 200)
- Negative price hours: always DA (SGDFNet not proven better on negatives)
- Spike hours (DA > 200): DA default, SGDFNet only on large gaps
