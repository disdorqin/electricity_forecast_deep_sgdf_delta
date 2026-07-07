# P2 Realtime — Cutoff Safety Audit (D14)

- decision_hour enforced: 14 (D14: only D-1 14:00 and earlier realtime actuals visible)
- runs audited: 7

| Run | Verdict | Rows | Days | trend NaN | delta NaN | Audit Rows | Issues |
| --- | --- | --: | --: | --: | --: | --: | --- |
| da_anchor_d14_20260706_203328 | PASS | 12864 | 536 | 0 | 0 | 536 | none |
| tcn_day_d14_20260706_203328 | PASS | 12864 | 536 | 0 | 0 | 536 | none |
| gru_day_d14_20260706_203600 | PASS | 12864 | 536 | 0 | 0 | 536 | none |
| sgdfnet_d14_d14_20260706_203328 | PASS | 12864 | 536 | 0 | 0 | 12864 | none |
| linear_day_d14_20260706_205109 | PASS | 12864 | 536 | 0 | 0 | 536 | none |
| dlinear_day_d14_20260706_205844 | PASS | 12864 | 536 | 0 | 0 | 536 | none |
| tcn_abs_d14_20260706_210259 | PASS | 12864 | 536 | 0 | 0 | 536 | none |

## 8. Cutoff Safety Verdict
- D14 mode implemented: YES (framework asserts decision_hour==14; SGDFNet run overrides decision_hour=14)
- post-D14 realtime actual used: NO (visible frame masked at hour 14)
- future (target-day) actual used: NO (target actual only used for y_true / metrics, never as feature)
- leakage risks: structural — features assembled strictly from business_day < T-1 full days and D-1 hours 1..14; lag-24 post-cutoff masked.
- verdict: ALL PASS
