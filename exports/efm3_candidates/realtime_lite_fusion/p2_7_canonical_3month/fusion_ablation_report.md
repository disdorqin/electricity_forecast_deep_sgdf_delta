# P2.7 Canonical Fusion Ablation

| Variant | Overall | vs DA_anchor | vs SGDFNet | Keep/Drop |
| DA_anchor | 18.14 | +0.00 | -0.89 | Baseline |
| gating_model | 18.76 | +0.62 | -0.27 | KEEP — beats SGDFNet |
| SGDFNet | 19.03 | +0.89 | +0.00 | Baseline |
| sgd_dominant_08 | 19.29 | +1.15 | +0.26 | KEEP — within 2% of SGDFNet |
| p3_risk_proxy | 19.3 | +1.16 | +0.27 | KEEP — within 2% of SGDFNet |
| simple_avg | 20.68 | +2.54 | +1.65 | DROP — worse than SGDFNet |
| TimesFM | 25.09 | +6.95 | +6.06 | Baseline |