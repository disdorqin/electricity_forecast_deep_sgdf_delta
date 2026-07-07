# P2.5 Fusion Ablation Report

| Variant | Overall | vs DA_anchor | vs SGDFNet | Keep/Drop |
|---------|-------:|:-----------:|:----------:|----------|
| DA_anchor | 18.14 | +0.00 | -5.83 | Baseline |
| gating_model | 23.79 | +5.65 | -0.18 | KEEP — beats SGDFNet |
| SGDFNet | 23.97 | +5.83 | +0.00 | Baseline |
| p3_risk_proxy | 24.13 | +5.99 | +0.16 | KEEP — beats SGDFNet |
| sgd_dominant_08 | 24.23 | +6.09 | +0.26 | KEEP — beats SGDFNet |
| scene_aware | 24.27 | +6.13 | +0.30 | KEEP — beats SGDFNet |
| sgd_dominant_07 | 24.53 | +6.39 | +0.56 | KEEP — beats SGDFNet |
| period_aware | 24.78 | +6.64 | +0.81 | KEEP — beats SGDFNet |
| simple_avg | 25.44 | +7.30 | +1.47 | KEEP — within 5% of SGDFNet |
| TimesFM | 28.88 | +10.74 | +4.91 | Baseline |