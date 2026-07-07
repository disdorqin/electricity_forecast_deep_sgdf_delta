# P2.9 Leakage Audit

| Check | Result | Notes |
|-------|--------|-------|
| No target-day actual features | ✅ PASS | Only DA, SGD, TFM, hour, dow, gap, flags used |
| No D14-after realtime actual | ✅ PASS | All features available at D14 cutoff |
| Canonical hour mapping used | ✅ PASS | common/realtime_canonical_loader.py |
| Rolling features use past only | ✅ PASS | All features are per-hour, no rolling lookahead |
| Selector fallback safe | ✅ PASS | ConservativeGate defaults to DA |
| No future data in training | ⚠️ LOMO | LOMO uses other months; time split is clean |
