# SGDFNet Baseline Consistency Audit

**Date:** 2026-07-03 13:01
**Period:** 2026-02-01 to 2026-02-28

## Results

| Source | Rows Matched | sMAPE_floor50 | Detail |
|--------|-------------|---------------|--------|
| teacher_adapter | 672 | 32.0227 | sgdfnet_teacher.load_predictions() |
| fusion_trial | 672 | 32.0227 | run_simple_fusion_trial.py --scheme sgdfnet_only |

> **Note:** p0 source not available. To generate:
> ```
> python scripts/p0_reproduce_sgdfnet_baseline.py \
>     --start-date 2026-02-01 --end-date 2026-02-28
> ```

## Consistency Check

- Sources available: 2
- sMAPE range across sources: 0.0000
- Rows consistent: True

## Verdict: **PARTIAL_PASS_2_SOURCE**

2 sources agree within tolerance. Baseline is partially consistent.
Full 3-source audit requires p0 output (run p0_reproduce_sgdfnet_baseline.py).