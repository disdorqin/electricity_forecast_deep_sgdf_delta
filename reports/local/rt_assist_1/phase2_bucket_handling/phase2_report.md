# Phase 2: Data Classification + Bucket-specific Handling

## Summary

Bucket distribution:
- Bucket 0 (Normal): 23382 rows (59.7%)
- Bucket 1 (Negative DA): 3818 rows (9.7%)
- Bucket 2 (Large residual): 6211 rows (15.9%)
- Bucket 3 (Spike): 5757 rows (14.7%)

## Approach 1: Bucket as Feature

Best improvement: 0.9373pp

## Approach 2: Bucket-specific Alpha/Clip

Improvement: 0.0000pp

## Bucket-specific Results

 bucket    bucket_name  best_alpha  best_clip  best_improvement  n_rows
      0         Normal         0.1          0          0.104707    1571
      1    Negative DA         0.1          0          0.568030     173
      2 Large residual         0.1          0          3.298230     411
      3          Spike         0.1          0          2.506158      53

## Verdict

**✅ KEEP bucket as feature**

Bucket-specific handling shows promise. Continue to Phase 3.
