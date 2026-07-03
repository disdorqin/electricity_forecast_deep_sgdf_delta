# Risk Pack Quality Report

**Verdict:** PASS
**Timestamp:** 2026-07-04T00:33:52.748496
**Pack:** `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\deep_model_for_electricity\reports\local\risk_modules\risk_feature_pack_2026_01_05\risk_feature_pack.csv`
**Manifest:** `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\deep_model_for_electricity\reports\local\risk_modules\risk_feature_pack_2026_01_05\manifest.json`

**Checks:** 10/10 passed, 0 failed

## Checks

| # | Check | Status | Critical | Detail |
|---|-------|--------|----------|--------|
| 1 | row_count_matches_manifest_unique_keys | PASS | Yes | pack rows=3624, manifest unique_keys=3624 |
| 2 | no_y_true_in_online_mode | PASS | No | mode=online, y_true present=False |
| 3 | unique_keys_within_target_month | PASS | Yes | duplicate rows=0 |
| 4 | probability_columns_in_unit_interval | PASS | Yes | out-of-range values=0 |
| 5 | risk_feature_version_starts_with_v1 | PASS | Yes | versions found=['v1.1.0'] |
| 6 | metric_alignment_status_valid | PASS | Yes | statuses found=['WARN'] |
| 7 | module_status_not_all_unknown | PASS | Yes | status columns checked=['module_status_delta_supply', 'module_status_spike', 'module_status_negative'], all_unknown=False |
| 8 | target_month_rows_match_monthly_manifest | PASS | No | all months match |
| 9 | no_duplicate_ds_hour_within_target_month | PASS | No | ds duplicates=0, ds+hour duplicates=0 |
| 10 | nan_risk_columns_only_for_nogo_or_insufficient | PASS | No | all NaN risk columns have valid NO-GO/INSUFFICIENT status |
