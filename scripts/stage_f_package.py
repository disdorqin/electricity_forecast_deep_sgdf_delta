"""Stage F: assemble the 3.0 realtime_trend candidate package + final report.

Verdict for this exploration = NO_GO (no deep candidate beats the DA-anchor /
faithful SGDFNet D14 baseline, let alone the 2.5 fused realtime ~23%).

Exports:
  exports/efm3_candidates/realtime_trend/<run_id>/
    trend_predictions.csv, metrics.json, comparison_report.md,
    cutoff_safety_report.md, ablation_report.md, manifest.json,
    promotion_decision.json
And the master report:
  outputs/p2_realtime/P2_REALTIME_FINAL_REPORT.md
"""
import json
import os
import shutil
import datetime as dt
import pandas as pd
import subprocess

ROOT = "outputs/p2_realtime"
EXPORT_ROOT = "exports/efm3_candidates/realtime_trend"
RUN_ID = "p2_realtime_20260706"
SRC_REPO = "electricity_forecast_deep_sgdf_delta"
SRC_COMMIT = "479ee3c"  # deep_sgdf_delta HEAD at clone

CANON = {
    "da_anchor":   "da_anchor_d14_20260706_203328",
    "sgdfnet_d14": "sgdfnet_d14_d14_20260706_203328",
    "tcn_day":     "tcn_day_d14_20260706_203328",
    "gru_day":     "gru_day_d14_20260706_203600",
    "linear_day":  "linear_day_d14_20260706_205109",
    "dlinear_day": "dlinear_day_d14_20260706_205844",
    "tcn_abs":     "tcn_abs_d14_20260706_210259",
}

def load_metrics(rid):
    with open(os.path.join(ROOT, rid, "metrics", "metrics.json"), encoding="utf-8") as f:
        d = json.load(f)
    return d["meta"], d["metrics"]

def main():
    base_meta, base_m = load_metrics(CANON["da_anchor"])
    cand_meta, cand_m = load_metrics(CANON["gru_day"])  # best new deep model

    # ---- copy artifacts into export dir ----
    out = os.path.join(EXPORT_ROOT, RUN_ID)
    os.makedirs(os.path.join(out, "reports"), exist_ok=True)
    # trend predictions (best new deep candidate)
    shutil.copy(os.path.join(ROOT, CANON["gru_day"], "predictions", "predictions.csv"),
                os.path.join(out, "trend_predictions.csv"))
    # metrics.json (per-model summary)
    summary = {}
    for model, rid in CANON.items():
        meta, m = load_metrics(rid)
        summary[model] = {"run_id": rid, "metrics": m, "meta": meta}
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    # comparison / cutoff / ablation reports
    shutil.copy(os.path.join(ROOT, "comparison_report.md"), os.path.join(out, "comparison_report.md"))
    shutil.copy(os.path.join(ROOT, "ablation_report.md"), os.path.join(out, "ablation_report.md"))
    shutil.copy(os.path.join(ROOT, "_comparison", "realtime_cutoff_safety_report.md"),
                os.path.join(out, "cutoff_safety_report.md"))
    shutil.copytree(os.path.join(ROOT, CANON["gru_day"]), os.path.join(out, "reports", CANON["gru_day"]),
                    dirs_exist_ok=True)

    # ---- manifest.json ----
    manifest = {
        "run_id": RUN_ID,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_repo": SRC_REPO,
        "source_commit": SRC_COMMIT,
        "target_task": "realtime_trend",
        "model_name": "gru_day (best new deep candidate; DA-anchor is the standing baseline)",
        "model_version": "v1",
        "data_range": f"{base_meta['start']}..{base_meta['end']}",
        "test_months": base_meta["months"],
        "baseline_reference": "da_anchor (D14) = 31.11% ; 2.5 fused realtime external ref ~23%",
        "metric_names": ["MAE", "RMSE", "sMAPE_floor50", "sMAPE_1_8", "sMAPE_9_16",
                         "sMAPE_17_24", "spike_sMAPE_floor50", "negative_sMAPE_floor50",
                         "nan_count", "failed_days", "missing_hour_count"],
        "output_schema_version": "p2_realtime_v1",
        "cutoff": "D14",
        "leakage_check": "PASS (audit: visible frame masked at D-1 14:00; no target-day actual as feature)",
        "nan_check": "PASS (24 missing-day rows filled with lag-168 realtime fallback; trend/delta no NaN)",
        "hour_completeness_check": "PASS (24 rows/day, hour_business 1..24)",
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ---- promotion_decision.json ----
    decision = {
        "recommended_status": "no_go",
        "reason": ("No deep realtime candidate beats the DA-anchor (31.11%) / faithful SGDFNet D14 "
                   "(31.99%) single-model baseline; all new deep models (tcn/gru/linear/dlinear) land "
                   "34-35%, and abs-target TCN 46%. The 2.5 fused realtime (~23%) is an ensemble of 4 "
                   "diverse models; single-architecture residual learning cannot reach it because the "
                   "RT-DA residual autocorrelation is ~0 (structural). Recommend keeping DA-anchor as the "
                   "trend source and pursuing ensemble diversity, not a new solo deep model."),
        "baseline_smape_floor50": round(base_m["sMAPE_floor50"], 2),
        "candidate_smape_floor50": round(cand_m["sMAPE_floor50"], 2),
        "period_results": {
            "1_8": {"baseline": round(base_m["sMAPE_1_8"], 2), "candidate": round(cand_m["sMAPE_1_8"], 2)},
            "9_16": {"baseline": round(base_m["sMAPE_9_16"], 2), "candidate": round(cand_m["sMAPE_9_16"], 2)},
            "17_24": {"baseline": round(base_m["sMAPE_17_24"], 2), "candidate": round(cand_m["sMAPE_17_24"], 2)},
        },
        "spike_results": {"baseline": round(base_m["spike_sMAPE_floor50"], 2),
                          "candidate": round(cand_m["spike_sMAPE_floor50"], 2)},
        "negative_results": {"baseline": round(base_m["negative_sMAPE_floor50"], 2),
                             "candidate": round(cand_m["negative_sMAPE_floor50"], 2)},
        "cutoff_safety_result": "PASS (D14, ALL runs audited PASS)",
        "known_risks": [
            "RT-DA residual near-unpredictable from cutoff-safe features (root cause).",
            "Deep models add variance/noise vs the strong DA anchor.",
            "2.5 fused realtime (~23%) needs 4 diverse models not reproduced here individually.",
        ],
        "required_followup": [
            "Reproduce timesfm / timemixer / rt916 single-model D14 outputs to build a true diverse ensemble.",
            "Re-evaluate only if a model strictly beats da_anchor on >=3 months in BOTH 2025 and 2026.",
            "Consider period-specific / segment heads only as ensemble members, not solo replacements.",
        ],
    }
    with open(os.path.join(out, "promotion_decision.json"), "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)

    print("EXPORTED package ->", out)
    print("recommended_status =", decision["recommended_status"])

if __name__ == "__main__":
    main()
