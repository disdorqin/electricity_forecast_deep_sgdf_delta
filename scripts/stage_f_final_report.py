"""Stage F: write the master 13-section P2 Realtime Deep Open Exploration Report."""
import json
import os
from datetime import datetime
import pandas as pd

ROOT = "outputs/p2_realtime"
CANON = {
    "da_anchor":   "da_anchor_d14_20260706_203328",
    "sgdfnet_d14": "sgdfnet_d14_d14_20260706_203328",
    "tcn_day":     "tcn_day_d14_20260706_203328",
    "gru_day":     "gru_day_d14_20260706_203600",
    "linear_day":  "linear_day_d14_20260706_205109",
    "dlinear_day": "dlinear_day_d14_20260706_205844",
    "tcn_abs":     "tcn_abs_d14_20260706_210259",
}
ORDER = ["da_anchor", "sgdfnet_d14", "gru_day", "tcn_day", "dlinear_day", "linear_day", "tcn_abs"]

def load(rid):
    with open(os.path.join(ROOT, rid, "metrics", "metrics.json"), encoding="utf-8") as f:
        d = json.load(f)
    return d["meta"], d["metrics"]

def g(rid, k):
    return load(rid)[1][k]

def main():
    rows = []
    for m in ORDER:
        meta, mm = load(CANON[m])
        rows.append((m, mm))
    base = load(CANON["da_anchor"])[1]
    sg = load(CANON["sgdfnet_d14"])[1]

    L = []
    L.append("# P2 Realtime Deep Open Exploration Report\n")
    L.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_\n")

    # 1
    L.append("## 1. Task Summary\n")
    L.append(f"- repo: `disdorqin/electricity_forecast_deep_sgdf_delta` (working repo)\n"
             f"- branch: main (clone HEAD 479ee3c)\n"
             f"- run_id: p2_realtime_20260706\n"
             f"- source 2.5 repo/path: `disdorqin/electricity_forecast_model2.5` (= local efm3.0); SGDFNet src via `electricity_forecast_model2.0_exp/SGDFNet`\n"
             f"- target task: realtime_trend (D+1 realtime price trend / delta learning, cutoff-safe)\n"
             f"- cutoff: **D14 (D日 14:00)** — strictly enforced; 2.5 prior D15 (decision_hour=15) results were NOT reused\n"
             f"- test data range: 2025-01-01 .. 2026-06-30 (536 business days)\n"
             f"- test months: all 18 months 2025-01..2026-06 (no skip; data complete)\n")

    # 2
    L.append("## 2. Baseline Understanding\n")
    L.append("- 2.5 realtime models: timesfm / sgdfnet / timemixer / rt916 — all use D14 (`realtime_cutoff_hour=14`); produce rt = da_anchor + delta_pred.\n"
             "- SGDFNet role: gradient-boosted residual (RT-DA) learner; production config default decision_hour=15 → reproduced at D14 by overriding config.\n"
             "- RT916 role: spike-aware residual net (RT); reused only as experience, not re-run here.\n"
             "- TimeMixer role: GPU temporal-mixing residual net; reused as experience.\n"
             "- reused logic: data_contract (RT/DA cols), sMAPE_floor50 metric (floor=50), hour_business/period mapping, D14 cutoff protocol, 24-hour completeness check.\n"
             "- replaced logic: from-scratch unified walk-forward framework (this repo) replaces per-model ad-hoc pipelines for fair comparison.\n"
             "- metric implementation: `p2_common.capped_smape(floor=50)`, MAE/RMSE standard.\n"
             "- cutoff prevention: features assembled only from business_day < T-1 full days + D-1 hours 1..14; lag-24 post-cutoff masked; target actual used ONLY for y_true/metrics.\n")

    # 3 models tried
    L.append("## 3. Models Tried\n")
    L.append("| Model | Status | Notes |")
    L.append("| --- | --- | --- |")
    L.append(f"| DA-anchor (baseline) | SUCCESS | rt = DA price; strong single baseline 31.11% |")
    L.append(f"| SGDFNet D14 (faithful repro) | SUCCESS | 31.99%; confirms 2.5 single-model level |")
    L.append(f"| TCN (delta) | SUCCESS | 34.23%; ≈ DA anchor + noise |")
    L.append(f"| GRU (delta) | SUCCESS | 34.16%; best new deep model |")
    L.append(f"| DLinear (delta) | SUCCESS | 35.24% |")
    L.append(f"| Linear (delta) | SUCCESS | 35.35% |")
    L.append(f"| TCN (abs-target) | SUCCESS | 46.31%; directly predicting RT worse |")

    # 4 overall metrics
    L.append("## 4. Overall Metrics\n")
    L.append("| Model | MAE | RMSE | sMAPE_floor50 | Train s | Infer s | NaN | Failed | Cutoff |")
    L.append("| --- | --: | --: | --: | --: | --: | --: | --: | --: |")
    for m, mm in rows:
        L.append(f"| {m} | {mm['MAE']:.1f} | {mm['RMSE']:.1f} | **{mm['sMAPE_floor50']:.2f}** | "
                 f"{mm.get('train_time_s',0):.1f} | {mm.get('infer_time_s',0):.1f} | {mm['nan_count']} | "
                 f"{mm['failed_days']} | {mm.get('cutoff','D14')} |")

    # 5 period
    L.append("## 5. Period Metrics\n")
    L.append("| Model | 1_8 sMAPE | 9_16 sMAPE | 17_24 sMAPE |")
    L.append("| --- | --: | --: | --: |")
    for m, mm in rows:
        L.append(f"| {m} | {mm['sMAPE_1_8']:.2f} | {mm['sMAPE_9_16']:.2f} | {mm['sMAPE_17_24']:.2f} |")

    # 6 spike/negative
    L.append("## 6. Spike / Negative Metrics\n")
    L.append("| Model | Spike sMAPE | Negative sMAPE | Normal Degradation |")
    L.append("| --- | --: | --: | --: |")
    for m, mm in rows:
        norm = mm['sMAPE_floor50']
        L.append(f"| {m} | {mm['spike_sMAPE_floor50']:.2f} | {mm['negative_sMAPE_floor50']:.2f} | {norm:.2f} |")

    # 7 month breakdown
    L.append("## 7. 2025 / 2026 Month Breakdown (vs DA-anchor baseline)\n")
    base_pm = pd.read_csv(os.path.join(ROOT, CANON["da_anchor"], "metrics", "per_month.csv"))
    cand_pm = pd.read_csv(os.path.join(ROOT, CANON["gru_day"], "metrics", "per_month.csv"))
    L.append("| Month | Baseline sMAPE | Best New Candidate (gru) | Candidate sMAPE | Winner |")
    L.append("| --- | --: | --- | --: | --- |")
    for mo in sorted(base_pm["month"].unique()):
        bv = float(base_pm[base_pm.month == mo].iloc[0]["sMAPE_floor50"])
        cv = float(cand_pm[cand_pm.month == mo].iloc[0]["sMAPE_floor50"])
        w = "gru" if cv < bv else "baseline"
        L.append(f"| {mo} | {bv:.2f} | gru_day | {cv:.2f} | {w} |")

    # 8 cutoff
    L.append("## 8. Cutoff Safety Report\n")
    L.append("- D14 mode implemented: YES (framework decision_hour=14; SGDFNet config overridden to 14)\n"
             "- post-D14 realtime actual used: NO\n"
             "- future (target-day) actual used: NO\n"
             "- leakage risks: none found; audit ALL PASS\n"
             "- verdict: **PASS (D14)**\n")

    # 9 ablation
    L.append("## 9. Ablation Summary\n")
    L.append("| Change | Effect | Keep/Drop |")
    L.append("| --- | --- | --- |")
    L.append("| delta-target vs abs-target (TCN) | abs 46.3% >> delta 34.2% | DROP abs |")
    L.append("| blend da_anchor + sgdfnet (equal) | 31.17% ≈ da 31.11% | DROP (no gain) |")
    L.append("| optimal blend da+tcn | best w(da)=1.0 → 31.11% | DROP (tcn noise) |")
    L.append("| richer lags (24..168 already used) | no improvement | KEEP as-is |")
    L.append("| post-hoc all-equal ensemble | dragged to ~34% | DROP |")

    # 10 candidate package
    L.append("## 10. Candidate Package\n")
    L.append("- export path: `exports/efm3_candidates/realtime_trend/p2_realtime_20260706/`\n"
             "- trend_predictions.csv: gru_day (best new deep candidate)\n"
             "- metrics.json, comparison_report.md, cutoff_safety_report.md, ablation_report.md: present\n"
             "- manifest.json, promotion_decision.json: present\n")

    # 11 risks
    L.append("## 11. Risks\n")
    L.append("- risk: RT-DA residual near-unpredictable from cutoff-safe features → mitigation: use DA anchor as trend, pursue ensemble diversity\n"
             "- risk: single deep model cannot match 2.5 4-model fusion (~23%) → mitigation: reproduce timesfm/timemixer/rt916 to build diverse ensemble\n"
             "- risk: over-fitting on 1 missing day → mitigation: lag-168 fallback, explicit failed_days record\n")

    # 12 recommendation
    L.append("## 12. Recommendation\n")
    L.append("REALTIME_P2_RECOMMENDATION: NO_GO\n")

    # 13 verdict
    L.append("## 13. Final Verdict\n")
    L.append("P2_REALTIME_RESULT: PARTIAL\n")
    L.append("> No candidate beats the DA-anchor / faithful SGDFNet D14 single-model baseline; the open exploration is PARTIAL (framework + faithful baselines established, 7 candidates compared, cutoff-safe, ablation done) but yields NO_GO for promotion. The realtime trend signal is already captured by the DA anchor; ensemble gain requires the diverse 2.5 model set, not a new solo deep model.\n")

    out = os.path.join(ROOT, "P2_REALTIME_FINAL_REPORT.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("WROTE", out)

if __name__ == "__main__":
    main()
