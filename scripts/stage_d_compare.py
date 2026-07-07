"""Stage D: aggregate all per-model run results into a comparison report.

Reads outputs/p2_realtime/<run_id>/metrics/{metrics.json,per_month.csv}
Groups by model, keeps latest run per model, builds:
  - overall metrics table
  - period table
  - spike/negative table
  - 2025 / 2026 month breakdown (baseline = da_anchor)
Writes:
  outputs/p2_realtime/comparison_report.md
  outputs/p2_realtime/comparison_metrics.json
"""
import json
import glob
import os
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = "outputs/p2_realtime"
BASELINE = "da_anchor"  # strong single realtime baseline (DA anchor at D14)

# Canonical full-range (2025-01..2026-06, D14) runs. The abs-target TCN run is
# tracked separately under "tcn_abs" (same model name in metrics, distinct dir).
CANONICAL = {
    "da_anchor":   "da_anchor_d14_20260706_203328",
    "sgdfnet_d14": "sgdfnet_d14_d14_20260706_203328",
    "tcn_day":     "tcn_day_d14_20260706_203328",
    "gru_day":     "gru_day_d14_20260706_203600",
    "linear_day":  "linear_day_d14_20260706_205109",
    "dlinear_day": "dlinear_day_d14_20260706_205844",
}

def discover():
    runs = {}
    for model, run_id in CANONICAL.items():
        mj = os.path.join(ROOT, run_id, "metrics", "metrics.json")
        if not os.path.exists(mj):
            continue
        with open(mj, encoding="utf-8") as f:
            d = json.load(f)
        pm = mj.replace("metrics.json", "per_month.csv")
        runs[model] = (run_id, d, pm if os.path.exists(pm) else None)
    return runs

def main():
    runs = discover()
    rows = []
    per_month_all = {}
    for model, (run_id, d, pm) in runs.items():
        m = d["metrics"]
        rows.append(dict(
            model=model, run_id=run_id,
            MAE=round(m["MAE"], 2), RMSE=round(m["RMSE"], 2),
            sMAPE_floor50=round(m["sMAPE_floor50"], 2),
            sMAPE_1_8=round(m["sMAPE_1_8"], 2),
            sMAPE_9_16=round(m["sMAPE_9_16"], 2),
            sMAPE_17_24=round(m["sMAPE_17_24"], 2),
            spike_sMAPE=round(m["spike_sMAPE_floor50"], 2),
            neg_sMAPE=round(m["negative_sMAPE_floor50"], 2),
            train_s=round(m.get("train_time_s", 0), 1),
            infer_s=round(m.get("infer_time_s", 0), 1),
            nan=m["nan_count"], failed=m["failed_days"],
            cutoff=m.get("cutoff", d["meta"].get("cutoff", "D14")),
        ))
        if pm:
            pmdf = pd.read_csv(pm)
            pmdf["model"] = model
            per_month_all[model] = pmdf
    rows.sort(key=lambda r: r["sMAPE_floor50"])
    overall = pd.DataFrame(rows)

    # ---- markdown tables ----
    lines = []
    lines.append("# P2 Realtime Candidate Comparison Report (Stage D)\n")
    lines.append(f"- cutoff: D14 (D日14:00); range 2025-01 .. 2026-06 ({int(overall.iloc[0]['nan']) and 536} days)\n")
    lines.append("## Overall Metrics (lower is better)\n")
    lines.append("| Model | MAE | RMSE | sMAPE_floor50 | 1_8 | 9_16 | 17_24 | spike sMAPE | neg sMAPE | train_s | infer_s | NaN | failed | cutoff |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|")
    for _, r in overall.iterrows():
        lines.append(
            f"| {r.model} | {r.MAE} | {r.RMSE} | **{r.sMAPE_floor50}** | {r.sMAPE_1_8} | {r.sMAPE_9_16} | {r.sMAPE_17_24} | {r.spike_sMAPE} | {r.neg_sMAPE} | {r.train_s} | {r.infer_s} | {r.nan} | {r.failed} | {r.cutoff} |")
    base_smape = float(overall[overall.model == BASELINE].iloc[0].sMAPE_floor50)
    lines.append(f"\n> Reference strong single baseline **{BASELINE}** sMAPE_floor50 = {base_smape}%")
    lines.append("> External 2.5 fused realtime reference ≈ 23% (multi-model ensemble; not reproduced in this repo).\n")

    lines.append("## Period Metrics\n")
    lines.append("| Model | 1_8 sMAPE | 9_16 sMAPE | 17_24 sMAPE |")
    lines.append("|---|---:|---:|---:|")
    for _, r in overall.iterrows():
        lines.append(f"| {r.model} | {r.sMAPE_1_8} | {r.sMAPE_9_16} | {r.sMAPE_17_24} |")

    lines.append("\n## Spike / Negative / Normal\n")
    lines.append("| Model | Spike sMAPE | Negative sMAPE | Normal count | Spike count | Neg count |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in overall.iterrows():
        lines.append(f"| {r.model} | {r.spike_sMAPE} | {r.neg_sMAPE} | - | - | - |")

    # ---- month breakdown ----
    if BASELINE in per_month_all:
        base_pm = per_month_all[BASELINE]
        lines.append("\n## 2025 / 2026 Month Breakdown (vs DA-anchor baseline)\n")
        lines.append("| Month | Baseline sMAPE | Best Candidate | Best sMAPE | Winner |")
        lines.append("|---|---:|---:|---:|:--:|")
        months = sorted(base_pm["month"].unique())
        for mo in months:
            bv = float(base_pm[base_pm.month == mo].iloc[0]["sMAPE_floor50"])
            best_model, best_v, winner = None, 1e9, "baseline"
            for model, pmdf in per_month_all.items():
                if model == BASELINE:
                    continue
                sub = pmdf[pmdf.month == mo]
                if sub.empty:
                    continue
                v = float(sub.iloc[0]["sMAPE_floor50"])
                if v < best_v:
                    best_v, best_model = v, model
            if best_model and best_v < bv:
                winner = best_model
            lines.append(f"| {mo} | {bv:.2f} | {best_model or '-'} | {best_v if best_model else float('nan'):.2f} | {winner} |")
    else:
        lines.append("\n> baseline per-month data missing\n")

    os.makedirs(ROOT, exist_ok=True)
    with open(os.path.join(ROOT, "comparison_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    overall.to_json(os.path.join(ROOT, "comparison_metrics.json"), orient="records", indent=2, force_ascii=False)
    print("WROTE comparison_report.md + comparison_metrics.json")
    print(overall.to_string(index=False))

if __name__ == "__main__":
    main()
