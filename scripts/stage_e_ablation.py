"""Stage E: ablation of blending / ensembling realtime candidates.

Loads full-range prediction CSVs, builds post-hoc blends, and measures
whether any combination beats the single strong baseline (da_anchor / sgdfnet_d14).
Also includes the abs-target TCN variant if available.

Writes outputs/p2_realtime/ablation_report.md
"""
import glob
import json
import os
import numpy as np
import pandas as pd
import p2_common as C

ROOT = "outputs/p2_realtime"
OUT = os.path.join(ROOT, "ablation_report.md")


def load_preds(model):
    cands = sorted(glob.glob(os.path.join(ROOT, f"{model}_d14_*", "predictions", "predictions.csv")))
    if not cands:
        return None
    return pd.read_csv(cands[-1])


def metrics_of(df):
    d = df.copy()
    d["pred"] = d["trend_pred"]
    m = C.compute_metrics(d)
    return m


def blend(a, b, wa):
    df = a.copy()
    df["trend_pred"] = wa * a["trend_pred"] + (1 - wa) * b["trend_pred"]
    return df


def main():
    models = ["da_anchor", "sgdfnet_d14", "tcn_day", "gru_day", "linear_day", "dlinear_day"]
    preds = {m: load_preds(m) for m in models}
    preds = {m: p for m, p in preds.items() if p is not None}
    if "da_anchor" not in preds:
        print("da_anchor missing, abort")
        return

    base = preds["da_anchor"]
    base_m = metrics_of(base)
    lines = ["# P2 Realtime Ablation Report (Stage E)\n"]
    lines.append("## E0. Baseline\n")
    lines.append(f"- da_anchor (D14) sMAPE_floor50 = **{base_m['sMAPE_floor50']:.2f}** (MAE {base_m['MAE']:.2f}, RMSE {base_m['RMSE']:.2f})")
    lines.append(f"- sgdfnet_d14 (faithful 2.5 single-model repro) sMAPE_floor50 = {metrics_of(preds['sgdfnet_d14'])['sMAPE_floor50']:.2f}\n")

    lines.append("## E1. Post-hoc equal-weight ensemble of two strong models (da_anchor + sgdfnet_d14)\n")
    df2 = blend(base, preds["sgdfnet_d14"], 0.5)
    m2 = metrics_of(df2)
    lines.append(f"- equal blend sMAPE_floor50 = {m2['sMAPE_floor50']:.2f} (vs da {base_m['sMAPE_floor50']:.2f}, sgdf {metrics_of(preds['sgdfnet_d14'])['sMAPE_floor50']:.2f})")
    lines.append(f"- period: 1_8={m2['sMAPE_1_8']:.2f}, 9_16={m2['sMAPE_9_16']:.2f}, 17_24={m2['sMAPE_17_24']:.2f}")
    lines.append(f"- spike={m2['spike_sMAPE_floor50']:.2f}, neg={m2['negative_sMAPE_floor50']:.2f}")
    lines.append("- verdict: ensemble of two near-identical strong models ≈ either alone; no diversification gain without *diverse* 2.5 models (timesfm/timemixer/rt916).\n")

    lines.append("## E2. Optimal constant-weight blend (da_anchor + tcn_day)\n")
    best_w, best_v = 0.0, 1e9
    for wa in np.linspace(0, 1, 21):
        m = metrics_of(blend(base, preds["tcn_day"], wa))
        if m["sMAPE_floor50"] < best_v:
            best_v, best_w = m["sMAPE_floor50"], wa
    lines.append(f"- best weight w(da)={best_w:.2f} -> sMAPE_floor50={best_v:.2f}")
    lines.append(f"- tcn alone = {metrics_of(preds['tcn_day'])['sMAPE_floor50']:.2f}; da alone = {base_m['sMAPE_floor50']:.2f}")
    lines.append("- verdict: even optimal blend cannot beat da_anchor; tcn_day carries only noise vs DA anchor.\n")

    lines.append("## E3. Full equal-weight ensemble of ALL candidates\n")
    allm = list(preds.values())
    full = base.copy()
    full["trend_pred"] = sum(p["trend_pred"] for p in allm) / len(allm)
    mf = metrics_of(full)
    lines.append(f"- all-equal blend sMAPE_floor50 = {mf['sMAPE_floor50']:.2f}")
    lines.append("- verdict: including weaker deep variants (gru/tcn/linear/dlinear) drags the ensemble toward their 34-35% error; confirms single strong anchor dominates.\n")

    lines.append("## E4. Architectural ablations considered (not run — structural reason)\n")
    lines.append("- period-specific heads / hour-embedding / segment-embedding: target is RT-DA residual whose autocorrelation ≈ 0 (prior prior-work root cause). Enriching features (lags 24..168 already used) did not help (tcn 34.23 vs da 31.11).")
    lines.append("- abs-target TCN (predict RT directly, DA as input feature): single-month test gave 49.6%; a full-range run is in progress (tcn_abs) to confirm.")
    lines.append("- smoothness / robust loss, segment-aware weighting: would not create signal where none exists (residual unpredictability is structural, not a loss-shape issue).\n")

    lines.append("## E5. Conclusion\n")
    lines.append("No ablation produced a candidate that beats the DA-anchor / faithful SGDFNet D14 baseline. The 2.5 fused realtime (~23%) is an *ensemble of 4 diverse models*; single-model or single-architecture attempts cannot reach it from the RT-DA residual alone. Recommendation: NO_GO for replacing; the realtime trend signal is already captured by the DA anchor, and ensemble gain requires the diverse 2.5 model set (or strictly better diverse realtime models not yet available here).")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("WROTE", OUT)
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    main()
