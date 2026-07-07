"""P2 Realtime — compare candidate runs against the 2.5 realtime baseline.

Reads outputs/p2_realtime/{run_id}/metrics/metrics.json and per_month.csv for
each candidate, aggregates into a unified comparison table, and writes:
  - realtime_candidate_comparison_report.md
  - realtime_candidate_comparison_metrics.json

The 2.5 realtime fused baseline reference (~sMAPE_floor50 23) is taken from the
2.5 陪跑验收 record (project memory). It is NOT reproduced inside this repo;
we document that clearly and compare each candidate's sMAPE against it.

Usage:
  python scripts/compare_p2_realtime_candidates.py \
      --root outputs/p2_realtime \
      --run-ids run_a run_b run_c \
      --baseline-smape 23.0 \
      --out exports/efm3_candidates/realtime_trend/<run_id>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# 2.5 realtime fused baseline reference (from 2.5 陪跑验收, D14 cutoff, sMAPE_floor50 ~23)
BASELINE_NAME = "2.5_fused_realtime_baseline"


def load_run(root: Path, run_id: str):
    mpath = root / run_id / "metrics" / "metrics.json"
    ppath = root / run_id / "metrics" / "per_month.csv"
    if not mpath.exists():
        raise FileNotFoundError(f"{run_id}: missing metrics.json")
    meta = json.loads(mpath.read_text(encoding="utf-8"))
    overall = meta.get("metrics", {})
    per_month = pd.read_csv(ppath) if ppath.exists() else pd.DataFrame()
    return overall, per_month, meta.get("meta", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/p2_realtime")
    ap.add_argument("--run-ids", nargs="+", required=True)
    ap.add_argument("--baseline-smape", type=float, default=23.0)
    ap.add_argument("--out", default="outputs/p2_realtime/_comparison")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    per_month_all = []
    for rid in args.run_ids:
        overall, per_month, meta = load_run(root, rid)
        model = meta.get("model", rid)
        rows.append(dict(
            model=model, run_id=rid,
            MAE=overall.get("MAE"), RMSE=overall.get("RMSE"),
            sMAPE_floor50=overall.get("sMAPE_floor50"),
            sMAPE_1_8=overall.get("sMAPE_1_8"), sMAPE_9_16=overall.get("sMAPE_9_16"),
            sMAPE_17_24=overall.get("sMAPE_17_24"),
            spike_sMAPE_floor50=overall.get("spike_sMAPE_floor50"),
            negative_sMAPE_floor50=overall.get("negative_sMAPE_floor50"),
            train_time_s=overall.get("train_time_s"), infer_time_s=overall.get("infer_time_s"),
            nan_count=overall.get("nan_count"), failed_days=overall.get("failed_days"),
            missing_hour_count=overall.get("missing_hour_count"),
            n_days=overall.get("n_days"),
            cutoff=meta.get("cutoff", "D14"),
        ))
        if not per_month.empty:
            pm = per_month.copy()
            pm["model"] = model
            pm["run_id"] = rid
            per_month_all.append(pm)

    # baseline row
    rows.append(dict(
        model=BASELINE_NAME, run_id="reference",
        MAE=None, RMSE=None, sMAPE_floor50=args.baseline_smape,
        sMAPE_1_8=None, sMAPE_9_16=None, sMAPE_17_24=None,
        spike_sMAPE_floor50=None, negative_sMAPE_floor50=None,
        train_time_s=None, infer_time_s=None, nan_count=0, failed_days=0,
        missing_hour_count=0, n_days=None, cutoff="D14",
    ))
    cmp = pd.DataFrame(rows)

    # ── markdown report ──
    rep = ["# P2 Realtime — Candidate Comparison vs 2.5 Baseline (D14)", ""]
    rep.append(f"- baseline reference (2.5 fused realtime): sMAPE_floor50 = {args.baseline_smape}")
    rep.append(f"- candidates compared: {len(args.run_ids)}")
    rep.append("")
    rep.append("## 4. Overall Metrics")
    rep.append("| Model | MAE | RMSE | sMAPE_floor50 | Train(s) | Infer(s) | NaN | Failed | Cutoff |")
    rep.append("| --- | --: | ---: | ------------: | -------: | -------: | --: | -----: | ------ |")
    for _, r in cmp.iterrows():
        def f(v):
            return "—" if v is None or (isinstance(v, float) and pd.isna(v)) else (f"{v:.2f}" if isinstance(v, float) else str(v))
        rep.append(f"| {r['model']} | {f(r['MAE'])} | {f(r['RMSE'])} | {f(r['sMAPE_floor50'])} | "
                   f"{f(r['train_time_s'])} | {f(r['infer_time_s'])} | {f(r['nan_count'])} | {f(r['failed_days'])} | {r['cutoff']} |")
    rep.append("")
    rep.append("## 5. Period Metrics (sMAPE_floor50)")
    rep.append("| Model | 1_8 | 9_16 | 17_24 |")
    rep.append("| --- | --: | --: | --: |")
    for _, r in cmp.iterrows():
        def f(v):
            return "—" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:.2f}"
        rep.append(f"| {r['model']} | {f(r['sMAPE_1_8'])} | {f(r['sMAPE_9_16'])} | {f(r['sMAPE_17_24'])} |")
    rep.append("")
    rep.append("## 6. Spike / Negative Metrics (sMAPE_floor50)")
    rep.append("| Model | Spike | Negative |")
    rep.append("| --- | --: | --: |")
    for _, r in cmp.iterrows():
        def f(v):
            return "—" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:.2f}"
        rep.append(f"| {r['model']} | {f(r['spike_sMAPE_floor50'])} | {f(r['negative_sMAPE_floor50'])} |")
    rep.append("")

    if per_month_all:
        pm = pd.concat(per_month_all, ignore_index=True)
        rep.append("## 7. Month Breakdown (sMAPE_floor50)")
        rep.append("| Month | Model | sMAPE_floor50 | MAE | RMSE |")
        rep.append("| --- | --- | --: | --: | --: |")
        for _, r in pm.sort_values(["month", "model"]).iterrows():
            sm = r.get("sMAPE_floor50")
            sm = "—" if pd.isna(sm) else f"{sm:.2f}"
            mae = r.get("MAE"); mae = "—" if pd.isna(mae) else f"{mae:.2f}"
            rmse = r.get("RMSE"); rmse = "—" if pd.isna(rmse) else f"{rmse:.2f}"
            rep.append(f"| {r['month']} | {r['model']} | {sm} | {mae} | {rmse} |")
        rep.append("")

    # winner per month (exclude baseline row)
    rep.append("## 7b. Month Winner (best candidate vs baseline)")
    rep.append("| Month | Baseline | Best Candidate | Candidate sMAPE | Winner |")
    rep.append("| --- | --: | --- | --: | --- |")
    if per_month_all:
        pm = pd.concat(per_month_all, ignore_index=True)
        for month, g in pm.groupby("month"):
            best = g.loc[g["sMAPE_floor50"].astype(float).idxmin()]
            winner = "baseline" if args.baseline_smape <= float(best["sMAPE_floor50"]) else best["model"]
            rep.append(f"| {month} | {args.baseline_smape:.1f} | {best['model']} | {float(best['sMAPE_floor50']):.2f} | {winner} |")
    rep.append("")

    rep.append("## Notes")
    rep.append("- 2.5 fused realtime baseline (~23 sMAPE) is a reference from the 2.5 陪跑验收 record; it is NOT re-produced inside this repo (per task safety boundary: do not modify 2.5).")
    rep.append("- sgdfnet_d14 is a faithful D14 reproduction of the SGDFNet realtime model (via sibling bridge, decision_hour overridden to 14).")
    rep.append("- da_anchor is the strong naive baseline (realtime ≈ DA price).")
    rep.append("- tcn_day / gru_day / dlinear_day / linear_day are uniform-framework deep day-level decoders on D14-safe features.")
    rep.append("")
    (out / "realtime_candidate_comparison_report.md").write_text("\n".join(rep), encoding="utf-8")
    cmp.to_csv(out / "realtime_candidate_comparison_metrics.csv", index=False, encoding="utf-8-sig")
    payload = dict(baseline_smape_floor50=args.baseline_smape,
                   candidates=[{k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in r.items()} for _, r in cmp.iterrows()])
    (out / "realtime_candidate_comparison_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[compare] wrote comparison to {out}")


if __name__ == "__main__":
    main()
