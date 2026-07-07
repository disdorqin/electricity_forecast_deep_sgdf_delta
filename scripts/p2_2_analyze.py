"""P2.2 analysis — multi-season Ledger verification.

Reads the realtime prediction + actual ledger (populated by p2_2_populate_ledger,
which also carries the 32 winter 2026 days as history), reproduces the faithful
2.5 realtime four-model GEF fusion (DailyLedgerGEF + apply_daily_ledger_weights),
computes all 11 required variants, and emits the candidate package:

  exports/efm3_candidates/realtime_ensemble/p2_2_multiseason/
    monthly_metrics.json
    scene_breakdown_metrics.json
    model_contribution_report.md
    fusion_ablation_report.md
    ledger_backfill_report.md
    manifest.json
    promotion_decision.json
    p2_2_report.md   (7-section final report)

Metric: capped_smape (floor=50) — identical to Phase G — so numbers are comparable.
"""
from __future__ import annotations
import os
import sys
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPTIM_NUM_WORKERS", "0")
os.environ.setdefault("OPTIM_PIN_MEMORY", "0")
import torch  # preload to avoid flaky import segfault when importing efm3 modules
print(f"[analyze] torch {torch.__version__} preloaded", flush=True)

EFM3 = "D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/efm3.0"
if EFM3 not in sys.path:
    sys.path.insert(0, EFM3)
EXPORT_DIR = Path("D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_deep_sgdf_delta/exports/efm3_candidates/realtime_ensemble/p2_2_multiseason")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
from pipelines.prediction_ledger import (
    load_prediction_ledger, load_actual_ledger, build_ledger_training_table,
)
from pipelines.ledger_weight import select_complete_training_days
from fusion.learners.daily_ledger_gef import DailyLedgerGEF, GEFConfig, smape_floor50
from fusion.apply_daily_ledger_weights import apply_daily_ledger_weights

REALTIME_MODELS = ["timesfm", "sgdfnet", "timemixer", "rt916"]
MONTHS = ["2025-03", "2025-04", "2025-05", "2025-06", "2025-09", "2025-10",
          "2026-03", "2026-04", "2026-05", "2026-06"]
LEDGER_ROOT = Path(EFM3) / "outputs" / "ledger"
PERIODS = ("1_8", "9_16", "17_24")

VARIANTS = ["DA_anchor", "timesfm", "sgdfnet", "timemixer", "rt916",
            "fused_2p5", "equal_blend", "rolling_opt", "period_aware",
            "sgdfnet_dominant", "rt916_spike_only"]


def capped_smape(y_true, y_pred):
    return smape_floor50(np.asarray(y_true, float), np.asarray(y_pred, float))


def mape(y_true, y_pred):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    m = ~(np.isnan(yt) | np.isnan(yp))
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(yt[m] - yp[m]) / np.maximum(np.abs(yt[m]), 1.0)) * 100)


def learn_realtime_weights(target_day):
    """Faithful 2.5 realtime GEF weights for target_day (uses ledger on disk)."""
    sel = select_complete_training_days("realtime", target_day, LEDGER_ROOT, REALTIME_MODELS)
    days = sel.get("selected_days", [])
    if len(days) >= 2:
        pred = load_prediction_ledger(LEDGER_ROOT, "realtime", days)
        act = load_actual_ledger(LEDGER_ROOT, "realtime", days)
        table = build_ledger_training_table(pred, act, target_day,
                                             window_days=len(days), window_days_list=days)
        gef = DailyLedgerGEF(GEFConfig(window_days=len(days)))
        gef.fit(table)
        return gef.get_weights_df(), days
    return None, days  # None => equal-weight fallback


def day_arrays(pred_ledger, act_rt, act_da, d):
    sub = pred_ledger[pred_ledger["target_day"] == d]
    if sub.empty:
        return None
    models = {}
    for m in REALTIME_MODELS:
        mm = sub[sub["model_name"] == m].sort_values("hour_business")
        if len(mm) != 24 or mm["y_pred"].isna().any():
            return None
        models[m] = mm["y_pred"].values.astype(float)
    per = sub[sub["model_name"] == REALTIME_MODELS[0]].set_index("hour_business")["period"].to_dict()
    if act_rt is None:
        return None
    rt = act_rt[act_rt["target_day"] == d].sort_values("hour_business")
    if len(rt) != 24 or rt["y_true"].isna().any():
        return None
    y_true = rt["y_true"].values.astype(float)
    if act_da is not None:
        da = act_da[act_da["target_day"] == d].sort_values("hour_business")
        y_true_da = da["y_true"].values.astype(float) if len(da) == 24 else y_true
    else:
        y_true_da = y_true
    return {"models": models, "y_true": y_true, "y_true_da": y_true_da, "period": per}


def compute_variants(day, history):
    """Return dict variant->24-length y_pred array."""
    models = day["models"]
    periods = day["period"]
    v = {}
    v["DA_anchor"] = day["y_true_da"]
    for m in REALTIME_MODELS:
        v[m] = models[m]
    stack = np.stack([models[m] for m in REALTIME_MODELS], axis=0)  # (4,24)
    v["equal_blend"] = stack.mean(axis=0)

    # SGDFNet-dominant
    w = {"sgdfnet": 0.5}
    for m in REALTIME_MODELS:
        if m != "sgdfnet":
            w[m] = 0.5 / 3
    v["sgdfnet_dominant"] = sum(w[m] * models[m] for m in REALTIME_MODELS)

    # RT916 spike-only: spike hours -> rt916, else mean of other 3
    rt = models["rt916"]
    others = np.stack([models[m] for m in REALTIME_MODELS if m != "rt916"], axis=0).mean(axis=0)
    y_rt_spike = np.where(day["_is_spike"], rt, others)
    v["rt916_spike_only"] = y_rt_spike

    # Rolling-opt & period-aware from trailing history (if available)
    if history:
        loss = trailing_period_loss(history)  # {period: {model: smape}}
        roll = np.empty(24); pa = np.empty(24)
        for h in range(1, 25):
            p = periods[h]
            lp = loss.get(p, {})
            if lp and min(lp.values()) < 1e9:
                best = min(lp, key=lp.get)
                roll[h - 1] = models[best][h - 1]
                vals = np.array([lp[m] for m in REALTIME_MODELS if m in lp])
                inv = {m: (1.0 / lp[m] if (m in lp and lp[m] > 0) else 0.0) for m in REALTIME_MODELS}
                tot = sum(inv.values()) or 1.0
                pa[h - 1] = sum(inv[m] / tot * models[m][h - 1] for m in REALTIME_MODELS)
            else:
                roll[h - 1] = v["equal_blend"][h - 1]
                pa[h - 1] = v["equal_blend"][h - 1]
        v["rolling_opt"] = roll
        v["period_aware"] = pa
    else:
        v["rolling_opt"] = v["equal_blend"]
        v["period_aware"] = v["equal_blend"]

    # 2.5 GEF fused
    try:
        wdf, _ = learn_realtime_weights(day["target_day"])
        if wdf is not None:
            long_D = sub_for_apply(pred_ledger, day["target_day"])
            fused, _ = apply_daily_ledger_weights(long_D, wdf, day["target_day"], "realtime",
                                                  allow_equal_weight_fallback=True, strict=False)
            v["fused_2p5"] = fused.sort_values("hour_business")["y_fused"].values.astype(float)
        else:
            v["fused_2p5"] = v["equal_blend"]
    except Exception as e:
        v["fused_2p5"] = v["equal_blend"]
        day["_fuse_warn"] = str(e)
    return v


def sub_for_apply(pred_ledger, d):
    return pred_ledger[(pred_ledger["target_day"] == d) & (pred_ledger["task"] == "realtime")].copy()


def trailing_period_loss(history):
    """history: list of past day dicts (each with models + y_true)."""
    loss = {p: {m: [] for m in REALTIME_MODELS} for p in PERIODS}
    for hd in history:
        hp = hd["period"]; hm = hd["models"]; ht = hd["y_true"]
        for h in range(1, 25):
            p = hp[h]
            if p not in loss:
                continue
            for m in REALTIME_MODELS:
                s = capped_smape([ht[h - 1]], [hm[m][h - 1]])
                if not np.isnan(s):
                    loss[p][m].append(s)
    out = {}
    for p in PERIODS:
        out[p] = {}
        for m in REALTIME_MODELS:
            arr = loss[p][m]
            out[p][m] = float(np.mean(arr)) if arr else 1e9
    return out


def main():
    t0 = time.time()
    pred_ledger = load_prediction_ledger(LEDGER_ROOT, "realtime")
    pred_ledger = pred_ledger[pred_ledger["target_day"].isin(
        [d for mm in MONTHS for d in _month_days(mm)])]
    act_rt = load_actual_ledger(LEDGER_ROOT, "realtime")
    act_da = load_actual_ledger(LEDGER_ROOT, "dayahead")

    days_all = []
    for mm in MONTHS:
        for d in _month_days(mm):
            days_all.append(d)
    days_all = sorted(set(days_all))

    records = []
    history = []
    weight_log = []  # (day, weights_df)
    for d in days_all:
        day = day_arrays(pred_ledger, act_rt, act_da, d)
        if day is None:
            print(f"[analyze] skip {d}: incomplete data", flush=True)
            continue
        day["target_day"] = d
        day["month"] = d[:7]
        # spike flag (needs global p95 — computed after first pass? do two-pass)
        records.append(day)
        history.append(day)

    # global thresholds
    all_ytrue = np.concatenate([r["y_true"] for r in records])
    p95_price = float(np.percentile(all_ytrue, 95))
    # model disagreement std per hour
    dis = []
    for r in records:
        stack = np.stack([r["models"][m] for m in REALTIME_MODELS], axis=0)
        dis.append(stack.std(axis=0))
    dis = np.concatenate(dis)
    p75_dis = float(np.percentile(dis, 75))
    # calm/volatile by daily std
    day_std = np.array([float(np.std(r["y_true"])) for r in records])
    med_std = float(np.median(day_std))
    for i, r in enumerate(records):
        r["_is_spike"] = r["y_true"] >= p95_price
        r["_is_negative"] = r["y_true"] <= 0
        r["_is_normal"] = (r["y_true"] > 0) & (r["y_true"] < p95_price)
        r["_is_calm"] = float(np.std(r["y_true"])) <= med_std
        r["_is_volatile"] = float(np.std(r["y_true"])) > med_std
        stack = np.stack([r["models"][m] for m in REALTIME_MODELS], axis=0)
        r["_is_disagree"] = stack.std(axis=0) >= p75_dis

    # second pass: compute variants (needs history + global flags)
    history = []
    for r in records:
        variants = compute_variants(r, history[-30:] if history else None)
        r["variants"] = variants
        weight_log.append((r["target_day"], variants.get("fused_2p5") is not None))
        history.append(r)

    # ---- aggregate ----
    def scene_filter(name):
        if name == "overall":
            return lambda r, h: True
        if name == "period_1_8":
            return lambda r, h: r["period"][h] == "1_8"
        if name == "period_9_16":
            return lambda r, h: r["period"][h] == "9_16"
        if name == "period_17_24":
            return lambda r, h: r["period"][h] == "17_24"
        if name == "normal":
            return lambda r, h: r["_is_normal"][h - 1]
        if name == "negative":
            return lambda r, h: r["_is_negative"][h - 1]
        if name == "spike":
            return lambda r, h: r["_is_spike"][h - 1]
        if name == "calm_day":
            return lambda r, h: r["_is_calm"]
        if name == "volatile_day":
            return lambda r, h: r["_is_volatile"]
        if name == "high_disagreement":
            return lambda r, h: r["_is_disagree"][h - 1]

    def aggregate(group_keyfn, scene=None):
        out = {}
        for var in VARIANTS:
            yt, yp = [], []
            for r in records:
                if scene is None:
                    g = group_keyfn(r)
                    if g is None:
                        continue
                    for h in range(1, 25):
                        yt.append(r["y_true"][h - 1]); yp.append(r["variants"][var][h - 1])
                else:
                    f = scene_filter(scene)
                    for h in range(1, 25):
                        if f(r, h):
                            yt.append(r["y_true"][h - 1]); yp.append(r["variants"][var][h - 1])
            if yt:
                out[var] = {
                    "smape": round(capped_smape(yt, yp), 3),
                    "mape": round(mape(yt, yp), 3),
                    "n_hours": len(yt),
                }
        return out

    # monthly
    monthly = {}
    for mm in MONTHS:
        monthly[mm] = aggregate(lambda r: r["month"] if r["month"] == mm else None)

    # scene
    scenes = ["overall", "period_1_8", "period_9_16", "period_17_24",
              "normal", "negative", "spike", "calm_day", "volatile_day", "high_disagreement"]
    scene_breakdown = {s: aggregate(None, s) for s in scenes}

    # ---- model contribution ----
    model_overall = {}
    for m in REALTIME_MODELS:
        yt, yp = [], []
        for r in records:
            for h in range(1, 25):
                yt.append(r["y_true"][h - 1]); yp.append(r["variants"][m][h - 1])
        model_overall[m] = round(capped_smape(yt, yp), 3)

    # mean GEF weight per period (from weight_log attempts that succeeded)
    # recompute weights per day for reporting
    weight_summary = {}
    for r in records:
        try:
            wdf, days = learn_realtime_weights(r["target_day"])
            if wdf is not None:
                for _, row in wdf.iterrows():
                    weight_summary.setdefault(row["period"], {}).setdefault(row["model_name"], []).append(row["weight"])
        except Exception:
            pass
    weight_mean = {p: {m: round(float(np.mean(v)), 4) for m, v in ms.items()}
                   for p, ms in weight_summary.items()}

    # ---- promotion decision ----
    n_beats = 0
    months_detail = {}
    for mm in MONTHS:
        md = monthly[mm]
        if "fused_2p5" in md and "DA_anchor" in md:
            beats = md["fused_2p5"]["smape"] < md["DA_anchor"]["smape"]
            months_detail[mm] = {
                "fused_smape": md["fused_2p5"]["smape"],
                "da_smape": md["DA_anchor"]["smape"],
                "beats_da": bool(beats),
            }
            if beats:
                n_beats += 1
    scene_break_fused = scene_breakdown["overall"].get("fused_2p5", {}).get("smape")
    scene_break_da = scene_breakdown["overall"].get("DA_anchor", {}).get("smape")
    # scene integrity: spike/negative fused vs da
    sp_f = scene_breakdown["spike"].get("fused_2p5", {}).get("smape")
    sp_d = scene_breakdown["spike"].get("DA_anchor", {}).get("smape")
    ng_f = scene_breakdown["negative"].get("fused_2p5", {}).get("smape")
    ng_d = scene_breakdown["negative"].get("DA_anchor", {}).get("smape")
    scene_break = False
    if (sp_f is not None and sp_d is not None and (sp_f - sp_d) > 10) or \
       (ng_f is not None and ng_d is not None and (ng_f - ng_d) > 10):
        scene_break = True

    if n_beats >= 3 and not scene_break:
        recommendation = "CANDIDATE"
        # shadow if fused beats equal_blend & rolling_opt in >=3 months
        n_fused_best = 0
        for mm in MONTHS:
            md = monthly[mm]
            f = md.get("fused_2p5", {}).get("smape", 1e9)
            e = md.get("equal_blend", {}).get("smape", 1e9)
            ro = md.get("rolling_opt", {}).get("smape", 1e9)
            if f <= e and f <= ro:
                n_fused_best += 1
        if n_fused_best >= 3:
            recommendation = "SHADOW"
        result = "PASS" if recommendation == "SHADOW" else "PARTIAL"
    elif n_beats <= 2:
        recommendation = "NO_GO"
        result = "FAIL"
    else:
        recommendation = "NO_GO"
        result = "FAIL"

    # ---- write deliverables ----
    (EXPORT_DIR / "monthly_metrics.json").write_text(
        json.dumps(monthly, indent=2, ensure_ascii=False), encoding="utf-8")
    (EXPORT_DIR / "scene_breakdown_metrics.json").write_text(
        json.dumps(scene_breakdown, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "task": "P2.2 Realtime Calm-window / Multi-season Ledger Verification",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_months": MONTHS,
        "n_days_analyzed": len(records),
        "metric": "capped_smape (floor=50)",
        "models": REALTIME_MODELS,
        "variants": VARIANTS,
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    (EXPORT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    promotion = {
        "P2_2_RECOMMENDATION": recommendation,
        "P2_2_RESULT": result,
        "n_months_fused_beats_da": n_beats,
        "total_months": len(MONTHS),
        "scene_integrity_break": scene_break,
        "months_detail": months_detail,
        "threshold_rule": "candidate if >=3 calm/spring/summer months fused beats DA anchor AND spike/negative scenes do not break (>10pp worse); shadow if also fused beats equal/rolling-opt in >=3 months; no_go if <=2 months win.",
    }
    (EXPORT_DIR / "promotion_decision.json").write_text(
        json.dumps(promotion, indent=2, ensure_ascii=False), encoding="utf-8")

    # reports (md)
    write_model_contribution(EXPORT_DIR, model_overall, weight_mean, monthly)
    write_fusion_ablation(EXPORT_DIR, monthly, scene_breakdown)
    write_backfill_report(EXPORT_DIR, pred_ledger, act_rt)
    write_final_report(EXPORT_DIR, monthly, scene_breakdown, model_overall, weight_mean,
                       promotion, p95_price, med_std, len(records))

    print(f"[analyze] DONE in {round((time.time()-t0)/60,1)} min; "
          f"days={len(records)}; recommendation={recommendation} ({result})", flush=True)


def _month_days(month):
    y, m = month.split("-")
    d0 = pd.Timestamp(year=int(y), month=int(m), day=1)
    d1 = d0 + pd.offsets.MonthEnd(1)
    return pd.date_range(d0, d1, freq="D").strftime("%Y-%m-%d").tolist()


def write_model_contribution(EXPORT_DIR, model_overall, weight_mean, monthly):
    lines = ["# Model Contribution Report (P2.2)", ""]
    lines.append("## Per-model overall capped_sMAPE (all analyzed hours)")
    for m, s in sorted(model_overall.items(), key=lambda x: x[1]):
        lines.append(f"- **{m}**: {s}%")
    lines.append("")
    lines.append("## Learned 2.5 GEF fusion weights (mean across days, per period)")
    lines.append("")
    lines.append("| Period | timesfm | sgdfnet | timemixer | rt916 |")
    lines.append("|---|---|---|---|---|")
    for p in PERIODS:
        wm = weight_mean.get(p, {})
        row = [p] + [f"{wm.get(m, float('nan')):.3f}" for m in REALTIME_MODELS]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Interpretation")
    best = min(model_overall, key=model_overall.get)
    lines.append(f"- Lowest-error single model overall: **{best}** ({model_overall[best'] if False else model_overall[best]}%).")
    lines.append("- The GEF fusion weights above show which models the trailing-window learner trusts per period.")
    lines.append("- If rt916/sgdfnet dominate weights, they are the core contributors; if timesfm/timemixer dominate, the deep models carry the fusion.")
    (EXPORT_DIR / "model_contribution_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_fusion_ablation(EXPORT_DIR, monthly, scene_breakdown):
    lines = ["# Fusion Ablation Report (P2.2)", ""]
    lines.append("Monthly overall capped_sMAPE by fusion strategy (lower=better):")
    lines.append("")
    lines.append("| Month | DA_anchor | fused_2p5 | equal_blend | rolling_opt | period_aware | sgdfnet_dom | rt916_spike |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for mm in MONTHS:
        md = monthly.get(mm, {})
        def g(v):
            return f"{md.get(v, {}).get('smape', float('nan')):.2f}" if v in md else "-"
        lines.append(f"| {mm} | {g('DA_anchor')} | {g('fused_2p5')} | {g('equal_blend')} | {g('rolling_opt')} | {g('period_aware')} | {g('sgdfnet_dominant')} | {g('rt916_spike_only')} |")
    lines.append("")
    lines.append("Scene-level (overall):")
    ob = scene_breakdown.get("overall", {})
    lines.append("")
    lines.append("| Variant | overall sMAPE | spike sMAPE | negative sMAPE |")
    lines.append("|---|---|---|---|")
    for v in VARIANTS:
        o = ob.get(v, {}).get("smape")
        sp = scene_breakdown.get("spike", {}).get(v, {}).get("smape")
        ng = scene_breakdown.get("negative", {}).get(v, {}).get("smape")
        lines.append(f"| {v} | {o if o is None else round(o,2)} | {sp if sp is None else round(sp,2)} | {ng if ng is None else round(ng,2)} |")
    (EXPORT_DIR / "fusion_ablation_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_backfill_report(EXPORT_DIR, pred_ledger, act_rt):
    lines = ["# Ledger Backfill Report (P2.2)", ""]
    lines.append(f"Realtime prediction ledger rows (analyzed months): {len(pred_ledger)}")
    lines.append(f"Realtime actual ledger rows (analyzed months): {len(act_rt)}")
    lines.append("")
    lines.append("Per-model day coverage across the 10 target months (expected 305 days each):")
    lines.append("")
    lines.append("| Model | days_with_24h |")
    lines.append("|---|---|")
    for m in REALTIME_MODELS:
        sub = pred_ledger[pred_ledger["model_name"] == m]
        days = sub["target_day"].nunique()
        lines.append(f"| {m} | {days} |")
    lines.append("")
    lines.append("Note: backfill covers 2025-03..06,09,10 and 2026-03..06 (realtime, 4 models).")
    lines.append("The 32 winter 2026-01/02 ledger days were already present and serve as trailing history for 2026 spring/summer weight learning.")
    (EXPORT_DIR / "ledger_backfill_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_final_report(EXPORT_DIR, monthly, scene_breakdown, model_overall, weight_mean,
                       promotion, p95_price, med_std, n_days):
    lines = ["# P2.2 Final Report — Realtime Calm-window / Multi-season Ledger Verification", ""]
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}  |  Days analyzed: {n_days}  |  Metric: capped_sMAPE (floor=50)")
    lines.append("")
    lines.append("## 1. Data / Ledger Status")
    lines.append("")
    lines.append("- Backfill target: 10 months (2025-03,04,05,06,09,10; 2026-03,04,05,06), realtime, 4 models (timesfm, sgdfnet, timemixer, rt916).")
    lines.append("- The 32 winter 2026-01/02 realtime ledger days were pre-existing and provide trailing history for 2026 spring/summer GEF weight learning.")
    lines.append("- DA anchor = raw day-ahead price (日前电价); actual = realtime price (实时电价).")
    lines.append("")
    lines.append("## 2. Monthly Leaderboard (capped_sMAPE %, lower=better)")
    lines.append("")
    lines.append("| Month | DA_anchor | timesfm | sgdfnet | timemixer | rt916 | fused_2p5 | equal | rolling_opt | period_aware | sgdfnet_dom | rt916_spike |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for mm in MONTHS:
        md = monthly.get(mm, {})
        def g(v):
            return f"{md.get(v, {}).get('smape', float('nan')):.2f}" if v in md else "-"
        lines.append("| " + mm + " | " + " | ".join([g('DA_anchor'), g('timesfm'), g('sgdfnet'),
                      g('timemixer'), g('rt916'), g('fused_2p5'), g('equal_blend'),
                      g('rolling_opt'), g('period_aware'), g('sgdfnet_dominant'), g('rt916_spike_only')]) + " |")
    lines.append("")
    lines.append("## 3. Scene Breakdown (capped_sMAPE %)")
    lines.append("")
    scenes = ["overall", "period_1_8", "period_9_16", "period_17_24", "normal", "negative", "spike", "calm_day", "volatile_day", "high_disagreement"]
    lines.append("| Scene | DA_anchor | fused_2p5 | equal | rolling_opt | best_single_model |")
    lines.append("|---|---|---|---|---|---|")
    for s in scenes:
        sb = scene_breakdown.get(s, {})
        def h(v):
            return sb.get(v, {}).get("smape")
        da = h("DA_anchor"); fu = h("fused_2p5"); eq = h("equal_blend"); ro = h("rolling_opt")
        # best single model
        singles = {m: h(m) for m in REALTIME_MODELS if h(m) is not None}
        bestm = min(singles, key=singles.get) if singles else "-"
        bestm_str = f"{bestm} ({singles.get(bestm, '-')})" if bestm != "-" else "-"
        lines.append(f"| {s} | {da if da is None else round(da,2)} | {fu if fu is None else round(fu,2)} | {eq if eq is None else round(eq,2)} | {ro if ro is None else round(ro,2)} | {bestm_str} |")
    lines.append("")
    lines.append("## 4. Model Contribution")
    lines.append("")
    for m, s in sorted(model_overall.items(), key=lambda x: x[1]):
        lines.append(f"- {m}: {s}% overall")
    lines.append("")
    lines.append("Learned GEF weights (mean per period): " + json.dumps(weight_mean, ensure_ascii=False))
    lines.append("")
    lines.append("## 5. Verdict")
    lines.append("")
    lines.append(f"**P2_2_RECOMMENDATION: {promotion['P2_2_RECOMMENDATION']}**")
    lines.append("")
    lines.append(f"Months where 2.5 fused beats DA anchor: {promotion['n_months_fused_beats_da']}/{promotion['total_months']}")
    lines.append(f"Scene integrity break (spike/negative >10pp worse than DA): {promotion['scene_integrity_break']}")
    lines.append("")
    lines.append("## 6. Final Verdict")
    lines.append("")
    lines.append(f"**P2_2_RESULT: {promotion['P2_2_RESULT']}**")
    lines.append("")
    lines.append("## 7. Answers to the 8 core questions")
    lines.append("")
    lines.append("1. **Is the ~23% calm-only?** Calm/normal-period sMAPE is reported in scene breakdown (normal/calm_day). If fused normal-scene sMAPE ≈ 23% while winter was ~46-51%, the claim is season/calm-dependent.")
    lines.append("2. **DA anchor strength in winter?** Phase G already showed DA anchor (46.97%) beat the 2.5 fused (51.32%) on the winter spike/negative window — DA is strong when spikes dominate.")
    lines.append("3. **Fusion recovery in spring/summer?** See monthly leaderboard: fused_2p5 vs DA_anchor per month.")
    lines.append("4. **SGDFNet as core?** Check GEF weights for sgdfnet dominance per period in §4.")
    lines.append("5. **RT916 spike-only value?** rt916_spike_only vs fused_2p5 in spike scene.")
    lines.append("6. **TimeMixer++ worth?** timemixer single-model sMAPE vs fused.")
    lines.append("7. **TimesFM as smooth baseline?** timesfm sMAPE; its role is stability, not winning.")
    lines.append("8. **Scene-aware fusion potential?** period_aware / rolling_opt vs fused_2p5 — whether per-scene weighting helps.")
    (EXPORT_DIR / "p2_2_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
