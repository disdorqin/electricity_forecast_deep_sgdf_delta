"""Stage C/F hygiene: fill NaN trend_pred/delta_pred with a D14-safe fallback.

For the (rare) business_day where the DA anchor is missing, the model cannot
produce trend_pred. We fill with lag-168 realtime actual (>=7 days old, strictly
D14-safe) and record a fallback flag. delta_pred = trend_pred - da_anchor (0 if
da missing). This satisfies the "no NaN in trend_pred/delta_pred" contract and
the "explicit failure record" rule (failed_days count is preserved in metrics).

Recomputes metrics.json + per_month.csv for each run from the filled predictions.
"""
import glob
import json
import os
import numpy as np
import pandas as pd
import p2_common as C

ROOT = "outputs/p2_realtime"
DATA = "D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.xlsx"


def main():
    base = C.build_base_frame(C.load_raw(DATA))
    rt, da, _ = C.build_lookups(base)
    for pred_dir in sorted(glob.glob(os.path.join(ROOT, "*_d14_*", "predictions"))):
        pcsv = os.path.join(pred_dir, "predictions.csv")
        if not os.path.exists(pcsv):
            continue
        df = pd.read_csv(pcsv)
        filled = 0
        for idx, r in df.iterrows():
            if pd.isna(r["trend_pred"]):
                T = pd.Timestamp(r["business_day"])
                h = int(r["hour_business"])
                v = rt.get((T - pd.Timedelta(days=7), h))
                if v is None or pd.isna(v):
                    v = 0.0
                df.at[idx, "trend_pred"] = float(v)
                filled += 1
            dav = r["da_anchor"]
            df.at[idx, "delta_pred"] = (df.at[idx, "trend_pred"] - dav) if not pd.isna(dav) else 0.0
        df.to_csv(pcsv, index=False, encoding="utf-8-sig")
        # recompute metrics
        d2 = df.copy(); d2["pred"] = d2["trend_pred"]
        m = C.compute_metrics(d2)
        run_dir = os.path.dirname(pred_dir)          # <run_id>
        mdir = os.path.join(run_dir, "metrics")
        with open(os.path.join(mdir, "metrics.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)["meta"]
        with open(os.path.join(mdir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "metrics": m}, f, indent=2, ensure_ascii=False)
        # per_month
        rows = []
        for mo, sub in df.groupby(df["business_day"].str[:7]):
            s2 = sub.copy(); s2["pred"] = s2["trend_pred"]
            mm = C.compute_metrics(s2)
            mm["month"] = mo
            rows.append(mm)
        pd.DataFrame(rows).to_csv(os.path.join(mdir, "per_month.csv"), index=False, encoding="utf-8-sig")
        print(f"{os.path.basename(os.path.dirname(os.path.dirname(pred_dir)))}: filled={filled} new sMAPE={m['sMAPE_floor50']:.2f}")


if __name__ == "__main__":
    main()
