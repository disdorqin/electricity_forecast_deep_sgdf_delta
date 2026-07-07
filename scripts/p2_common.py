"""P2 Realtime — shared framework utilities (D14 cutoff-safe).

Single source of truth for:
  * data loading + business_time alignment (hour_business 1..24, segment 1_8/9_16/17_24)
  * D14-cutoff-safe feature assembly (walk-forward, monthly retrain)
  * deep model registry (TCN / GRU / DLinear / Linear day-level decoders)
  * unified metrics (MAE / RMSE / sMAPE_floor50 + period / spike / negative)
  * cutoff audit
  * output writers (predictions csv / metrics json / report md)

Cutoff rule (D14): when predicting target day T, only information visible at
decision day D = T-1, 14:00 may be used:
  - all realtime actuals with business_day < T-1 (full past days)
  - realtime actuals of business_day == T-1 with hour_business <= 14
  - DA price for T (forecast)
  - forecast-side load/renewable predictions for T (published before cutoff)
Never: T-1 hours 15..24, target-day actuals, or any future label as online feature.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Column contract ───────────────────────────────────────────────────
DA_COL = "日前电价"
RT_COL = "实时电价"
FORECAST_COLS = [
    "直调负荷预测值",
    "竞价空间预测值",
    "风电总加预测值",
    "光伏总加预测值",
    "联络线受电负荷预测值",
]
DECISION_HOUR = 14  # D14 cutoff


# ── Data loading + business time ──────────────────────────────────────
def load_raw(path: str) -> pd.DataFrame:
    if str(path).endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="gbk")
    return df


def add_business_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["时刻"])
    df["hour"] = ts.dt.hour
    df["hour_business"] = np.where(df["hour"] == 0, 24, df["hour"]).astype(int)
    df["business_day"] = (ts - pd.Timedelta(hours=1)).dt.normalize()
    df["ds"] = ts
    df["month"] = ts.dt.month.astype(int)
    df["weekday"] = ts.dt.weekday.astype(int)
    seg = pd.cut(
        df["hour_business"],
        bins=[0, 8, 16, 24],
        labels=["1_8", "9_16", "17_24"],
        include_lowest=True,
        right=True,
    )
    df["segment"] = seg.astype(str)
    df["segment_id"] = seg.cat.codes.astype(int)
    return df


def build_base_frame(raw: pd.DataFrame) -> pd.DataFrame:
    cols = ["business_day", "hour_business", "ds", DA_COL, RT_COL,
            "month", "weekday", "segment", "segment_id"] + FORECAST_COLS
    base = add_business_time(raw)[cols].copy()
    base[DA_COL] = pd.to_numeric(base[DA_COL], errors="coerce")
    base[RT_COL] = pd.to_numeric(base[RT_COL], errors="coerce")
    for c in FORECAST_COLS:
        base[c] = pd.to_numeric(base[c], errors="coerce")
    base = base.sort_values(["business_day", "hour_business"]).reset_index(drop=True)
    return base


# ── Feature assembly (D14-safe) ───────────────────────────────────────
@dataclass
class FeatureSpec:
    lags: tuple = (24, 48, 72, 96, 120, 144, 168)  # days; lag-24 gets visibility flag (only h<=14 visible)
    use_trajectory: bool = True          # visible D-1 hours 1..14 (rt-da)
    use_forecast_side: bool = True
    use_calendar: bool = True


def _segment_id_of_hour(h: int) -> int:
    if h <= 8:
        return 0
    if h <= 16:
        return 1
    return 2


def build_lookups(base: pd.DataFrame):
    rt = {}
    da = {}
    fc = {c: {} for c in FORECAST_COLS}
    for r in base.to_dict("records"):
        k = (r["business_day"], int(r["hour_business"]))
        rt[k] = r[RT_COL]
        da[k] = r[DA_COL]
        for c in FORECAST_COLS:
            fc[c][k] = r[c]
    return rt, da, fc


def assemble_day_features(T_bd: pd.Timestamp, rt, da, fc, spec: FeatureSpec, target_is_abs: bool = False):
    """Return (X[24,F] float32, y[24] float32 or nan, audit_ts).

    X uses only D14-visible information. y is delta (rt-da) unless target_is_abs.
    y is nan where target rt missing.
    """
    T_bd = pd.Timestamp(T_bd).normalize()
    D = T_bd - pd.Timedelta(days=1)          # decision day
    # visible trajectory: D hours 1..14 (rt - da)
    traj = np.zeros(14, dtype=np.float32)
    if spec.use_trajectory:
        ok = True
        for k in range(1, 15):
            r = rt.get((D, k))
            d = da.get((D, k))
            if r is None or d is None or pd.isna(r) or pd.isna(d):
                ok = False
                break
            traj[k - 1] = r - d
        if not ok:
            traj = np.zeros(14, dtype=np.float32)

    feats = []
    y = np.full(24, np.nan, dtype=np.float32)
    for h in range(1, 25):
        row = []
        d_h = da.get((T_bd, h))
        row.append(0.0 if d_h is None or pd.isna(d_h) else float(d_h))  # da anchor
        # lag features (multi-day); lag<=24 masked after cutoff hour
        for L in spec.lags:
            src = T_bd - pd.Timedelta(days=L)
            v = rt.get((src, h))
            if L <= 24 and h > DECISION_HOUR:
                val = 0.0  # post-cutoff realtime actual -> masked (no leakage)
            else:
                val = 0.0 if (v is None or pd.isna(v)) else float(v)
            row.append(val)
            if L <= 24:
                flag = 1.0 if (h <= DECISION_HOUR and v is not None and not pd.isna(v)) else 0.0
                row.append(flag)
        # trajectory (same 14-dim for all hours)
        if spec.use_trajectory:
            row.extend(traj.tolist())
        # forecast-side
        if spec.use_forecast_side:
            for c in FORECAST_COLS:
                v = fc[c].get((T_bd, h))
                row.append(0.0 if v is None or pd.isna(v) else float(v))
        # calendar
        if spec.use_calendar:
            row.append(float(T_bd.month))
            row.append(float(T_bd.weekday()))
            row.append(float(h))
            row.append(float(_segment_id_of_hour(h)))
        feats.append(row)
        # target
        r_h = rt.get((T_bd, h))
        d_h = da.get((T_bd, h))
        if r_h is not None and d_h is not None and not pd.isna(r_h) and not pd.isna(d_h):
            y[h - 1] = r_h if target_is_abs else (r_h - d_h)
    X = np.array(feats, dtype=np.float32)
    # cutoff audit: latest visible realtime timestamp
    audit_ts = D + pd.Timedelta(hours=DECISION_HOUR)
    return X, y, audit_ts


def feature_dim(spec: FeatureSpec) -> int:
    d = 1  # da anchor
    for L in spec.lags:
        d += 1
        if L <= 24:
            d += 1  # visibility flag
    if spec.use_trajectory:
        d += 14
    if spec.use_forecast_side:
        d += len(FORECAST_COLS)
    if spec.use_calendar:
        d += 4
    return d


# ── Metrics ───────────────────────────────────────────────────────────
def capped_smape(y_true, y_pred, floor: float = 50.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return float("nan")
    yt = np.maximum(np.abs(y_true[mask]), floor)
    yp = np.maximum(np.abs(y_pred[mask]), floor)
    return float(np.mean(200.0 * np.abs(y_true[mask] - y_pred[mask]) / (yt + yp)))


def _mae_rmse(yt, yp):
    yt = np.asarray(yt, dtype=float)
    yp = np.asarray(yp, dtype=float)
    m = ~(np.isnan(yt) | np.isnan(yp))
    if m.sum() == 0:
        return float("nan"), float("nan")
    err = yt[m] - yp[m]
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))


def compute_metrics(df: pd.DataFrame) -> dict:
    """df must have columns: y_true, trend_pred (or pred), segment, is_spike, is_negative."""
    pred_col = "pred" if "pred" in df.columns else "trend_pred"
    seg_col = "segment" if "segment" in df.columns else "period"
    yt = df["y_true"].to_numpy(dtype=float)
    yp = df[pred_col].to_numpy(dtype=float)
    out = {}
    out["MAE"], out["RMSE"] = _mae_rmse(yt, yp)
    out["sMAPE_floor50"] = capped_smape(yt, yp)
    for seg in ["1_8", "9_16", "17_24"]:
        m = df[seg_col] == seg
        out[f"sMAPE_{seg}"] = capped_smape(df.loc[m, "y_true"], df.loc[m, pred_col])
    sp = df["is_spike"].to_numpy(bool)
    ng = df["is_negative"].to_numpy(bool)
    out["spike_sMAPE_floor50"] = capped_smape(df.loc[sp, "y_true"], df.loc[sp, pred_col])
    out["negative_sMAPE_floor50"] = capped_smape(df.loc[ng, "y_true"], df.loc[ng, pred_col])
    out["normal_count"] = int((~sp & ~ng).sum())
    out["spike_count"] = int(sp.sum())
    out["negative_count"] = int(ng.sum())
    out["nan_count"] = int(np.isnan(yp).sum())
    out["failed_days"] = int(df.groupby("business_day")[pred_col].apply(lambda s: np.isnan(s).any()).sum())
    out["missing_hour_count"] = int((df.groupby(["business_day", "hour_business"]).size() < 1).sum())
    return out


# ── Deep models (day-level 24h decoder) ───────────────────────────────
class TCN(nn.Module):
    def __init__(self, feat_dim, hidden=64, layers=2, kernel=3, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Sequential(
            nn.Conv1d(feat_dim, hidden, kernel, padding=(kernel - 1) // 2, dilation=1),
            nn.ReLU(), nn.Dropout(dropout)))
        for i in range(1, layers):
            dil = 2 ** i
            self.blocks.append(nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel, padding=dil * (kernel - 1) // 2, dilation=dil),
                nn.ReLU(), nn.Dropout(dropout)))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # x: [B, 24, F]
        h = x.transpose(1, 2)            # [B, F, 24]
        for b in self.blocks:
            h = b(h)                     # [B, hidden, 24]
        h = h.transpose(1, 2)            # [B, 24, hidden]
        return self.head(h).squeeze(-1)  # [B, 24]


class GRU(nn.Module):
    def __init__(self, feat_dim, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(h).squeeze(-1)


class DLinear(nn.Module):
    """Decomposition Linear (trend via mean over time + remainder), per-hour head."""
    def __init__(self, feat_dim, hidden=64):
        super().__init__()
        self.trend = nn.Linear(feat_dim, hidden)
        self.season = nn.Linear(feat_dim, hidden)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x):  # x: [B, 24, F]
        trend = x.mean(dim=1, keepdim=True)                      # [B,1,F]
        season = x - trend                                       # [B,24,F]
        t = self.trend(trend.squeeze(1)).unsqueeze(1).expand(-1, x.size(1), -1)  # [B,24,hidden]
        s = self.season(season)                                  # [B,24,hidden]
        h = torch.cat([t, s], dim=-1)                            # [B,24,2*hidden]
        return self.head(h).squeeze(-1)                          # [B,24]


class LinearDay(nn.Module):
    def __init__(self, feat_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


MODEL_REGISTRY = {
    "tcn_day": lambda f, **k: TCN(f, **k),
    "gru_day": lambda f, **k: GRU(f, **k),
    "dlinear_day": lambda f, hidden=64, **k: DLinear(f, hidden=hidden),
    "linear_day": lambda f, hidden=64, **k: LinearDay(f, hidden=hidden),
}


# ── Training (monthly walk-forward) ───────────────────────────────────
def train_monthly(model, Xtr, ytr, Xval, yval, device, epochs=30, lr=1e-3,
                  batch_size=64, patience=5, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr_m = ~np.isnan(ytr).any(axis=1)
    va_m = ~np.isnan(yval).any(axis=1)
    if tr_m.sum() == 0:
        return
    Xtr = torch.tensor(Xtr[tr_m], dtype=torch.float32).to(device)
    ytr = torch.tensor(ytr[tr_m], dtype=torch.float32).to(device)
    Xval = torch.tensor(Xval[va_m], dtype=torch.float32).to(device)
    yval = torch.tensor(yval[va_m], dtype=torch.float32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = torch.utils.data.TensorDataset(Xtr, ytr)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    best = float("inf")
    best_state = None
    bad = 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = torch.nanmean((pred - yb) ** 2)
            if torch.isnan(loss):
                continue
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xval)
            vloss = torch.nanmean((vp - yval) ** 2).item()
        if vloss < best:
            best = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)


def predict_month(model, X, device):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32).to(device)
        pred = model(Xt).cpu().numpy()
    return pred  # [N_days, 24]


# ── Output writers ────────────────────────────────────────────────────
def write_predictions_csv(path: Path, rows: list[dict]):
    cols = ["business_day", "ds", "hour_business", "period", "da_anchor",
            "delta_pred", "trend_pred", "model_name", "model_version",
            "confidence", "run_id", "y_true", "spike_pred", "negative_pred",
            "residual_for_spike_module", "residual_for_negative_module",
            "is_spike", "is_negative"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def write_metrics_json(path: Path, metrics: dict, meta: dict):
    payload = {"meta": meta, "metrics": metrics}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
