"""Multi-month walk-forward backtest for DeepRT-SOTA v2."""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.deep_sgdf_delta.business_time import add_business_time_columns
from models.deep_sgdf_delta.deep_rt_sota_dataset import (
    DeepRTSOTADatasetConfig,
    build_deep_rt_sota_dataset,
)
from models.deep_sgdf_delta.deep_rt_sota_features import (
    build_deep_rt_sota_features,
    get_feature_columns,
)
from models.deep_sgdf_delta.deep_rt_sota_model import (
    DeepRTSOTAModel,
    DeepRTSOTAModelConfig,
)
from models.deep_sgdf_delta.metrics import smape_floor50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--target-months", type=str, required=True,
                        help="Comma-separated months, e.g., 2026-01,2026-02")
    parser.add_argument("--model-profile", type=str, default="deep_rt_tcn")
    parser.add_argument("--target-granularity", type=str, default="day")
    parser.add_argument("--target-mode", type=str, default="residual_to_da")
    parser.add_argument("--seq-len-days", type=int, default=7)
    parser.add_argument("--risk-features", type=str, default="off")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    """Load and preprocess data."""
    for enc in ["gbk", "gb18030", "utf-8"]:
        try:
            df = pd.read_csv(data_path, encoding=enc)
            print(f"  Loaded {len(df)} rows (encoding: {enc})")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not read {data_path} with any encoding")
    
    rename_map = {}
    if "时刻" in df.columns and "ds" not in df.columns:
        rename_map["时刻"] = "ds"
    if "日前电价" in df.columns and "da_anchor" not in df.columns:
        rename_map["日前电价"] = "da_anchor"
    if "实时电价" in df.columns and "rt_actual" not in df.columns:
        rename_map["实时电价"] = "rt_actual"
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"  Renamed columns: {rename_map}")

    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"])
    else:
        raise ValueError("Column 'ds' (timestamp) not found")

    df = add_business_time_columns(df, timestamp_col="ds")
    return df


def train_and_evaluate_month(args, df, target_month, out_dir):
    """Train and evaluate for a single target month (walk-forward)."""
    print(f"\n{'='*80}")
    print(f"Target Month: {target_month}")
    print(f"{'='*80}")

    # Split data (walk-forward)
    target_start = pd.Timestamp(year=int(target_month[:4]), month=int(target_month[5:7]), day=1)
    if target_month == "2026-02":
        target_end = pd.Timestamp(year=2026, month=3, day=1)
    else:
        y, m = map(int, target_month.split("-"))
        if m == 12:
            target_end = pd.Timestamp(year=y+1, month=1, day=1)
        else:
            target_end = pd.Timestamp(year=y, month=m+1, day=1)

    # Train = all data before target month
    # Val = last 30 days before target month
    # Test = target month
    train_mask = df["business_day"] < target_start
    test_mask = (df["business_day"] >= target_start) & (df["business_day"] < target_end)
    
    train_df = df[train_mask].copy()
    
    # Val = last 30 days of train
    train_dates = sorted(train_df["business_day"].unique())
    if len(train_dates) >= 30:
        val_start = train_dates[-30]
        val_df = train_df[train_df["business_day"] >= val_start].copy()
        train_df = train_df[train_df["business_day"] < val_start].copy()
    else:
        val_df = train_df.copy()

    test_df = df[test_mask].copy()

    print(f"  Train: {len(train_df)} rows, {train_df['business_day'].nunique()} days")
    print(f"  Val: {len(val_df)} rows, {val_df['business_day'].nunique()} days")
    print(f"  Test: {len(test_df)} rows, {test_df['business_day'].nunique()} days")

    # Merge train + test for feature building (test needs history from train)
    merged_df = pd.concat([train_df, test_df], ignore_index=True)
    merged_df = merged_df.sort_values(["business_day", "hour_business"]).reset_index(drop=True)

    # Build features
    merged_df, feature_manifest = build_deep_rt_sota_features(
        merged_df,
        risk_features=args.risk_features,
        forecast_features=False,
    )

    # Get feature columns
    feature_columns = get_feature_columns(
        risk_features=args.risk_features,
        forecast_features=False,
    )
    feature_columns = [c for c in feature_columns if c in merged_df.columns]
    print(f"  Feature columns: {len(feature_columns)}")

    # Split back
    train_df = merged_df[merged_df["business_day"] < target_start].copy()
    test_df = merged_df[
        (merged_df["business_day"] >= target_start) & (merged_df["business_day"] < target_end)
    ].copy()

    # Build datasets
    train_days = sorted(train_df["business_day"].unique())
    test_days = sorted(test_df["business_day"].unique())

    config = DeepRTSOTADatasetConfig(
        seq_len_days=args.seq_len_days,
        target_mode=args.target_mode,
        risk_features=args.risk_features,
        forecast_features=False,
        target_granularity=args.target_granularity,
    )

    train_dataset = build_deep_rt_sota_dataset(
        config=config,
        full_df=train_df,
        target_days=train_days,
        feature_columns=feature_columns,
    )

    test_dataset = build_deep_rt_sota_dataset(
        config=config,
        full_df=merged_df,
        target_days=test_days,
        feature_columns=feature_columns,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    # Convert to tensors
    train_X, train_y = [], []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        train_X.append(torch.tensor(sample["X_seq"], dtype=torch.float32))
        train_y.append(torch.tensor(sample["y"], dtype=torch.float32))
    
    test_X, test_y = [], []
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        test_X.append(torch.tensor(sample["X_seq"], dtype=torch.float32))
        test_y.append(torch.tensor(sample["y"], dtype=torch.float32))

    train_X = torch.stack(train_X)
    train_y = torch.stack(train_y)
    test_X = torch.stack(test_X)
    test_y = torch.stack(test_y)

    # Create DataLoaders
    train_loader = DataLoader(
        TensorDataset(train_X, train_y),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_X, test_y),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dim = 24 if args.target_granularity == "day" else 1

    model_config = DeepRTSOTAModelConfig(
        model_profile=args.model_profile,
        seq_len_days=args.seq_len_days,
        target_mode=args.target_mode,
        n_features=len(feature_columns),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.1,
        output_dim=output_dim,
    )
    model = DeepRTSOTAModel(model_config).to(device)
    print(f"\n  Model created: {args.model_profile}, params: {sum(p.numel() for p in model.parameters())}")

    # Train
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()

    print(f"\n  Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred, _ = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{args.epochs}: loss={train_loss:.4f}")

    # Evaluate
    print(f"\n  Evaluating...")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred, _ = model(x_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # For residual_to_da mode, compute final rt_pred = da_anchor + residual_pred
    # Get da_anchor for test days
    test_df_original = test_df.copy()
    da_anchor_list = []
    rt_actual_list = []
    for day in test_days:
        day_mask = test_df_original["business_day"] == day
        day_rows = test_df_original[day_mask].sort_values("hour_business")
        if len(day_rows) >= 20:
            da_anchor_list.append(day_rows["da_anchor"].values)
            rt_actual_list.append(day_rows["rt_actual"].values)

    if args.target_granularity == "day":
        da_anchor_flat = np.concatenate([x for x in da_anchor_list])
        rt_actual_flat = np.concatenate([x for x in rt_actual_list])
        residual_pred_flat = all_preds.flatten()
        residual_true_flat = rt_actual_flat - da_anchor_flat
        final_rt_pred_flat = da_anchor_flat + residual_pred_flat
    else:
        da_anchor_flat = np.concatenate([x for x in da_anchor_list])
        rt_actual_flat = np.concatenate([x for x in rt_actual_list])
        residual_pred_flat = all_preds.flatten()
        residual_true_flat = rt_actual_flat - da_anchor_flat
        final_rt_pred_flat = da_anchor_flat + residual_pred_flat

    # Compute metrics
    mae = mean_absolute_error(rt_actual_flat, final_rt_pred_flat)
    rmse = np.sqrt(mean_squared_error(rt_actual_flat, final_rt_pred_flat))
    smape = smape_floor50(rt_actual_flat, final_rt_pred_flat)

    # Diagnostics
    diagnostics = {
        "da_anchor_std": float(np.std(da_anchor_flat)),
        "rt_actual_std": float(np.std(rt_actual_flat)),
        "residual_true_std": float(np.std(residual_true_flat)),
        "residual_pred_std": float(np.std(residual_pred_flat)),
        "final_rt_pred_std": float(np.std(final_rt_pred_flat)),
        "corr_da_rt": float(np.corrcoef(da_anchor_flat, rt_actual_flat)[0, 1]),
        "corr_residual_true_pred": float(np.corrcoef(residual_true_flat, residual_pred_flat)[0, 1]) if len(residual_true_flat) > 1 else -999.0,
        "corr_final_pred_true": float(np.corrcoef(final_rt_pred_flat, rt_actual_flat)[0, 1]),
    }

    # Collapse warnings
    if diagnostics["residual_pred_std"] / diagnostics["residual_true_std"] < 0.10:
        diagnostics["RESIDUAL_COLLAPSE_WARNING"] = True
    else:
        diagnostics["RESIDUAL_COLLAPSE_WARNING"] = False

    if diagnostics["final_rt_pred_std"] / diagnostics["rt_actual_std"] < 0.50:
        diagnostics["FINAL_PRED_COLLAPSE_WARNING"] = True
    else:
        diagnostics["FINAL_PRED_COLLAPSE_WARNING"] = False

    # Determine if beats baselines
    beats_naive = smape < 63.42
    beats_da = smape < 26.69

    print(f"\n  Results for {target_month}:")
    print(f"    sMAPE_floor50: {smape:.4f}")
    print(f"    MAE: {mae:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    Beats naive baseline (63.42): {beats_naive}")
    print(f"    Beats DA anchor (26.69): {beats_da}")
    print(f"    RESIDUAL_COLLAPSE_WARNING: {diagnostics['RESIDUAL_COLLAPSE_WARNING']}")
    print(f"    FINAL_PRED_COLLAPSE_WARNING: {diagnostics['FINAL_PRED_COLLAPSE_WARNING']}")

    # Save results
    out_path = Path(out_dir) / target_month
    out_path.mkdir(parents=True, exist_ok=True)

    results = {
        "target_month": target_month,
        "model_profile": args.model_profile,
        "target_granularity": args.target_granularity,
        "target_mode": args.target_mode,
        "seq_len_days": args.seq_len_days,
        "sMAPE_floor50": float(smape),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "beats_naive_baseline": beats_naive,
        "beats_da_anchor": beats_da,
        "test_samples": len(test_dataset),
        "test_days": len(test_days),
        "diagnostics": diagnostics,
    }

    with open(out_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(out_path / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    # Save predictions
    pred_df = pd.DataFrame({
        "business_day": [str(d) for d in test_days],
        "da_anchor_mean": [np.mean(x) for x in da_anchor_list],
        "rt_actual_mean": [np.mean(x) for x in rt_actual_list],
        "residual_pred_mean": all_preds.mean(axis=1) if all_preds.ndim > 1 else all_preds,
        "final_rt_pred_mean": (np.array([np.mean(x) for x in da_anchor_list]) + 
                           (all_preds.mean(axis=1) if all_preds.ndim > 1 else all_preds)),
    })
    pred_df.to_csv(out_path / "predictions.csv", index=False)

    print(f"\n  Results saved to {out_path}")

    return results


def main():
    args = parse_args()
    target_months = args.target_months.split(",")

    print(f"Starting multi-month backtest for: {target_months}")
    print(f"Model: {args.model_profile}, {args.target_granularity}, {args.target_mode}")

    # Load data once
    print("\nLoading data...")
    df = load_data(args.data_path)

    # Run backtest for each month
    all_results = []
    for month in target_months:
        results = train_and_evaluate_month(args, df, month, args.out_dir)
        all_results.append(results)

    # Aggregate results
    print(f"\n{'='*80}")
    print("Backtest Summary")
    print(f"{'='*80}")

    smape_list = [r["sMAPE_floor50"] for r in all_results]
    beats_da_list = [r["beats_da_anchor"] for r in all_results]

    print(f"\nMonthly sMAPE_floor50:")
    for r in all_results:
        print(f"  {r['target_month']}: {r['sMAPE_floor50']:.4f} (beats DA: {r['beats_da_anchor']})")

    print(f"\nAggregate:")
    print(f"  Mean sMAPE_floor50: {np.mean(smape_list):.4f}")
    print(f"  Std sMAPE_floor50: {np.std(smape_list):.4f}")
    print(f"  Min sMAPE_floor50: {np.min(smape_list):.4f}")
    print(f"  Max sMAPE_floor50: {np.max(smape_list):.4f}")
    print(f"  Beats DA anchor count: {sum(beats_da_list)}/{len(beats_da_list)}")

    # Save aggregate results
    out_path = Path(args.out_dir)
    aggregate = {
        "target_months": target_months,
        "model_profile": args.model_profile,
        "target_granularity": args.target_granularity,
        "target_mode": args.target_mode,
        "mean_sMAPE_floor50": float(np.mean(smape_list)),
        "std_sMAPE_floor50": float(np.std(smape_list)),
        "min_sMAPE_floor50": float(np.min(smape_list)),
        "max_sMAPE_floor50": float(np.max(smape_list)),
        "beats_da_count": sum(beats_da_list),
        "total_months": len(beats_da_list),
        "monthly_results": all_results,
    }

    with open(out_path / "aggregate_results.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    # Create leaderboard
    leaderboard = []
    for r in all_results:
        leaderboard.append({
            "target_month": r["target_month"],
            "sMAPE_floor50": r["sMAPE_floor50"],
            "MAE": r["MAE"],
            "RMSE": r["RMSE"],
            "beats_naive": r["beats_naive_baseline"],
            "beats_da": r["beats_da_anchor"],
            "test_days": r["test_days"],
        })

    leaderboard_df = pd.DataFrame(leaderboard)
    leaderboard_df.to_csv(out_path / "leaderboard.csv", index=False)

    print(f"\nAggregate results saved to {out_path}")

    # Determine verdict
    mean_smape = np.mean(smape_list)
    if mean_smape < 15.0:
        verdict = "TARGET_SOTA (mean sMAPE < 15)"
    elif mean_smape < 17.0:
        verdict = "STRONG_SOTA (mean sMAPE < 17)"
    elif mean_smape < 26.69 and sum(beats_da_list) == len(beats_da_list):
        verdict = "SOTA_CANDIDATE (beats DA in all months)"
    elif sum(beats_da_list) > 0:
        verdict = "UNSTABLE_SIGNAL (beats DA in some months)"
    else:
        verdict = "NO_GO (does not beat DA)"

    print(f"\nVerdict: {verdict}")

    with open(out_path / "verdict.txt", "w") as f:
        f.write(verdict)

    return aggregate


if __name__ == "__main__":
    main()
