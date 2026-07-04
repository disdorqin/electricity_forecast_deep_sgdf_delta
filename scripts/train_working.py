"""Working training script for DeepRT-SOTA v2.

Supports CLI arguments for flexible experimentation.
"""

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
    parser.add_argument("--target-month", type=str, default="2026-02")
    parser.add_argument("--model-profile", type=str, default="deep_rt_tcn",
                        choices=["deep_rt_mlp", "deep_rt_tcn", "deep_rt_gru", "deep_rt_transformer"])
    parser.add_argument("--target-granularity", type=str, default="day", choices=["day", "hourly"])
    parser.add_argument("--target-mode", type=str, default="direct", choices=["direct", "residual_to_da"])
    parser.add_argument("--seq-len-days", type=int, default=7)
    parser.add_argument("--risk-features", type=str, default="off", choices=["off", "real", "synthetic"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    """Load and preprocess data (handles both Chinese and English column names)."""
    # Try different encodings
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

    # Ensure ds is datetime
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"])
    else:
        raise ValueError("Column 'ds' (timestamp) not found")

    # Add business time columns
    df = add_business_time_columns(df, timestamp_col="ds")
    return df


def split_data(df: pd.DataFrame, target_month: str):
    """Split into train/test."""
    target_start = pd.Timestamp(year=int(target_month[:4]), month=int(target_month[5:7]), day=1)
    if target_month == "2026-02":
        target_end = pd.Timestamp(year=2026, month=3, day=1)
    else:
        y, m = map(int, target_month.split("-"))
        if m == 12:
            target_end = pd.Timestamp(year=y+1, month=1, day=1)
        else:
            target_end = pd.Timestamp(year=y, month=m+1, day=1)

    train_mask = df["business_day"] < target_start
    test_mask = (df["business_day"] >= target_start) & (df["business_day"] < target_end)
    val_mask = train_mask  # Simpler: use last 30 days of train as val

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    # Val = last 30 days of train
    train_dates = sorted(train_df["business_day"].unique())
    if len(train_dates) >= 30:
        val_start = train_dates[-30]
        val_df = train_df[train_df["business_day"] >= val_start].copy()
        train_df = train_df[train_df["business_day"] < val_start].copy()
    else:
        val_df = train_df.copy()  # Not enough data

    print(f"  Train: {len(train_df)} rows, {train_df['business_day'].nunique()} days")
    print(f"  Val: {len(val_df)} rows, {val_df['business_day'].nunique()} days")
    print(f"  Test: {len(test_df)} rows, {test_df['business_day'].nunique()} days")

    return train_df, val_df, test_df


def build_datasets(train_df, test_df, merged_df, feature_columns, args):
    """Build train/test datasets."""
    train_days = sorted(train_df["business_day"].unique())
    test_days = sorted(test_df["business_day"].unique())

    config = DeepRTSOTADatasetConfig(
        seq_len_days=args.seq_len_days,
        target_mode=args.target_mode,
        risk_features=args.risk_features,  # Fixed: use args.risk_features instead of False
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

    return train_dataset, test_dataset, train_days, test_days


def dataset_to_tensors(dataset):
    """Convert dataset to X, y tensors."""
    X_list = []
    y_list = []
    for i in range(len(dataset)):
        sample = dataset[i]
        X_list.append(torch.tensor(sample["X_seq"], dtype=torch.float32))
        y_list.append(torch.tensor(sample["y"], dtype=torch.float32))
    
    X = torch.stack(X_list)
    y = torch.stack(y_list)
    return X, y


def train_and_evaluate(args):
    """Main training and evaluation function."""
    print("=" * 80)
    print(f"Experiment: {args.model_profile}, {args.target_granularity}, {args.target_mode}")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df = load_data(args.data_path)

    # Split
    print("\nSplitting data...")
    train_df, val_df, test_df = split_data(df, args.target_month)

    # Merge + build features
    print("\nMerging and building features...")
    merged_df = pd.concat([train_df, test_df], ignore_index=True)
    merged_df = merged_df.sort_values(["business_day", "hour_business"]).reset_index(drop=True)

    merged_df, feature_manifest = build_deep_rt_sota_features(
        merged_df,
        risk_features=args.risk_features,
        forecast_features=False,
    )

    # Get feature columns (after feature building)
    feature_columns = get_feature_columns(
        risk_features=args.risk_features,
        forecast_features=False,
    )
    feature_columns = [c for c in feature_columns if c in merged_df.columns]
    print(f"  Feature columns: {len(feature_columns)}")

    # Split back
    target_start = pd.Timestamp(year=int(args.target_month[:4]), month=int(args.target_month[5:7]), day=1)
    if args.target_month == "2026-02":
        target_end = pd.Timestamp(year=2026, month=3, day=1)
    else:
        y, m = map(int, args.target_month.split("-"))
        if m == 12:
            target_end = pd.Timestamp(year=y+1, month=1, day=1)
        else:
            target_end = pd.Timestamp(year=y, month=m+1, day=1)

    train_df = merged_df[merged_df["business_day"] < target_start].copy()
    test_df = merged_df[
        (merged_df["business_day"] >= target_start) & (merged_df["business_day"] < target_end)
    ].copy()

    # Build datasets
    print("\nBuilding datasets...")
    train_dataset, test_dataset, train_days, test_days = build_datasets(
        train_df, test_df, merged_df, feature_columns, args
    )

    if len(test_dataset) < 27:
        print(f"  WARNING: Test samples ({len(test_dataset)}) < 27!")

    # Convert to tensors
    print("\nConverting to tensors...")
    train_X, train_y = dataset_to_tensors(train_dataset)
    test_X, test_y = dataset_to_tensors(test_dataset)

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
    print(f"\nModel created: {args.model_profile}, params: {sum(p.numel() for p in model.parameters())}")

    # Train
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()  # Fixed: HuberLoss -> SmoothL1Loss

    print(f"\nTraining for {args.epochs} epochs...")
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
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={train_loss:.4f}")

    # Evaluate
    print("\nEvaluating...")
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

    # For residual_to_da mode, need to compute final rt_pred = da_anchor + residual_pred
    # Load original test data to get da_anchor values
    test_df_original = merged_df[
        (merged_df["business_day"] >= target_start) & 
        (merged_df["business_day"] < target_end)
    ].copy()
    test_days_sorted = sorted(test_df_original["business_day"].unique())
    
    # Get da_anchor for test days (day-level: one value per day, hourly: 24 values per day)
    da_anchor_list = []
    rt_actual_list = []
    for day in test_days_sorted:
        day_mask = test_df_original["business_day"] == day
        day_rows = test_df_original[day_mask].sort_values("hour_business")
        if len(day_rows) >= 20:  # Valid day
            da_anchor_list.append(day_rows["da_anchor"].values)
            rt_actual_list.append(day_rows["rt_actual"].values)
    
    # Flatten if day-level (24h per day)
    if args.target_granularity == "day":
        da_anchor_flat = np.concatenate([x for x in da_anchor_list])
        rt_actual_flat = np.concatenate([x for x in rt_actual_list])
        # For day-level, all_preds is (n_days, 24), need to flatten
        residual_pred_flat = all_preds.flatten()
        residual_true_flat = rt_actual_flat - da_anchor_flat
        final_rt_pred_flat = da_anchor_flat + residual_pred_flat
    else:
        # Hourly mode
        da_anchor_flat = np.concatenate([x for x in da_anchor_list])
        rt_actual_flat = np.concatenate([x for x in rt_actual_list])
        residual_pred_flat = all_preds.flatten()
        residual_true_flat = rt_actual_flat - da_anchor_flat
        final_rt_pred_flat = da_anchor_flat + residual_pred_flat

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
        "pred_min": float(np.min(final_rt_pred_flat)),
        "pred_max": float(np.max(final_rt_pred_flat)),
        "target_min": float(np.min(rt_actual_flat)),
        "target_max": float(np.max(rt_actual_flat)),
        "residual_pred_min": float(np.min(residual_pred_flat)),
        "residual_pred_max": float(np.max(residual_pred_flat)),
        "residual_true_min": float(np.min(residual_true_flat)),
        "residual_true_max": float(np.max(residual_true_flat)),
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

    print(f"\nResults:")
    print(f"  sMAPE_floor50: {smape:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  final_rt_pred_std: {diagnostics['final_rt_pred_std']:.4f}")
    print(f"  rt_actual_std: {diagnostics['rt_actual_std']:.4f}")
    print(f"  final_rt_pred_std/rt_actual_std: {diagnostics['final_rt_pred_std']/diagnostics['rt_actual_std']:.4f}" if diagnostics["rt_actual_std"] > 0 else "  final_rt_pred_std/rt_actual_std: N/A")
    print(f"  RESIDUAL_COLLAPSE_WARNING: {diagnostics['RESIDUAL_COLLAPSE_WARNING']}")
    print(f"  FINAL_PRED_COLLAPSE_WARNING: {diagnostics['FINAL_PRED_COLLAPSE_WARNING']}")

    # Determine if beats baselines
    beats_naive = smape < 63.42  # naive baseline
    beats_da = smape < 26.69  # DA anchor

    print(f"\n  Beats naive baseline (63.42): {beats_naive}")
    print(f"  Beats DA anchor (26.69): {beats_da}")

    # Save results
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = {
        "model_profile": args.model_profile,
        "target_granularity": args.target_granularity,
        "target_mode": args.target_mode,
        "seq_len_days": args.seq_len_days,
        "risk_features": args.risk_features,
        "sMAPE_floor50": float(smape),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "beats_naive_baseline": beats_naive,
        "beats_da_anchor": beats_da,
        "test_samples": len(test_dataset),
        "test_days": len(test_days),
    }

    # Save diagnostics
    with open(out_path / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    # Check for collapse warnings
    if diagnostics.get("RESIDUAL_COLLAPSE_WARNING", False):
        print(f"\n  WARNING: RESIDUAL_COLLAPSE_WARNING detected!")
        print(f"    residual_pred_std / residual_true_std = {diagnostics['residual_pred_std'] / diagnostics['residual_true_std']:.4f}")
    if diagnostics.get("FINAL_PRED_COLLAPSE_WARNING", False):
        print(f"\n  WARNING: FINAL_PRED_COLLAPSE_WARNING detected!")
        print(f"    final_rt_pred_std / rt_actual_std = {diagnostics['final_rt_pred_std'] / diagnostics['rt_actual_std']:.4f}")

    with open(out_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions (only for days that have predictions)
    pred_days = [str(s["business_day"]) for s in test_dataset.samples]
    pred_df = pd.DataFrame({
        "business_day": pred_days,
        "pred_mean": all_preds.mean(axis=1) if all_preds.ndim > 1 else all_preds,
        "actual_mean": all_targets.mean(axis=1) if all_targets.ndim > 1 else all_targets,
    })
    pred_df.to_csv(out_path / "predictions.csv", index=False)

    print(f"\nResults saved to {out_path}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    args = parse_args()
    results = train_and_evaluate(args)
