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
    # Shrink/gate parameters
    parser.add_argument("--alpha-candidates", type=str, default="0,0.05,0.1,0.2,0.3,0.5,0.7,1.0",
                        help="Comma-separated alpha candidates")
    parser.add_argument("--clip-candidates", type=str, default="50,100,150,200,300",
                        help="Comma-separated clip candidates")
    parser.add_argument("--use-residual-history-features", action="store_true",
                        help="Use residual history features (residual_lag_*, etc.)")
    parser.add_argument("--checkpoint-metric", type=str, default="val_final_smape",
                        choices=["val_loss", "val_residual_smape", "val_final_smape"],
                        help="Metric to select best checkpoint")
    parser.add_argument("--loss", type=str, default="huber",
                        choices=["huber", "mse", "hybrid"],
                        help="Loss function")
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


def apply_shrink_gate(residual_pred, alpha, clip):
    """Apply shrink/gate: alpha * clip(residual_pred, -clip, clip)."""
    return alpha * np.clip(residual_pred, -clip, clip)


def evaluate_with_shrink_gate(residual_pred, da_anchor, rt_actual, alpha, clip):
    """Evaluate with shrink/gate, return sMAPE."""
    final_pred = da_anchor + apply_shrink_gate(residual_pred, alpha, clip)
    return smape_floor50(rt_actual, final_pred)


def select_best_alpha_clip(residual_pred, da_anchor, rt_actual, alpha_candidates, clip_candidates):
    """Select best (alpha, clip) on validation set."""
    best_smape = float("inf")
    best_alpha = 0.0
    best_clip = 100.0
    
    for alpha in alpha_candidates:
        for clip in clip_candidates:
            smape_val = evaluate_with_shrink_gate(
                residual_pred, da_anchor, rt_actual, alpha, clip
            )
            if smape_val < best_smape:
                best_smape = smape_val
                best_alpha = alpha
                best_clip = clip
    
    return best_alpha, best_clip, best_smape


def train_and_evaluate(args):
    """Main training and evaluation function with shrink/gate."""
    print("=" * 80)
    print(f"Experiment: {args.model_profile}, {args.target_granularity}, {args.target_mode}")
    print("=" * 80)
    
    # Parse alpha and clip candidates
    alpha_candidates = [float(x) for x in args.alpha_candidates.split(",")]
    clip_candidates = [float(x) for x in args.clip_candidates.split(",")]
    print(f"Alpha candidates: {alpha_candidates}")
    print(f"Clip candidates: {clip_candidates}")
    
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
        use_residual_history_features=args.use_residual_history_features,
    )
    
    # Get feature columns (after feature building)
    feature_columns = get_feature_columns(
        risk_features=args.risk_features,
        forecast_features=False,
        use_residual_history_features=args.use_residual_history_features,
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
    
    # Also get val_df from merged_df
    train_dates = sorted(train_df["business_day"].unique())
    if len(train_dates) >= 30:
        val_start = train_dates[-30]
        val_df = train_df[train_df["business_day"] >= val_start].copy()
        train_df = train_df[train_df["business_day"] < val_start].copy()
    else:
        val_df = train_df.copy()
    
    # Build datasets
    print("\nBuilding datasets...")
    train_days = sorted(train_df["business_day"].unique())
    test_days = sorted(test_df["business_day"].unique())
    val_days = sorted(val_df["business_day"].unique())
    
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
    
    val_dataset = build_deep_rt_sota_dataset(
        config=config,
        full_df=merged_df,  # Use merged_df so history is available
        target_days=val_days,
        feature_columns=feature_columns,
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    if len(test_dataset) < 27:
        print(f"  WARNING: Test samples ({len(test_dataset)}) < 27!")
    
    # Convert to tensors
    print("\nConverting to tensors...")
    train_X, train_y = dataset_to_tensors(train_dataset)
    val_X, val_y = dataset_to_tensors(val_dataset)
    test_X, test_y = dataset_to_tensors(test_dataset)
    
    # Create DataLoaders
    train_loader = DataLoader(
        TensorDataset(train_X, train_y),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_X, val_y),
        batch_size=args.batch_size,
        shuffle=False,
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
    
    # Train with checkpoint saving
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    if args.loss == "huber":
        criterion = nn.SmoothL1Loss()
    elif args.loss == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.SmoothL1Loss()  # default huber
    
    best_val_smape = float("inf")
    best_epoch = 0
    checkpoint_path = Path(args.out_dir) / "best_checkpoint.pt"
    
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
        
        # Evaluate on validation set every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            val_preds = []
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    pred, _ = model(x_batch)
                    val_preds.append(pred.cpu().numpy())
            val_preds = np.concatenate(val_preds, axis=0)
            
            # Compute validation sMAPE (final price)
            val_residual_pred_flat = val_preds.flatten() if val_preds.ndim > 1 else val_preds
            val_final_pred_flat = val_da_anchor_flat + apply_shrink_gate(
                val_residual_pred_flat, 1.0, 300.0
            )
            val_smape = smape_floor50(val_rt_actual_flat, val_final_pred_flat)
            
            if val_smape < best_val_smape:
                best_val_smape = val_smape
                best_epoch = epoch + 1
                torch.save(model.state_dict(), checkpoint_path)
            
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={train_loss:.4f}, val_sMAPE={val_smape:.4f} (best={best_val_smape:.4f} @ epoch {best_epoch})")
        elif (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={train_loss:.4f}")
    
    # Load best checkpoint
    if checkpoint_path.exists():
        print(f"\nLoading best checkpoint from epoch {best_epoch}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"\nNo best checkpoint found, using last epoch...")
    
    # Evaluate on validation set to select alpha and clip
    print("\nEvaluating on validation set to select alpha and clip...")
    model.eval()
    val_preds = []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred, _ = model(x_batch)
            val_preds.append(pred.cpu().numpy())
    val_preds = np.concatenate(val_preds, axis=0)
    
    # Get validation da_anchor and rt_actual
    val_df_original = merged_df[merged_df["business_day"].isin(val_days)].copy()
    val_days_sorted = sorted(val_df_original["business_day"].unique())
    
    val_da_anchor_list = []
    val_rt_actual_list = []
    for day in val_days_sorted:
        day_mask = val_df_original["business_day"] == day
        day_rows = val_df_original[day_mask].sort_values("hour_business")
        if len(day_rows) >= 20:
            val_da_anchor_list.append(day_rows["da_anchor"].values)
            val_rt_actual_list.append(day_rows["rt_actual"].values)
    
    val_da_anchor_flat = np.concatenate([x for x in val_da_anchor_list])
    val_rt_actual_flat = np.concatenate([x for x in val_rt_actual_list])
    val_residual_pred_flat = val_preds.flatten() if val_preds.ndim > 1 else val_preds
    
    # Select best alpha and clip on validation set
    best_alpha, best_clip, best_val_smape = select_best_alpha_clip(
        val_residual_pred_flat, val_da_anchor_flat, val_rt_actual_flat,
        alpha_candidates, clip_candidates
    )
    print(f"  Best alpha: {best_alpha}")
    print(f"  Best clip: {best_clip}")
    print(f"  Best val sMAPE: {best_val_smape:.4f}")
    
    # Evaluate on test set with selected alpha and clip
    print("\nEvaluating on test set with selected alpha and clip...")
    test_preds = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred, _ = model(x_batch)
            test_preds.append(pred.cpu().numpy())
    test_preds = np.concatenate(test_preds, axis=0)
    
    # Get test da_anchor and rt_actual
    test_df_original = merged_df[
        (merged_df["business_day"] >= target_start) &
        (merged_df["business_day"] < target_end)
    ].copy()
    test_days_sorted = sorted(test_df_original["business_day"].unique())
    
    test_da_anchor_list = []
    test_rt_actual_list = []
    for day in test_days_sorted:
        day_mask = test_df_original["business_day"] == day
        day_rows = test_df_original[day_mask].sort_values("hour_business")
        if len(day_rows) >= 20:
            test_da_anchor_list.append(day_rows["da_anchor"].values)
            test_rt_actual_list.append(day_rows["rt_actual"].values)
    
    test_da_anchor_flat = np.concatenate([x for x in test_da_anchor_list])
    test_rt_actual_flat = np.concatenate([x for x in test_rt_actual_list])
    test_residual_pred_flat = test_preds.flatten() if test_preds.ndim > 1 else test_preds
    
    # Apply shrink/gate
    test_final_rt_pred_flat = test_da_anchor_flat + apply_shrink_gate(
        test_residual_pred_flat, best_alpha, best_clip
    )
    
    # Compute metrics
    test_mae = mean_absolute_error(test_rt_actual_flat, test_final_rt_pred_flat)
    test_rmse = np.sqrt(mean_squared_error(test_rt_actual_flat, test_final_rt_pred_flat))
    test_smape = smape_floor50(test_rt_actual_flat, test_final_rt_pred_flat)
    
    # Also compute DA anchor sMAPE for comparison
    da_pred_flat = test_da_anchor_flat
    da_smape = smape_floor50(test_rt_actual_flat, da_pred_flat)
    
    # Diagnostics
    test_residual_true_flat = test_rt_actual_flat - test_da_anchor_flat
    diagnostics = {
        "da_anchor_std": float(np.std(test_da_anchor_flat)),
        "rt_actual_std": float(np.std(test_rt_actual_flat)),
        "residual_true_std": float(np.std(test_residual_true_flat)),
        "residual_pred_std": float(np.std(test_residual_pred_flat)),
        "final_rt_pred_std": float(np.std(test_final_rt_pred_flat)),
        "corr_da_rt": float(np.corrcoef(test_da_anchor_flat, test_rt_actual_flat)[0, 1]),
        "corr_residual_true_pred": float(np.corrcoef(test_residual_true_flat, test_residual_pred_flat)[0, 1]) if len(test_residual_true_flat) > 1 else -999.0,
        "corr_final_pred_true": float(np.corrcoef(test_final_rt_pred_flat, test_rt_actual_flat)[0, 1]),
        "selected_alpha": best_alpha,
        "selected_clip": best_clip,
        "val_smape": float(best_val_smape),
        "da_smape": float(da_smape),
        "test_smape": float(test_smape),
        "improvement_vs_da": float(da_smape - test_smape),
    }
    
    print(f"\nResults:")
    print(f"  DA anchor sMAPE: {da_smape:.4f}")
    print(f"  Test sMAPE (with shrink/gate): {test_smape:.4f}")
    print(f"  Improvement vs DA: {da_smape - test_smape:.4f} pp")
    print(f"  Selected alpha: {best_alpha}")
    print(f"  Selected clip: {best_clip}")
    
    # Determine if beats DA
    beats_da = test_smape < da_smape
    if not beats_da and best_alpha > 0:
        print(f"\n  WARNING: Model doesn't beat DA anchor. Consider alpha=0.")
    
    # Save results
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model_profile": args.model_profile,
        "target_granularity": args.target_granularity,
        "target_mode": args.target_mode,
        "seq_len_days": args.seq_len_days,
        "risk_features": args.risk_features,
        "selected_alpha": best_alpha,
        "selected_clip": best_clip,
        "val_smape": float(best_val_smape),
        "da_smape": float(da_smape),
        "sMAPE_floor50": float(test_smape),
        "MAE": float(test_mae),
        "RMSE": float(test_rmse),
        "improvement_vs_da": float(da_smape - test_smape),
        "beats_da_anchor": beats_da,
        "test_samples": len(test_dataset),
        "test_days": len(test_days),
    }
    
    with open(out_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save diagnostics
    with open(out_path / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)
    
    # Save predictions
    pred_days = [str(s["business_day"]) for s in test_dataset.samples]
    pred_df = pd.DataFrame({
        "business_day": pred_days,
        "da_anchor_mean": [test_da_anchor_flat[i*24:(i+1)*24].mean() for i in range(len(pred_days))],
        "rt_actual_mean": [test_rt_actual_flat[i*24:(i+1)*24].mean() for i in range(len(pred_days))],
        "final_rt_pred_mean": [test_final_rt_pred_flat[i*24:(i+1)*24].mean() for i in range(len(pred_days))],
        "residual_true_mean": [test_residual_true_flat[i*24:(i+1)*24].mean() for i in range(len(pred_days))],
        "residual_pred_mean": [test_residual_pred_flat[i*24:(i+1)*24].mean() for i in range(len(pred_days))],
    })
    pred_df.to_csv(out_path / "predictions.csv", index=False)
    
    print(f"\nResults saved to {out_path}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    args = parse_args()
    results = train_and_evaluate(args)
