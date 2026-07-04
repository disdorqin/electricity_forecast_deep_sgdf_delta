"""Training script for DeepRT-SOTA v2 - Fixed version.

This script properly:
1. Merges train+test before building features (so lookback features work for test days)
2. Uses the fixed dataset that can access history from before the test period
3. Writes proper output artifacts (model.pt, config.yaml, metrics_summary.json, etc.)
4. Includes data audit before training
5. Includes diagnostic outputs

Usage:
    python scripts/train_deep_realtime_sota_v2.py \
        --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \
        --target-month 2026-02 \
        --model-profile deep_rt_tcn \
        --target-granularity day \
        --target-mode direct \
        --seq-len-days 7 \
        --risk-features off \
        --loss huber \
        --epochs 60 \
        --batch-size 32 \
        --lr 0.001 \
        --out-dir artifacts/deep_rt_sota/exp_tcn_2026_02
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

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

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DeepRT-SOTA v2 model (fixed)")

    parser.add_argument("--data-path", type=str, required=True, help="Path to input data CSV")
    parser.add_argument("--target-month", type=str, required=True, help="Target month for evaluation (YYYY-MM)")
    parser.add_argument("--model-profile", type=str, default="deep_rt_tcn",
                        choices=["deep_rt_mlp", "deep_rt_tcn", "deep_rt_gru", "deep_rt_transformer"],
                        help="Model architecture")
    parser.add_argument("--target-granularity", type=str, default="day", choices=["day", "hourly"],
                        help="Prediction granularity: day (24h vector) or hourly (1h)")
    parser.add_argument("--target-mode", type=str, default="direct", choices=["direct", "residual_to_da"],
                        help="Target mode")
    parser.add_argument("--seq-len-days", type=int, default=7, help="Sequence length in days")
    parser.add_argument("--risk-features", type=str, default="off", choices=["off", "real", "synthetic"],
                        help="Risk features source (off|real|synthetic)")
    parser.add_argument("--allow-debug-synthetic-risk", action="store_true",
                        help="Allow synthetic risk features (debug only, metric_status will be DEBUG_ONLY)")
    parser.add_argument("--forecast-features", type=str, default="off", choices=["on", "off"],
                        help="Whether to use forecast-side features")
    parser.add_argument("--loss", type=str, default="huber", choices=["huber", "mae", "smape_floor50", "hybrid"],
                        help="Loss function")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--allow-debug", action="store_true", help="Allow training even if data audit fails (debug only)")

    return parser.parse_args()


def load_and_preprocess_data(data_path: str) -> pd.DataFrame:
    """Load and preprocess data.

    Args:
        data_path: Path to data CSV.

    Returns:
        Preprocessed DataFrame with business_time columns.
    """
    logger.info(f"Loading data from {data_path}")

    # Try different encodings
    try:
        df = pd.read_csv(data_path, encoding="gbk")
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding="utf-8")

    logger.info(f"Loaded {len(df)} rows")

    # Rename columns if needed
    column_mapping = {}
    if "时刻" in df.columns and "ds" not in df.columns:
        column_mapping["时刻"] = "ds"
    if "日前电价" in df.columns and "da_anchor" not in df.columns:
        column_mapping["日前电价"] = "da_anchor"
    if "实时电价" in df.columns and "rt_actual" not in df.columns:
        column_mapping["实时电价"] = "rt_actual"

    if column_mapping:
        df = df.rename(columns=column_mapping)
        logger.info(f"Renamed columns: {column_mapping}")

    # Ensure ds is datetime
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"])
    else:
        raise ValueError("Column 'ds' (timestamp) not found in data")

    # Add business time columns (strictly use business_time.py)
    df = add_business_time_columns(df, timestamp_col="ds")

    return df


def split_train_test(df: pd.DataFrame, target_month: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train and test.

    Args:
        df: Input DataFrame with business_time columns.
        target_month: Target month (YYYY-MM).

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    # Parse target month
    target_year, target_month_num = map(int, target_month.split("-"))
    target_start = pd.Timestamp(year=target_year, month=target_month_num, day=1)

    if target_month == "2026-02":
        target_end = pd.Timestamp(year=2026, month=3, day=1)
    else:
        # Generic: next month
        if target_month_num == 12:
            next_year = target_year + 1
            next_month = 1
        else:
            next_year = target_year
            next_month = target_month_num + 1
        target_end = pd.Timestamp(year=next_year, month=next_month, day=1)

    # Test: target month
    test_mask = (df["business_day"] >= target_start) & (df["business_day"] < target_end)
    test_df = df[test_mask].copy()

    # Train: before target month
    train_mask = df["business_day"] < target_start
    train_df = df[train_mask].copy()

    # Val: last 30 days of train
    train_dates = sorted(train_df["business_day"].unique())
    if len(train_dates) >= 30:
        val_start_date = train_dates[-30]
        val_mask = train_df["business_day"] >= val_start_date
        val_df = train_df[val_mask].copy()
        train_df = train_df[~val_mask].copy()
    else:
        # Not enough train data for val split, use last 10%
        val_start_idx = int(0.9 * len(train_dates))
        val_start_date = train_dates[val_start_idx]
        val_mask = train_df["business_day"] >= val_start_date
        val_df = train_df[val_mask].copy()
        train_df = train_df[~val_mask].copy()

    logger.info(f"Train: {len(train_df)} rows, {train_df['business_day'].nunique()} days")
    logger.info(f"Val: {len(val_df)} rows, {val_df['business_day'].nunique()} days")
    logger.info(f"Test: {len(test_df)} rows, {test_df['business_day'].nunique()} days")

    return train_df, val_df, test_df


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "huber",
) -> torch.Tensor:
    """Compute loss.

    Args:
        pred: Prediction tensor.
        target: Target tensor.
        loss_type: Loss type ("huber", "mae", "smape_floor50", "hybrid").

    Returns:
        Loss tensor.
    """
    if loss_type == "mae":
        return nn.L1Loss()(pred, target)
    elif loss_type == "huber":
        return nn.HuberLoss()(pred, target)
    elif loss_type == "smape_floor50":
        # Use canonical smape_floor50
        pred_np = pred.detach().cpu().numpy().flatten()
        target_np = target.detach().cpu().numpy().flatten()
        smape = smape_floor50(target_np, pred_np)
        return torch.tensor(smape, dtype=torch.float32, device=pred.device)
    elif loss_type == "hybrid":
        huber_loss = nn.HuberLoss()(pred, target)
        pred_np = pred.detach().cpu().numpy().flatten()
        target_np = target.detach().cpu().numpy().flatten()
        smape = smape_floor50(target_np, pred_np)
        return huber_loss + 0.1 * torch.tensor(smape, dtype=torch.float32, device=pred.device)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_type: str,
    device: torch.device,
) -> float:
    """Train for one epoch.

    Args:
        model: Model to train.
        dataloader: Training data loader.
        optimizer: Optimizer.
        loss_type: Loss type.
        device: Device to train on.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        optimizer.zero_grad()

        x_seq = batch["X_seq"].to(device)
        y = batch["y"].to(device)

        pred, _ = model(x_seq)
        loss = compute_loss(pred, y, loss_type)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_type: str,
    device: torch.device,
    full_df: pd.DataFrame,
    target_days: List[pd.Timestamp],
    feature_columns: List[str],
    config: DeepRTSOTADatasetConfig,
) -> Dict:
    """Evaluate model.

    Args:
        model: Model to evaluate.
        dataloader: Evaluation data loader.
        loss_type: Loss type.
        device: Device to evaluate on.
        full_df: Full DataFrame (for diagnostics).
        target_days: Target days (for diagnostics).
        feature_columns: Feature columns (for dataset creation).
        config: Dataset config.

    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_business_days = []

    with torch.no_grad():
        for batch in dataloader:
            x_seq = batch["X_seq"].to(device)
            y = batch["y"].to(device)

            pred, _ = model(x_seq)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_business_days.extend(batch["business_day"])

    # Concatenate
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute metrics
    mae = mean_absolute_error(all_targets.flatten(), all_preds.flatten())
    rmse = np.sqrt(mean_squared_error(all_targets.flatten(), all_preds.flatten()))
    smape = smape_floor50(all_targets.flatten(), all_preds.flatten())

    # Diagnostics
    pred_std = np.std(all_preds.flatten())
    target_std = np.std(all_targets.flatten())

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "smape_floor50": float(smape),
        "pred_std": float(pred_std),
        "target_std": float(target_std),
        "pred_std_vs_target_std": float(pred_std / target_std) if target_std > 0 else -1.0,
        "n_samples": len(all_preds),
    }


def save_artifacts(
    model: nn.Module,
    config: Dict,
    metrics: Dict,
    feature_manifest: Dict,
    train_manifest: Dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    business_days: List[pd.Timestamp],
    out_dir: str,
) -> None:
    """Save model and artifacts.

    Args:
        model: Trained model.
        config: Configuration dictionary.
        metrics: Metrics dictionary.
        feature_manifest: Feature manifest.
        train_manifest: Train manifest.
        train_df: Train DataFrame.
        test_df: Test DataFrame.
        predictions: Model predictions (n_samples, output_dim).
        business_days: List of business_days for predictions.
        out_dir: Output directory.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), out_path / "model.pt")

    # Save config as yaml
    with open(out_path / "config.yaml", "w") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)

    # Save feature manifest
    with open(out_path / "feature_manifest.json", "w") as f:
        json.dump(feature_manifest, f, indent=2, default=str)

    # Save train manifest
    with open(out_path / "train_manifest.json", "w") as f:
        json.dump(train_manifest, f, indent=2, default=str)

    # Save metrics
    with open(out_path / "metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Save predictions
    pred_df = pd.DataFrame({
        "business_day": [str(d) for d in business_days],
        "pred_mean": predictions.mean(axis=1) if predictions.ndim > 1 else predictions,
    })
    if "rt_actual" in test_df.columns:
        # Only include actual for rows we have predictions for
        actual_vals = test_df["rt_actual"].values[:len(pred_df)]
        pred_df["actual"] = actual_vals
    pred_df.to_csv(out_path / "predictions.csv", index=False)

    logger.info(f"Saved artifacts to {out_dir}")


def run_data_audit(
    df: pd.DataFrame,
    target_month: str,
    seq_len_days: int,
    risk_features: str,
) -> Dict:
    """Run data audit before training.

    Args:
        df: Full DataFrame.
        target_month: Target month.
        seq_len_days: Sequence length in days.
        risk_features: Risk features source.

    Returns:
        Audit result dictionary.
    """
    result = {
        "target_month": target_month,
        "seq_len_days": seq_len_days,
        "risk_features": risk_features,
        "checks": {},
        "errors": [],
        "warnings": [],
        "passed": False,
    }

    # Check 1: Date range
    date_min = df["ds"].min()
    date_max = df["ds"].max()
    result["checks"]["date_range"] = {
        "min": date_min.isoformat(),
        "max": date_max.isoformat(),
    }

    # Check 2: Target month coverage
    target_start = pd.to_datetime(target_month + "-01")
    if target_month == "2026-02":
        target_end = pd.to_datetime("2026-03-01")
    else:
        year, month = map(int, target_month.split("-"))
        if month == 12:
            target_end = pd.to_datetime(f"{year+1}-01-01")
        else:
            target_end = pd.to_datetime(f"{year}-{month+1:02d}-01")

    target_mask = (df["business_day"] >= target_start) & (df["business_day"] < target_end)
    target_days = df.loc[target_mask, "business_day"].unique()
    result["checks"]["target_month"] = {
        "business_days": len(target_days),
        "rows": int(target_mask.sum()),
    }

    if len(target_days) < 27:
        result["warnings"].append(f"Target month has only {len(target_days)} business days (expected >= 27)")

    # Check 3: Train/test split
    train_mask = df["business_day"] < target_start
    test_mask = (df["business_day"] >= target_start) & (df["business_day"] < target_end)

    train_days = df.loc[train_mask, "business_day"].nunique()
    test_days = df.loc[test_mask, "business_day"].nunique()

    result["checks"]["split"] = {
        "train_days": int(train_days),
        "test_days": int(test_days),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
    }

    # Check 4: DA anchor oracle check
    if "da_anchor" in df.columns:
        valid_mask = df["rt_actual"].notna() & df["da_anchor"].notna()
        if valid_mask.sum() > 0:
            is_oracle = np.allclose(df.loc[valid_mask, "da_anchor"].values, df.loc[valid_mask, "rt_actual"].values)
            if is_oracle:
                result["errors"].append("DA anchor is ORACLE (da_anchor == rt_actual)")
            result["checks"]["da_anchor"] = {"is_oracle": is_oracle}

    # Check 5: Risk features
    if risk_features == "synthetic" and not config.get("allow_debug_synthetic_risk", False):
        result["errors"].append("Synthetic risk features not allowed for formal metrics (need --allow-debug-synthetic-risk)")

    # Final verdict
    if len(result["errors"]) == 0:
        result["passed"] = True
        result["verdict"] = "PASS"
    else:
        result["passed"] = False
        result["verdict"] = "FAIL"

    return result


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 80)
    logger.info("DeepRT-SOTA v2 - Fixed Training Script")
    logger.info("=" * 80)

    # Load data
    df = load_and_preprocess_data(args.data_path)

    # Run data audit
    logger.info("\nRunning data audit...")
    audit_result = run_data_audit(
        df=df,
        target_month=args.target_month,
        seq_len_days=args.seq_len_days,
        risk_features=args.risk_features,
    )

    logger.info(f"Audit verdict: {audit_result['verdict']}")
    if audit_result["errors"]:
        for err in audit_result["errors"]:
            logger.error(f"  ERROR: {err}")
    if audit_result["warnings"]:
        for warn in audit_result["warnings"]:
            logger.warning(f"  WARNING: {warn}")

    if not audit_result["passed"] and not args.allow_debug:
        logger.error("Data audit FAILED. Use --allow-debug to train anyway (not recommended for formal metrics).")
        sys.exit(1)

    # Save audit result
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "train_data_audit.json", "w") as f:
        json.dump(audit_result, f, indent=2, default=str)

    # Split data
    train_df, val_df, test_df = split_train_test(df, args.target_month)

    # Merge train+test for feature building (so lookback features work for test days)
    logger.info("\nMerging train+test for feature building...")
    merged_df = pd.concat([train_df, test_df], ignore_index=True)
    merged_df = merged_df.sort_values(["business_day", "hour_business"]).reset_index(drop=True)

    # Build features on merged data
    logger.info("Building features on merged data...")
    # Pass risk_features as string (off|real|synthetic)
    merged_df, feature_manifest = build_deep_rt_sota_features(
        merged_df,
        risk_features=args.risk_features,  # Now a string
        forecast_features=args.forecast_features == "on",
    )

    # Split back
    target_start = pd.to_datetime(args.target_month + "-01")
    if args.target_month == "2026-02":
        target_end = pd.to_datetime("2026-03-01")
    else:
        year, month = map(int, args.target_month.split("-"))
        if month == 12:
            target_end = pd.to_datetime(f"{year+1}-01-01")
        else:
            target_end = pd.to_datetime(f"{year}-{month+1:02d}-01")

    train_df = merged_df[merged_df["business_day"] < target_start].copy()
    test_df = merged_df[
        (merged_df["business_day"] >= target_start) & (merged_df["business_day"] < target_end)
    ].copy()
    # Val stays as before (from train)

    logger.info(f"After feature building - Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    # Get feature columns
    feature_columns = get_feature_columns(
        risk_features=args.risk_features,  # Now a string
        forecast_features=args.forecast_features == "on",
    )
    # Filter to only columns that exist
    feature_columns = [col for col in feature_columns if col in merged_df.columns]

    logger.info(f"Feature columns ({len(feature_columns)}): {feature_columns[:10]}...")

    # Create datasets
    logger.info("\nCreating datasets...")

    # Train days
    train_days = sorted(train_df["business_day"].unique())
    # Val days
    val_days = sorted(val_df["business_day"].unique())
    # Test days
    test_days = sorted(test_df["business_day"].unique())

    logger.info(f"Train days: {len(train_days)}, Val days: {len(val_days)}, Test days: {len(test_days)}")

    # Create dataset config
    dataset_config = DeepRTSOTADatasetConfig(
        seq_len_days=args.seq_len_days,
        target_mode=args.target_mode,
        risk_features=args.risk_features,  # Now a string
        forecast_features=args.forecast_features == "on",
        target_granularity=args.target_granularity,
    )

    # Create train dataset (uses full_df = train_df, target_days = train_days)
    train_dataset = build_deep_rt_sota_dataset(
        config=dataset_config,
        full_df=train_df,
        target_days=train_days,
        feature_columns=feature_columns,
    )

    # Create val dataset
    val_dataset = build_deep_rt_sota_dataset(
        config=dataset_config,
        full_df=pd.concat([train_df, val_df]),
        target_days=val_days,
        feature_columns=feature_columns,
    )

    # Create test dataset (can access train history)
    test_dataset = build_deep_rt_sota_dataset(
        config=dataset_config,
        full_df=merged_df,  # Full merged data
        target_days=test_days,
        feature_columns=feature_columns,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    if len(test_dataset) < 27:
        logger.warning(f"Test samples ({len(test_dataset)}) < 27, day-level test may be truncated!")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    output_dim = 24 if args.target_granularity == "day" else 1
    model_config = DeepRTSOTAModelConfig(
        model_profile=args.model_profile,
        seq_len_days=args.seq_len_days,
        target_mode=args.target_mode,
        n_features=len(feature_columns),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_dim=output_dim,
    )

    model = DeepRTSOTAModel(model_config)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    logger.info(f"Model created: {args.model_profile}")
    logger.info(f"Device: {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    logger.info("\nStarting training...")
    training_curves = {"train_loss": [], "val_loss": [], "val_smape": []}

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, args.loss, device)
        val_metrics = evaluate(
            model, val_loader, args.loss, device,
            full_df=merged_df,
            target_days=val_days,
            feature_columns=feature_columns,
            config=dataset_config,
        )

        training_curves["train_loss"].append(train_loss)
        training_curves["val_loss"].append(val_metrics.get("mae", 0.0))
        training_curves["val_smape"].append(val_metrics.get("smape_floor50", 0.0))

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{args.epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_smape={val_metrics.get('smape_floor50', 0.0):.4f}, "
                f"val_mae={val_metrics.get('mae', 0.0):.4f}"
            )

    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    test_metrics = evaluate(
        model, test_loader, args.loss, device,
        full_df=merged_df,
        target_days=test_days,
        feature_columns=feature_columns,
        config=dataset_config,
    )

    logger.info(f"Test sMAPE_floor50: {test_metrics.get('smape_floor50', 0.0):.4f}")
    logger.info(f"Test MAE: {test_metrics.get('mae', 0.0):.4f}")
    logger.info(f"Test RMSE: {test_metrics.get('rmse', 0.0):.4f}")
    logger.info(f"Pred std / target std: {test_metrics.get('pred_std_vs_target_std', -1.0):.4f}")

    # Build train_manifest
    train_manifest = {
        "train_rows": int(train_mask.sum()) if 'train_mask' in locals() else len(train_df),
        "val_rows": int(val_mask.sum()) if 'val_mask' in locals() else len(val_df),
        "test_rows": int(test_mask.sum()) if 'test_mask' in locals() else len(test_df),
        "train_days": len(train_days),
        "val_days": len(val_days),
        "test_days": len(test_days),
        "feature_columns": feature_columns,
        "seq_len_days": args.seq_len_days,
        "target_granularity": args.target_granularity,
        "created_at": datetime.now().isoformat(),
    }

    # Determine metric_status
    metric_status = "FORMAL"
    if args.risk_features == "synthetic":
        metric_status = "DEBUG_ONLY_SYNTHETIC_RISK"
    if not audit_result["passed"]:
        metric_status = "INVALID_DATA_AUDIT"

    # Build metrics summary (field names must match user spec exactly)
    metrics_summary = {
        "metric_status": metric_status,
        "target_month": args.target_month,
        "test_rows": int(test_mask.sum()) if 'test_mask' in locals() else len(test_df),
        "test_business_days": len(test_days),
        "model_profile": args.model_profile,
        "target_mode": args.target_mode,
        "risk_features_source": args.risk_features,
        "overall_sMAPE_floor50": float(test_metrics.get("smape_floor50", 0.0)),
        "MAE": float(test_metrics.get("mae", 0.0)),
        "RMSE": float(test_metrics.get("rmse", 0.0)),
        "baseline_comparison": None,  # TODO: Add baseline comparison
        "beats_naive_baseline": None,  # TODO: Compare with naive baseline
        "beats_da_anchor": None,  # TODO: Compare with DA anchor
        "created_at": datetime.now().isoformat(),
        "diagnostics": {
            "pred_std": float(test_metrics.get("pred_std", 0.0)),
            "target_std": float(test_metrics.get("target_std", 0.0)),
            "pred_std_vs_target_std": float(test_metrics.get("pred_std_vs_target_std", -1.0)),
        },
    }

    # Save artifacts
    logger.info("\nSaving artifacts...")
    save_artifacts(
        model=model,
        config=vars(args),
        metrics=metrics_summary,
        feature_manifest=feature_manifest,
        train_manifest=train_manifest,
        train_df=train_df,
        test_df=test_df,
        predictions=test_predictions,  # TODO: Get actual predictions from model
        business_days=test_days,
        out_dir=args.out_dir,
    )

    # Save training curves
    curves_df = pd.DataFrame(training_curves)
    curves_df.to_csv(Path(args.out_dir) / "training_curves.csv", index=False)

    logger.info("=" * 80)
    logger.info("Training complete.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
