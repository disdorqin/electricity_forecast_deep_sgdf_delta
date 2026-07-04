"""Training script for DeepRT-SOTA v2.

This script trains a standalone realtime price deep model.

Usage:
    python scripts/train_deep_realtime_sota.py \
        --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \
        --target-month 2026-02 \
        --model-profile deep_rt_tcn \
        --target-mode direct \
        --seq-len-days 14 \
        --risk-features off \
        --forecast-features off \
        --loss huber \
        --epochs 80 \
        --batch-size 64 \
        --lr 0.001 \
        --out-dir artifacts/deep_rt_sota/exp_tcn_2026_02
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.deep_sgdf_delta.business_time import add_business_time_columns
from models.deep_sgdf_delta.deep_rt_sota_dataset import (
    DeepRTSOTADatasetConfig,
    build_deep_rt_sota_dataset,
)
from models.deep_sgdf_delta.deep_rt_sota_features import build_deep_rt_sota_features
from models.deep_sgdf_delta.deep_rt_sota_model import (
    DeepRTSOTAModel,
    DeepRTSOTAModelConfig,
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DeepRT-SOTA v2 model")

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to input data CSV",
    )
    parser.add_argument(
        "--target-month",
        type=str,
        required=True,
        help="Target month for evaluation (YYYY-MM)",
    )
    parser.add_argument(
        "--model-profile",
        type=str,
        default="deep_rt_tcn",
        choices=["deep_rt_mlp", "deep_rt_tcn", "deep_rt_gru", "deep_rt_transformer"],
        help="Model architecture",
    )
    parser.add_argument(
        "--target-mode",
        type=str,
        default="direct",
        choices=["direct", "residual_to_da"],
        help="Target mode",
    )
    parser.add_argument(
        "--seq-len-days",
        type=int,
        default=14,
        help="Sequence length in days",
    )
    parser.add_argument(
        "--risk-features",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Whether to use risk features",
    )
    parser.add_argument(
        "--forecast-features",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Whether to use forecast-side features",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="huber",
        choices=["huber", "mae", "smape_floor50", "hybrid"],
        help="Loss function",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of layers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory",
    )

    return parser.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    """Load and preprocess data.

    Args:
        data_path: Path to data CSV.

    Returns:
        Preprocessed DataFrame.
    """
    logger.info(f"Loading data from {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows")

    # Add business time columns
    df = add_business_time_columns(df, timestamp_col="ds")

    return df


def split_data(
    df: pd.DataFrame, target_month: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test.

    Args:
        df: Input DataFrame.
        target_month: Target month (YYYY-MM).

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    # Parse target month
    target_year, target_month_num = map(int, target_month.split("-"))
    target_start = pd.Timestamp(year=target_year, month=target_month_num, day=1)

    # Test: target month
    test_mask = (
        (df["business_day"].dt.year == target_year)
        & (df["business_day"].dt.month == target_month_num)
    )
    test_df = df[test_mask].copy()

    # Train: before target month
    train_mask = df["business_day"] < target_start
    train_df = df[train_mask].copy()

    # Val: last 30 days of train
    train_dates = sorted(train_df["business_day"].unique())
    val_start_date = train_dates[-30]
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
        # Simplified sMAPE loss
        eps = 1e-8
        smape = 200 * torch.abs(pred - target) / (torch.abs(pred) + torch.abs(target) + eps)
        return smape.mean()
    elif loss_type == "hybrid":
        # Hybrid: Huber + sMAPE
        huber_loss = nn.HuberLoss()(pred, target)
        eps = 1e-8
        smape = 200 * torch.abs(pred - target) / (torch.abs(pred) + torch.abs(target) + eps)
        return huber_loss + 0.1 * smape.mean()
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

        # Move batch to device
        x_seq = batch["X_seq"].to(device)
        y = batch["y"].to(device)

        # Forward pass
        pred, _ = model(x_seq)
        loss = compute_loss(pred, y, loss_type)

        # Backward pass
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
) -> Dict:
    """Evaluate model.

    Args:
        model: Model to evaluate.
        dataloader: Evaluation data loader.
        loss_type: Loss type.
        device: Device to evaluate on.

    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            x_seq = batch["X_seq"].to(device)
            y = batch["y"].to(device)

            pred, _ = model(x_seq)
            loss = compute_loss(pred, y, loss_type)

            total_loss += loss.item()
            n_batches += 1

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute metrics
    mae = mean_absolute_error(all_targets.flatten(), all_preds.flatten())
    rmse = np.sqrt(mean_squared_error(all_targets.flatten(), all_preds.flatten()))

    return {
        "loss": total_loss / n_batches if n_batches > 0 else 0.0,
        "mae": mae,
        "rmse": rmse,
    }


def save_artifacts(
    model: nn.Module,
    config: Dict,
    metrics: Dict,
    out_dir: str,
) -> None:
    """Save model and artifacts.

    Args:
        model: Trained model.
        config: Configuration dictionary.
        metrics: Metrics dictionary.
        out_dir: Output directory.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), out_path / "model.pt")

    # Save config
    with open(out_path / "config.yaml", "w") as f:
        import yaml
        yaml.dump(config, f)

    # Save metrics
    with open(out_path / "metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved artifacts to {out_dir}")


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load data
    df = load_data(args.data_path)

    # Split data
    train_df, val_df, test_df = split_data(df, args.target_month)

    # Build features
    risk_features = args.risk_features == "on"
    forecast_features = args.forecast_features == "on"

    train_df, _ = build_deep_rt_sota_features(
        train_df,
        risk_features=risk_features,
        forecast_features=forecast_features,
    )
    val_df, _ = build_deep_rt_sota_features(
        val_df,
        risk_features=risk_features,
        forecast_features=forecast_features,
    )
    test_df, _ = build_deep_rt_sota_features(
        test_df,
        risk_features=risk_features,
        forecast_features=forecast_features,
    )

    # Create datasets
    dataset_config = DeepRTSOTADatasetConfig(
        seq_len_days=args.seq_len_days,
        target_mode=args.target_mode,
        risk_features=risk_features,
        forecast_features=forecast_features,
    )

    # Create datasets
    # TODO: Implement proper dataset creation
    # For now, create simple datasets

    logger.info("Creating datasets...")

    # Get feature columns
    from models.deep_sgdf_delta.deep_rt_sota_features import get_feature_columns
    feature_columns = get_feature_columns(
        risk_features=risk_features,
        forecast_features=forecast_features,
    )

    # Create model config
    model_config = DeepRTSOTAModelConfig(
        model_profile=args.model_profile,
        seq_len_days=args.seq_len_days,
        target_mode=args.target_mode,
        n_features=len(feature_columns),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_dim=24,
    )

    # Create model
    model = DeepRTSOTAModel(model_config)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    logger.info(f"Model created: {args.model_profile}")
    logger.info(f"Device: {device}")
    logger.info(f"Feature columns: {len(feature_columns)}")

    # TODO: Implement actual training loop
    logger.info("Training script ready. TODO: Implement full training loop.")

    # Save model config
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "model_profile": args.model_profile,
        "target_mode": args.target_mode,
        "seq_len_days": args.seq_len_days,
        "risk_features": args.risk_features,
        "forecast_features": args.forecast_features,
        "loss": args.loss,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "target_month": args.target_month,
        "data_path": args.data_path,
    }

    with open(out_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Config saved to {out_path / 'config.json'}")


if __name__ == "__main__":
    main()
