"""Minimal training script for DeepRT-SOTA v2.

Simplest possible training loop for verification.
"""

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


def main():
    # ── Config ──────────────────────────────────────────────────────
    data_path = "tests/dummy_data_multi.csv"
    target_month = "2026-02"
    model_profile = "deep_rt_tcn"
    target_granularity = "day"
    target_mode = "direct"
    seq_len_days = 7
    risk_features = "off"
    epochs = 2
    batch_size = 32
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────
    df = pd.read_csv(data_path)
    df["ds"] = pd.to_datetime(df["ds"])
    df = add_business_time_columns(df, timestamp_col="ds")
    print(f"Loaded {len(df)} rows")

    # ── Split train/test ──────────────────────────────────────────
    target_start = pd.Timestamp("2026-02-01")
    target_end = pd.Timestamp("2026-03-01")

    train_mask = df["business_day"] < target_start
    test_mask = (df["business_day"] >= target_start) & (df["business_day"] < target_end)

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    print(f"Train: {len(train_df)} rows, {train_df['business_day'].nunique()} days")
    print(f"Test: {len(test_df)} rows, {test_df['business_day'].nunique()} days")

    # ── Merge + build features ───────────────────────────────────
    merged_df = pd.concat([train_df, test_df], ignore_index=True)
    merged_df = merged_df.sort_values(["business_day", "hour_business"]).reset_index(drop=True)

    merged_df, feature_manifest = build_deep_rt_sota_features(
        merged_df,
        risk_features=risk_features,
        forecast_features=False,
    )
    print(f"Features built. Manifest: {feature_manifest.get('risk_features_source', 'N/A')}")

    # Split back
    train_df = merged_df[merged_df["business_day"] < target_start].copy()
    test_df = merged_df[
        (merged_df["business_day"] >= target_start) & (merged_df["business_day"] < target_end)
    ].copy()

    # ── Get feature columns ──────────────────────────────────────
    feature_columns = get_feature_columns(
        risk_features=risk_features,
        forecast_features=False,
    )
    feature_columns = [c for c in feature_columns if c in merged_df.columns]
    print(f"Feature columns ({len(feature_columns)}): {feature_columns[:5]}")

    # ── Create datasets ─────────────────────────────────────────
    train_days = sorted(train_df["business_day"].unique())
    test_days = sorted(test_df["business_day"].unique())

    dataset_config = DeepRTSOTADatasetConfig(
        seq_len_days=seq_len_days,
        target_mode=target_mode,
        risk_features=False,  # Not used in dataset currently
        forecast_features=False,
        target_granularity=target_granularity,
    )

    train_dataset = build_deep_rt_sota_dataset(
        config=dataset_config,
        full_df=train_df,
        target_days=train_days,
        feature_columns=feature_columns,
    )

    test_dataset = build_deep_rt_sota_dataset(
        config=dataset_config,
        full_df=merged_df,
        target_days=test_days,
        feature_columns=feature_columns,
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")

    if len(test_dataset) < 27:
        print(f"WARNING: Test samples ({len(test_dataset)}) < 27!")

    # ── Create DataLoaders ─────────────────────────────────────
    # Convert datasets to tensors for DataLoader
    train_tensors = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        train_tensors.append((
            torch.tensor(sample["X_seq"], dtype=torch.float32),
            torch.tensor(sample["y"], dtype=torch.float32),
        ))

    test_tensors = []
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        test_tensors.append((
            torch.tensor(sample["X_seq"], dtype=torch.float32),
            torch.tensor(sample["y"], dtype=torch.float32),
        ))

    # Create datasets from tensors
    train_X = torch.stack([t[0] for t in train_tensors])
    train_y = torch.stack([t[1] for t in train_tensors])
    test_X = torch.stack([t[0] for t in test_tensors])
    test_y = torch.stack([t[1] for t in test_tensors])

    train_loader = DataLoader(
        TensorDataset(train_X, train_y),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_X, test_y),
        batch_size=batch_size,
        shuffle=False,
    )

    # ── Create model ────────────────────────────────────────────
    output_dim = 24 if target_granularity == "day" else 1
    model_config = DeepRTSOTAModelConfig(
        model_profile=model_profile,
        seq_len_days=seq_len_days,
        target_mode=target_mode,
        n_features=len(feature_columns),
        hidden_dim=128,
        num_layers=3,
        dropout=0.1,
        output_dim=output_dim,
    )

    model = DeepRTSOTAModel(model_config).to(device)
    print(f"Model created: {model_profile}, params: {sum(p.numel() for p in model.parameters())}")

    # ── Train ──────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.HuberLoss() if hasattr(nn, "HuberLoss") else nn.L1Loss()

    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(epochs):
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

        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}")

    # ── Evaluate ──────────────────────────────────────────────
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

    mae = mean_absolute_error(all_targets.flatten(), all_preds.flatten())
    rmse = np.sqrt(mean_squared_error(all_targets.flatten(), all_preds.flatten()))
    smape = smape_floor50(all_targets.flatten(), all_preds.flatten())

    print(f"\nTest Results:")
    print(f"  sMAPE_floor50: {smape:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
