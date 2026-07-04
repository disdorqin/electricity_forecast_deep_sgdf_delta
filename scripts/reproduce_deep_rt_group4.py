"""Reproduce Group 4 experiment exactly.

Group 4 frozen config:
  model_profile: deep_rt_tcn
  target_granularity: day
  target_mode: residual_to_da
  seq_len_days: 7
  risk_features: off
  loss: hybrid
  epochs: 60
  batch_size: 32
  lr: 0.001
  target_month: 2026-02

Output:
  config.yaml
  train_data_audit.json
  feature_manifest.json
  metrics_summary.json
  predictions.csv
  diagnostics.json
  reproduce_report.md
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


GROUP4_CONFIG = {
    "model_profile": "deep_rt_tcn",
    "target_granularity": "day",
    "target_mode": "residual_to_da",
    "seq_len_days": 7,
    "risk_features": "off",
    "loss": "hybrid",
    "epochs": 60,
    "batch_size": 32,
    "lr": 0.001,
    "target_month": "2026-02",
    "hidden_dim": 128,
    "num_layers": 3,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(data_path: str) -> pd.DataFrame:
    for enc in ["gbk", "gb18030", "utf-8"]:
        try:
            df = pd.read_csv(data_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not read {data_path}")
    
    rename_map = {}
    if "时刻" in df.columns and "ds" not in df.columns:
        rename_map["时刻"] = "ds"
    if "日前电价" in df.columns and "da_anchor" not in df.columns:
        rename_map["日前电价"] = "da_anchor"
    if "实时电价" in df.columns and "rt_actual" not in df.columns:
        rename_map["实时电价"] = "rt_actual"
    if rename_map:
        df = df.rename(columns=rename_map)
    
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"])
    else:
        raise ValueError("Column 'ds' not found")
    
    df = add_business_time_columns(df, timestamp_col="ds")
    return df


def split_data(df: pd.DataFrame, target_month: str):
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
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    train_dates = sorted(train_df["business_day"].unique())
    if len(train_dates) >= 30:
        val_start = train_dates[-30]
        val_df = train_df[train_df["business_day"] >= val_start].copy()
        train_df = train_df[train_df["business_day"] < val_start].copy()
    else:
        val_df = train_df.copy()
    
    print(f"  Train: {len(train_df)} rows, {train_df['business_day'].nunique()} days")
    print(f"  Val: {len(val_df)} rows, {val_df['business_day'].nunique()} days")
    print(f"  Test: {len(test_df)} rows, {test_df['business_day'].nunique()} days")
    
    return train_df, val_df, test_df


def build_datasets(train_df, test_df, merged_df, feature_columns, config):
    train_days = sorted(train_df["business_day"].unique())
    test_days = sorted(test_df["business_day"].unique())
    
    dataset_config = DeepRTSOTADatasetConfig(
        seq_len_days=config["seq_len_days"],
        target_mode=config["target_mode"],
        risk_features=config["risk_features"],
        forecast_features=False,
        target_granularity=config["target_granularity"],
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
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    return train_dataset, test_dataset


def dataset_to_tensors(dataset):
    X_list = []
    y_list = []
    for i in range(len(dataset)):
        sample = dataset[i]
        X_list.append(torch.tensor(sample["X_seq"], dtype=torch.float32))
        y_list.append(torch.tensor(sample["y"], dtype=torch.float32))
    
    X = torch.stack(X_list)
    y = torch.stack(y_list)
    return X, y


def main():
    args = parse_args()
    set_seed(args.seed)
    
    config = GROUP4_CONFIG.copy()
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(out_path / "config.yaml", "w") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)
    
    print("=" * 80)
    print("Reproduce Group 4 Experiment")
    print("=" * 80)
    print(f"Config: {config}")
    
    # Load data
    print("\nLoading data...")
    df = load_data(args.data_path)
    
    # Data audit
    print("\nAuditing data...")
    train_df, val_df, test_df = split_data(df, config["target_month"])
    
    train_data_audit = {
        "train_rows": len(train_df),
        "train_days": train_df["business_day"].nunique(),
        "val_rows": len(val_df),
        "val_days": val_df["business_day"].nunique(),
        "test_rows": len(test_df),
        "test_days": test_df["business_day"].nunique(),
        "train_date_range": [str(train_df["business_day"].min()), str(train_df["business_day"].max())],
        "test_date_range": [str(test_df["business_day"].min()), str(test_df["business_day"].max())],
    }
    with open(out_path / "train_data_audit.json", "w") as f:
        json.dump(train_data_audit, f, indent=2)
    
    # Merge + build features
    print("\nMerging and building features...")
    merged_df = pd.concat([train_df, test_df], ignore_index=True)
    merged_df = merged_df.sort_values(["business_day", "hour_business"]).reset_index(drop=True)
    
    merged_df, feature_manifest = build_deep_rt_sota_features(
        merged_df,
        risk_features=config["risk_features"],
        forecast_features=False,
    )
    
    # Save feature manifest
    with open(out_path / "feature_manifest.json", "w") as f:
        json.dump(feature_manifest, f, indent=2)
    
    # Get feature columns
    feature_columns = get_feature_columns(
        risk_features=config["risk_features"],
        forecast_features=False,
    )
    feature_columns = [c for c in feature_columns if c in merged_df.columns]
    print(f"  Feature columns: {len(feature_columns)}")
    
    # Split back
    target_start = pd.Timestamp(year=int(config["target_month"][:4]), month=int(config["target_month"][5:7]), day=1)
    if config["target_month"] == "2026-02":
        target_end = pd.Timestamp(year=2026, month=3, day=1)
    else:
        y, m = map(int, config["target_month"].split("-"))
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
    train_dataset, test_dataset = build_datasets(
        train_df, test_df, merged_df, feature_columns, config
    )
    
    # Convert to tensors
    print("\nConverting to tensors...")
    train_X, train_y = dataset_to_tensors(train_dataset)
    test_X, test_y = dataset_to_tensors(test_dataset)
    
    # Create DataLoaders
    train_loader = DataLoader(
        TensorDataset(train_X, train_y),
        batch_size=config["batch_size"],
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_X, test_y),
        batch_size=config["batch_size"],
        shuffle=False,
    )
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dim = 24 if config["target_granularity"] == "day" else 1
    
    model_config = DeepRTSOTAModelConfig(
        model_profile=config["model_profile"],
        seq_len_days=config["seq_len_days"],
        target_mode=config["target_mode"],
        n_features=len(feature_columns),
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=0.1,
        output_dim=output_dim,
    )
    model = DeepRTSOTAModel(model_config).to(device)
    print(f"\nModel created: {config['model_profile']}, params: {sum(p.numel() for p in model.parameters())}")
    
    # Train
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.SmoothL1Loss()
    
    print(f"\nTraining for {config['epochs']} epochs...")
    for epoch in range(config["epochs"]):
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
            print(f"  Epoch {epoch+1}/{config['epochs']}: loss={train_loss:.4f}")
    
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
    
    # Compute final predictions
    test_df_original = merged_df[
        (merged_df["business_day"] >= target_start) &
        (merged_df["business_day"] < target_end)
    ].copy()
    test_days_sorted = sorted(test_df_original["business_day"].unique())
    
    da_anchor_list = []
    rt_actual_list = []
    for day in test_days_sorted:
        day_mask = test_df_original["business_day"] == day
        day_rows = test_df_original[day_mask].sort_values("hour_business")
        if len(day_rows) >= 20:
            da_anchor_list.append(day_rows["da_anchor"].values)
            rt_actual_list.append(day_rows["rt_actual"].values)
    
    da_anchor_flat = np.concatenate([x for x in da_anchor_list])
    rt_actual_flat = np.concatenate([x for x in rt_actual_list])
    residual_pred_flat = all_preds.flatten()
    final_rt_pred_flat = da_anchor_flat + residual_pred_flat
    
    mae = mean_absolute_error(rt_actual_flat, final_rt_pred_flat)
    rmse = np.sqrt(mean_squared_error(rt_actual_flat, final_rt_pred_flat))
    smape = smape_floor50(rt_actual_flat, final_rt_pred_flat)
    
    # Diagnostics
    residual_true_flat = rt_actual_flat - da_anchor_flat
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
    }
    
    # Save diagnostics
    with open(out_path / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)
    
    # Save results
    results = {
        "model_profile": config["model_profile"],
        "target_granularity": config["target_granularity"],
        "target_mode": config["target_mode"],
        "seq_len_days": config["seq_len_days"],
        "risk_features": config["risk_features"],
        "sMAPE_floor50": float(smape),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "beats_naive_baseline": smape < 63.42,
        "beats_da_anchor": smape < 26.69,
        "test_samples": len(test_dataset),
        "test_days": len(test_days_sorted),
    }
    
    with open(out_path / "metrics_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    pred_days = [str(s["business_day"]) for s in test_dataset.samples]
    pred_df = pd.DataFrame({
        "business_day": pred_days,
        "da_anchor_mean": [da_anchor_flat[i*24:(i+1)*24].mean() for i in range(len(pred_days))],
        "rt_actual_mean": [rt_actual_flat[i*24:(i+1)*24].mean() for i in range(len(pred_days))],
        "final_rt_pred_mean": [final_rt_pred_flat[i*24:(i+1)*24].mean() for i in range(len(pred_days))],
        "residual_true_mean": [residual_true_flat[i*24:(i+1)*24].mean() for i in range(len(pred_days))],
        "residual_pred_mean": [residual_pred_flat[i*24:(i+1)*24].mean() for i in range(len(pred_days))],
    })
    pred_df.to_csv(out_path / "predictions.csv", index=False)
    
    # Generate report
    original_group4_smape = 17.26
    delta_vs_original = smape - original_group4_smape
    
    report = f"""# Group 4 Reproduce Report

## Config

{json.dumps(config, indent=2)}

## Results

| Metric | Value |
|--------|-------|
| sMAPE_floor50 | {smape:.4f} |
| MAE | {mae:.4f} |
| RMSE | {rmse:.4f} |
| Test samples | {len(test_dataset)} |
| Test days | {len(test_days_sorted)} |

## Comparison

| Metric | Value |
|--------|-------|
| original_group4_smape | {original_group4_smape} |
| reproduced_smape | {smape:.4f} |
| delta_vs_original | {delta_vs_original:+.4f} pp |

## Verdict

"""
    
    if abs(delta_vs_original) < 0.5:
        verdict = "REPRODUCED"
        report += "**REPRODUCED**: reproduced_smape within 0.5pp of 17.26\n"
    elif delta_vs_original > 5.0:
        verdict = "NOT_REPRODUCED"
        report += "**NOT_REPRODUCED**: reproduced_smape closer to 27.66 (backtest result)\n"
    else:
        verdict = "INCONSISTENT"
        report += "**INCONSISTENT**: unexplained delta\n"
    
    report += f"\n## Diagnostics\n\n{json.dumps(diagnostics, indent=2)}\n"
    
    with open(out_path / "reproduce_report.md", "w") as f:
        f.write(report)
    
    print(f"\nResults saved to {out_path}")
    print(f"  sMAPE_floor50: {smape:.4f}")
    print(f"  Original Group 4 sMAPE: {original_group4_smape}")
    print(f"  Delta: {delta_vs_original:+.4f} pp")
    print(f"  Verdict: {verdict}")
    
    return results


if __name__ == "__main__":
    main()
