#!/usr/bin/env python
"""Export a trained TrendKnightRT model as a self-contained deployable model pack.

Copies all required artifacts from a training run into a clean output directory,
generates a README.md with model metadata and usage examples, and validates
that the pack is complete and loadable.

Usage:
    python scripts/export_realtime_model_pack.py \
        --model-dir artifacts/trendknight_rt/exp_001 \
        --out-dir artifacts/trendknight_rt/champion

    python scripts/export_realtime_model_pack.py \
        --model-dir artifacts/trendknight_rt/exp_001 \
        --out-dir artifacts/trendknight_rt/champion \
        --skip-validation
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

# -- Path setup ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_realtime_model_pack")

# -- Required files in a model pack -------------------------------------------

REQUIRED_FILES = [
    "best_model.pt",
    "config.yaml",
    "feature_manifest.json",
    "train_manifest.json",
    "metrics_summary.json",
]

OPTIONAL_FILES = [
    "training_curves.csv",
]


# -- CLI ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export TrendKnightRT model as a deployable model pack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Required files in model-dir:
  best_model.pt             Trained model checkpoint
  config.yaml               Training configuration
  feature_manifest.json     Feature contract metadata
  train_manifest.json       Training run metadata
  metrics_summary.json      Evaluation metrics

Output:
  out-dir/                  Self-contained model pack with README.md
""",
    )
    parser.add_argument(
        "--model-dir", type=str, required=True,
        help="Directory containing trained model artifacts",
    )
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="Output directory for the model pack",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip model loading validation after export",
    )
    return parser.parse_args()


# -- Data loading helpers -----------------------------------------------------

def load_json(path: Path) -> dict:
    """Load a JSON file, returning empty dict on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return {}


def load_yaml_config(path: Path) -> dict:
    """Load a YAML config file, returning empty dict on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return {}


# -- Copy artifacts -----------------------------------------------------------

def copy_artifacts(model_dir: Path, out_dir: Path) -> list[str]:
    """Copy all required and optional files from model_dir to out_dir.

    Returns a list of copied filenames.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []

    for fname in REQUIRED_FILES + OPTIONAL_FILES:
        src = model_dir / fname
        if src.exists():
            dst = out_dir / fname
            shutil.copy2(src, dst)
            copied.append(fname)
            logger.info("Copied: %s", fname)
        elif fname in REQUIRED_FILES:
            logger.error("Required file missing: %s", fname)
        else:
            logger.info("Optional file not found (skipping): %s", fname)

    return copied


# -- README generation --------------------------------------------------------

def generate_readme(
    out_dir: Path,
    config: dict,
    train_manifest: dict,
    feature_manifest: dict,
    metrics_summary: dict,
) -> str:
    """Generate a README.md for the model pack."""
    model_name = config.get("model_profile", train_manifest.get("model_profile", "TrendKnightRT"))
    backbone = config.get("profile", {}).get("backbone", train_manifest.get("backbone", "unknown"))
    target_month = config.get("target_month", train_manifest.get("target_month", "unknown"))
    trained_at = train_manifest.get("trained_at", "unknown")
    n_params = train_manifest.get("n_params", "unknown")
    best_val_smape = train_manifest.get("best_val_smape_floor50",
                                        metrics_summary.get("best_val_smape_floor50", "N/A"))
    best_epoch = train_manifest.get("best_epoch", "N/A")
    data_path = config.get("data_path", "N/A")

    # Model config details
    model_cfg = config.get("model", {})
    hidden_dim = model_cfg.get("hidden_dim", "N/A")
    num_layers = model_cfg.get("num_layers", "N/A")
    dropout = model_cfg.get("dropout", "N/A")
    fusion_mode = model_cfg.get("fusion_mode", "N/A")
    multiscale = model_cfg.get("multiscale", "N/A")
    input_dim = model_cfg.get("input_dim", train_manifest.get("input_dim", "N/A"))

    # Training hyperparameters
    training_cfg = config.get("training", {})
    epochs = training_cfg.get("epochs", "N/A")
    batch_size = training_cfg.get("batch_size", "N/A")
    lr = training_cfg.get("lr", "N/A")
    patience = training_cfg.get("patience", "N/A")
    seed = training_cfg.get("seed", "N/A")

    # Feature info
    feature_version = feature_manifest.get("feature_version", "N/A")
    feature_cols = feature_manifest.get("feature_cols", feature_manifest.get("all_features", []))
    required_features = feature_manifest.get("required_features", [])
    optional_features = feature_manifest.get("optional_features", [])

    # Metrics
    test_metrics = metrics_summary.get("test_metrics", {})

    # Build README content
    lines = [
        f"# {model_name} — Deployable Model Pack",
        "",
        f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Model Version:** {model_name}",
        f"**Training Date:** {trained_at}",
        f"**Target Month:** {target_month}",
        "",
        "## Model Architecture",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Backbone | {backbone} |",
        f"| Hidden dim | {hidden_dim} |",
        f"| Num layers | {num_layers} |",
        f"| Dropout | {dropout} |",
        f"| Fusion mode | {fusion_mode} |",
        f"| Multiscale | {multiscale} |",
        f"| Input dim | {input_dim} |",
        f"| Parameters | {n_params:,} |" if isinstance(n_params, int) else f"| Parameters | {n_params} |",
        "",
        "## Training Configuration",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Epochs (max) | {epochs} |",
        f"| Best epoch | {best_epoch} |",
        f"| Batch size | {batch_size} |",
        f"| Learning rate | {lr} |",
        f"| Early stopping patience | {patience} |",
        f"| Random seed | {seed} |",
        f"| Data path | {data_path} |",
        "",
        "## Input Requirements",
        "",
        f"**Feature version:** {feature_version}",
        f"**Input dimension:** {input_dim}",
        "",
    ]

    if required_features:
        lines.append("### Required Features")
        lines.append("")
        for feat in required_features:
            lines.append(f"- `{feat}`")
        lines.append("")

    if optional_features:
        lines.append("### Optional Features")
        lines.append("")
        for feat in optional_features:
            lines.append(f"- `{feat}`")
        lines.append("")

    if feature_cols:
        lines.append("### All Feature Columns")
        lines.append("")
        lines.append("```")
        for feat in feature_cols:
            lines.append(f"  {feat}")
        lines.append("```")
        lines.append("")

    lines.extend([
        "### Data Format",
        "",
        "Input data should be an hourly CSV with at minimum:",
        "- Timestamp column (`ds` or `时刻`)",
        "- Day-ahead price (`da_anchor` or `forecast_price` or `日前电价`)",
        "- Realtime price (`rt_actual` or `rt_price` or `实时电价`) — for evaluation only",
        "",
        "The model expects 24-hour blocks aligned by business day convention:",
        "- Timestamp D 01:00~23:00 -> business_day=D, hour=1~23",
        "- Timestamp D 00:00 -> business_day=D-1, hour=24",
        "",
        "## Output Fields",
        "",
        "| Field | Description |",
        "|-------|-------------|",
        "| `rt_pred` | Predicted realtime electricity price |",
        "| `delta_pred` | Predicted delta (rt - da_anchor) |",
        "| `residual_to_sgdfnet` | Predicted residual to SGDFNet prediction |",
        "| `confidence` | Per-hour confidence score [0, 1] |",
        "| `da_anchor` | Day-ahead anchor price (passthrough) |",
        "| `sgdfnet_pred` | SGDFNet prediction (passthrough) |",
        "",
        "## Metrics Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Best val sMAPE_floor50 | {best_val_smape} |",
    ])

    if test_metrics:
        for key, val in test_metrics.items():
            if isinstance(val, float):
                lines.append(f"| {key} | {val:.4f} |")
            else:
                lines.append(f"| {key} | {val} |")

    lines.extend([
        "",
        "## Pack Contents",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `best_model.pt` | Trained model checkpoint (state_dict + config) |",
        "| `config.yaml` | Full training configuration |",
        "| `feature_manifest.json` | Feature contract and version |",
        "| `train_manifest.json` | Training run metadata |",
        "| `metrics_summary.json` | Evaluation metrics |",
        "| `README.md` | This file |",
        "",
        "## Usage Examples",
        "",
        "### Predict (single day)",
        "",
        "```bash",
        "python scripts/predict_realtime_deep_model.py \\",
        f"    --model-dir {out_dir} \\",
        "    --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \\",
        "    --decision-day 2026-02-01 \\",
        "    --out outputs/predictions.csv",
        "```",
        "",
        "### Predict (batch)",
        "",
        "```bash",
        "python scripts/predict_realtime_deep_model.py \\",
        f"    --model-dir {out_dir} \\",
        "    --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \\",
        "    --decision-days 2026-02-01,2026-02-02,2026-02-03 \\",
        "    --out outputs/predictions.csv",
        "```",
        "",
        "### Evaluate",
        "",
        "```bash",
        "python scripts/evaluate_realtime_deep_model.py \\",
        "    --predictions outputs/predictions.csv \\",
        "    --ground-truth ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \\",
        "    --out reports/eval_output",
        "```",
        "",
    ])

    readme_content = "\n".join(lines)
    readme_path = out_dir / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    logger.info("README.md written to %s", readme_path)

    return readme_content


# -- Validation ---------------------------------------------------------------

def validate_pack(out_dir: Path, skip_validation: bool = False) -> bool:
    """Validate the exported model pack.

    Checks:
    1. All required files present
    2. config.yaml is loadable
    3. JSON files are parseable
    4. Model checkpoint is loadable (unless --skip-validation)

    Returns True if all checks pass.
    """
    logger.info("Validating model pack at %s ...", out_dir)
    all_ok = True

    # Check required files
    for fname in REQUIRED_FILES:
        fpath = out_dir / fname
        if not fpath.exists():
            logger.error("MISSING required file: %s", fname)
            all_ok = False
        else:
            size_kb = fpath.stat().st_size / 1024
            logger.info("OK: %s (%.1f KB)", fname, size_kb)

    # Validate config.yaml is loadable
    config_path = out_dir / "config.yaml"
    if config_path.exists():
        config = load_yaml_config(config_path)
        if not config:
            logger.error("config.yaml is empty or not valid YAML")
            all_ok = False
        else:
            logger.info("OK: config.yaml loadable (%d keys)", len(config))

    # Validate JSON files
    for fname in ["feature_manifest.json", "train_manifest.json", "metrics_summary.json"]:
        fpath = out_dir / fname
        if fpath.exists():
            data = load_json(fpath)
            if not data:
                logger.error("%s is empty or not valid JSON", fname)
                all_ok = False
            else:
                logger.info("OK: %s loadable (%d keys)", fname, len(data))

    # Validate model checkpoint is loadable
    if not skip_validation:
        model_path = out_dir / "best_model.pt"
        if model_path.exists():
            try:
                import torch
                ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict):
                    if "state_dict" in ckpt:
                        n_params = sum(
                            v.numel() for v in ckpt["state_dict"].values()
                            if hasattr(v, "numel")
                        )
                        logger.info(
                            "OK: best_model.pt loadable (state_dict with %s params)",
                            f"{n_params:,}",
                        )
                    elif "model_state_dict" in ckpt:
                        logger.info("OK: best_model.pt loadable (model_state_dict format)")
                    else:
                        logger.info(
                            "OK: best_model.pt loadable (dict with keys: %s)",
                            list(ckpt.keys()),
                        )
                else:
                    logger.info("OK: best_model.pt loadable (type: %s)", type(ckpt).__name__)
            except Exception as e:
                logger.error("Failed to load best_model.pt: %s", e)
                all_ok = False
        else:
            logger.error("best_model.pt not found — cannot validate model")
            all_ok = False
    else:
        logger.info("Skipping model validation (--skip-validation)")

    # Validate model instantiable (try to build from config)
    if not skip_validation:
        try:
            _validate_model_instantiable(out_dir)
        except Exception as e:
            logger.warning("Model instantiation check failed: %s", e)
            # Non-fatal: the pack may still be usable
    return all_ok


def _validate_model_instantiable(out_dir: Path) -> None:
    """Try to instantiate the model from the pack config to verify compatibility."""
    import torch
    from models.deep_sgdf_delta.trendknight_rt import (
        TrendKnightRTConfig,
        build_trendknight_rt,
    )

    config = load_yaml_config(out_dir / "config.yaml")
    model_cfg = config.get("model", {})

    if not model_cfg:
        logger.warning("No model config found in config.yaml — skipping instantiation check")
        return

    rt_config = TrendKnightRTConfig(
        input_dim=model_cfg.get("input_dim", 40),
        hidden_dim=model_cfg.get("hidden_dim", 64),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.1),
        backbone=model_cfg.get("backbone", "tcn"),
        tcn_kernel_size=model_cfg.get("tcn_kernel_size", 3),
        tcn_dilation_base=model_cfg.get("tcn_dilation_base", 2),
        transformer_nhead=model_cfg.get("transformer_nhead", 4),
        transformer_dim_ff=model_cfg.get("transformer_dim_ff", 128),
        use_sgdfnet_residual_head=model_cfg.get("use_sgdfnet_residual_head", True),
        use_delta_head=model_cfg.get("use_delta_head", True),
        use_confidence_head=model_cfg.get("use_confidence_head", True),
        use_period_bias=model_cfg.get("use_period_bias", True),
        fusion_mode=model_cfg.get("fusion_mode", "C"),
        hour_embed_dim=model_cfg.get("hour_embed_dim", 8),
        segment_embed_dim=model_cfg.get("segment_embed_dim", 8),
        multiscale=model_cfg.get("multiscale", True),
        teacher_input_dim=model_cfg.get("teacher_input_dim", 0),
    )

    model = build_trendknight_rt(rt_config)
    model.eval()

    # Load state dict and verify
    ckpt = torch.load(out_dir / "best_model.pt", map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys when loading state_dict: %d keys", len(missing))
    if unexpected:
        logger.warning("Unexpected keys when loading state_dict: %d keys", len(unexpected))

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("OK: Model instantiable — %s backbone, %s params", rt_config.backbone, f"{n_params:,}")


# -- Main ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)

    if not model_dir.exists():
        logger.error("Model directory does not exist: %s", model_dir)
        sys.exit(1)

    if model_dir.resolve() == out_dir.resolve():
        logger.error("model-dir and out-dir must be different directories")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  TrendKnightRT Model Pack Export")
    logger.info("=" * 60)
    logger.info("  Source:  %s", model_dir)
    logger.info("  Target:  %s", out_dir)
    logger.info("=" * 60)

    # 1. Copy artifacts
    logger.info("")
    logger.info("Step 1: Copying artifacts ...")
    copied = copy_artifacts(model_dir, out_dir)

    # Check all required files were copied
    missing_required = [f for f in REQUIRED_FILES if f not in copied]
    if missing_required:
        logger.error("Missing required files: %s", missing_required)
        sys.exit(1)

    # 2. Load metadata for README
    logger.info("")
    logger.info("Step 2: Loading metadata ...")
    config = load_yaml_config(out_dir / "config.yaml")
    train_manifest = load_json(out_dir / "train_manifest.json")
    feature_manifest = load_json(out_dir / "feature_manifest.json")
    metrics_summary = load_json(out_dir / "metrics_summary.json")

    # 3. Generate README
    logger.info("")
    logger.info("Step 3: Generating README.md ...")
    generate_readme(out_dir, config, train_manifest, feature_manifest, metrics_summary)

    # 4. Validate
    logger.info("")
    logger.info("Step 4: Validating model pack ...")
    is_valid = validate_pack(out_dir, skip_validation=args.skip_validation)

    # Summary
    print()
    print("=" * 60)
    print("  Model Pack Export Complete")
    print("=" * 60)
    print(f"  Source:     {model_dir}")
    print(f"  Output:     {out_dir}")
    print(f"  Files:      {len(copied)} copied")
    print(f"  Valid:      {'YES' if is_valid else 'NO — check errors above'}")
    print()
    print("  Pack contents:")
    for fpath in sorted(out_dir.iterdir()):
        if fpath.is_file():
            size_kb = fpath.stat().st_size / 1024
            print(f"    {fpath.name:<30s}  {size_kb:>10.1f} KB")
    print("=" * 60)

    if not is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
