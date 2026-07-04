"""
Predict using RT-Assist model pack.

Usage:
    python scripts/predict_rt_assist_pack.py \
        --model-dir exported_models/rt_assist_pack \
        --data-path data/preprocessed_data.csv \
        --start 2025-01-01 \
        --end 2025-01-31 \
        --output predictions/rt_assist_2025_01.csv

Output: hour-level CSV with RT_ASSIST_OUTPUT_COLUMNS.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Predict using RT-Assist model pack")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to exported model pack directory")
    parser.add_argument("--data-path", type=str, required=True, help="Path to input data CSV")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD), inclusive")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD), inclusive")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--enable-safe-correction", action="store_true", help="Enable safe correction (default: disabled)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Correction strength (default: 1.0)")
    parser.add_argument("--clip", type=float, default=0.0, help="Max correction clip (default: 0 = no clip)")
    args = parser.parse_args()

    # Load model pack
    from models.deep_sgdf_delta.rt_assist_model import create_rt_assist_model
    model = create_rt_assist_model(
        model_dir=args.model_dir,
        enable_safe_correction=args.enable_safe_correction,
        alpha=args.alpha,
        clip_correction=args.clip,
    )
    logger.info(f"Model loaded from {args.model_dir}")
    logger.info(f"  safe_correction: {args.enable_safe_correction}")
    logger.info(f"  alpha: {args.alpha}")
    logger.info(f"  clip: {args.clip}")

    # Load data
    df = pd.read_csv(args.data_path, parse_dates=["ds"] if "ds" in pd.read_csv(args.data_path, nrows=0).columns else ["times"])
    logger.info(f"Data loaded: {len(df)} rows")

    # Normalize column names
    if "times" in df.columns and "ds" not in df.columns:
        df = df.rename(columns={"times": "ds"})
    if "rt_price" in df.columns and "rt_actual" not in df.columns:
        df["rt_actual"] = df["rt_price"]
    if "da_price" in df.columns and "da_anchor" not in df.columns:
        df["da_anchor"] = df["da_price"]

    # Filter by date range
    df["ds"] = pd.to_datetime(df["ds"])
    if args.start:
        df = df[df["ds"] >= pd.Timestamp(args.start)]
    if args.end:
        df = df[df["ds"] <= pd.Timestamp(args.end)]

    logger.info(f"Filtered data: {len(df)} rows ({args.start} to {args.end})")

    # Add business_time columns
    from models.deep_sgdf_delta.business_time import add_business_time_columns
    if "business_day" not in df.columns:
        df = add_business_time_columns(df, timestamp_col="ds")

    # Build feature columns (for residual model, if loaded)
    feature_cols = []
    if model.residual_model is not None:
        # Use features from manifest
        feature_cols = model.manifest.get("feature_columns", [])
        feature_cols = [c for c in feature_cols if c in df.columns]
        logger.info(f"Using {len(feature_cols)} feature columns for residual model")

    # Run prediction
    result_df = model.predict(df, feature_columns=feature_cols)
    logger.info(f"Prediction done: {len(result_df)} rows")

    # Compute sMAPE if rt_actual available
    if "rt_actual" in df.columns:
        from sklearn.metrics import mean_absolute_percentage_error
        y_true = df["rt_actual"].values
        y_pred = result_df["rt_pred"].values

        # Hourly sMAPE
        def smape(y, yhat):
            return 100 * np.mean(2 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat) + 1e-8))

        hourly_smape = smape(y_true, y_pred)
        logger.info(f"Hourly sMAPE: {hourly_smape:.2f}")

        # Day-level sMAPE
        result_df["date"] = pd.to_datetime(result_df["ds"]).dt.date
        daily_true = pd.DataFrame({"date": pd.to_datetime(df["ds"]).dt.date, "y": y_true}).groupby("date")["y"].mean()
        daily_pred = pd.DataFrame({"date": result_df["date"], "yhat": y_pred}).groupby("date")["yhat"].mean()
        day_smape = smape(daily_true.values, daily_pred.values)
        logger.info(f"Day-level sMAPE: {day_smape:.2f}")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Output saved to {output_path}")

    # Print sample
    print("\n" + "=" * 60)
    print("Sample predictions (first 5 rows):")
    print("=" * 60)
    print(result_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
