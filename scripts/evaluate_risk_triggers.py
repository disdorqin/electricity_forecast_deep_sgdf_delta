"""Risk Trigger Evaluation for Ledger-1.

Evaluates alert quality WITHOUT modifying prices.

Metrics:
  - negative alert precision/recall/F1
  - spike alert precision/recall/F1
  - combined high-risk alert precision/recall/F1
  - alert rate
  - top-k capture
  - lift
  - lead-time availability
  - monthly stability

Usage:
  python scripts/evaluate_risk_triggers.py \
    --risk-pack reports/local/risk_modules/risk_feature_pack_2026_01_05/risk_feature_pack.csv \
    --risk-pack-manifest reports/local/risk_modules/risk_feature_pack_2026_01_05/manifest.json \
    --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \
    --target-months 2026-01,2026-02,2026-03,2026-04,2026-05 \
    --out-dir reports/local/ledger_1/trigger_eval_2026_01_05
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json


@dataclass
class RiskTriggerEvalConfig:
    """Configuration for risk trigger evaluation."""
    risk_pack_path: str | Path
    manifest_path: Optional[str | Path] = None
    data_path: Optional[str | Path] = None
    target_months: List[str] = field(default_factory=list)
    out_dir: str | Path = "reports/local/ledger_1/trigger_eval"
    
    # Thresholds for alert
    negative_threshold: float = 0.6
    spike_threshold: float = 0.7
    delta_supply_threshold: float = 0.6
    
    # Top-k for capture
    top_k_list: List[int] = field(default_factory=lambda: [5, 10, 20])


@dataclass
class RiskTriggerEvalResult:
    """Result from risk trigger evaluation."""
    config: RiskTriggerEvalConfig
    summary: Dict[str, Any]
    monthly: pd.DataFrame
    threshold_sweep: pd.DataFrame
    report_path: Optional[Path] = None


def evaluate_risk_triggers(config: RiskTriggerEvalConfig) -> RiskTriggerEvalResult:
    """Evaluate risk trigger alert quality.

    Args:
        config: RiskTriggerEvalConfig.

    Returns:
        RiskTriggerEvalResult.
    """
    # Load risk pack
    from models.deep_sgdf_delta.risk_pack_loader import RiskPackLoader
    
    loader = RiskPackLoader()
    risk_result = loader.load(
        risk_pack_path=config.risk_pack_path,
        manifest_path=config.manifest_path,
        online_mode=False,  # Eval mode
    )
    
    if risk_result.status != "SUCCESS":
        raise RuntimeError(f"Failed to load risk pack: {risk_result.error_message}")
    
    df = risk_result.df
    
    # Load actual prices if data_path provided
    if config.data_path is not None:
        actual_df = _load_actual_prices(
            data_path=config.data_path,
            target_months=config.target_months,
        )
        
        # Ensure business_day has same dtype
        if "business_day" in df.columns:
            df["business_day"] = pd.to_datetime(df["business_day"])
        if "business_day" in actual_df.columns:
            actual_df["business_day"] = pd.to_datetime(actual_df["business_day"])
        
        # Merge actual prices
        df = df.merge(
            actual_df[["business_day", "hour_business", "target_month", "y_true"]],
            on=["business_day", "hour_business", "target_month"],
            how="left",
        )
    
    # Evaluate negative alerts
    negative_eval = _evaluate_negative_alerts(df, config)
    
    # Evaluate spike alerts
    spike_eval = _evaluate_spike_alerts(df, config)
    
    # Evaluate combined high-risk alerts
    combined_eval = _evaluate_combined_alerts(df, config)
    
    # Evaluate delta supply alerts
    delta_eval = _evaluate_delta_supply_alerts(df, config)
    
    # Compile summary
    summary = {
        "negative": negative_eval,
        "spike": spike_eval,
        "combined": combined_eval,
        "delta_supply": delta_eval,
    }
    
    # Create monthly breakdown
    monthly = _create_monthly_breakdown(df, config)
    
    # Threshold sweep
    threshold_sweep = _run_threshold_sweep(df, config)
    
    return RiskTriggerEvalResult(
        config=config,
        summary=summary,
        monthly=monthly,
        threshold_sweep=threshold_sweep,
    )


def _load_actual_prices(
    data_path: str | Path,
    target_months: List[str],
) -> pd.DataFrame:
    """Load actual prices from data file.

    Args:
        data_path: Path to data file.
        target_months: List of target months.

    Returns:
        DataFrame with actual prices.
    """
    data_path = Path(data_path)
    df = pd.read_csv(data_path)
    
    # Find price column
    price_col = None
    for col in ["price", "Price", "clearing_price", "actual"]:
        if col in df.columns:
            price_col = col
            break
    
    if price_col is None:
        raise ValueError(f"Cannot find price column in {data_path}")
    
    # Find timestamp column
    ts_col = None
    for col in ["ds", "timestamp", "date"]:
        if col in df.columns:
            ts_col = col
            break
    
    if ts_col is None:
        raise ValueError(f"Cannot find timestamp column in {data_path}")
    
    # Rename
    df = df.rename(columns={ts_col: "ds", price_col: "y_true"})
    df["ds"] = pd.to_datetime(df["ds"])
    
    # Add business time
    from models.deep_sgdf_delta.business_time import add_business_time_columns
    df = add_business_time_columns(df, timestamp_col="ds")
    
    # Filter to target months
    df["year_month"] = df["business_day"].dt.strftime("%Y-%m")
    df = df[df["year_month"].isin(set(target_months))].copy()
    
    # Add target_month
    df["target_month"] = df["business_day"].dt.strftime("%Y-%m")
    
    return df[["business_day", "hour_business", "target_month", "y_true"]].copy()


def _evaluate_negative_alerts(
    df: pd.DataFrame,
    config: RiskTriggerEvalConfig,
) -> Dict[str, Any]:
    """Evaluate negative risk alerts.

    Args:
        df: DataFrame with risk scores and y_true.
        config: RiskTriggerEvalConfig.

    Returns:
        Dict of metrics.
    """
    if "y_true" not in df.columns or "negative_prob" not in df.columns:
        return {"error": "Missing y_true or negative_prob"}
    
    # Define negative events (y_true < 100, e.g., near zero or negative)
    negative_threshold_price = 100.0
    true_negative = (df["y_true"] < negative_threshold_price).values
    
    # Predicted negative risk
    pred_negative = (df["negative_prob"] >= config.negative_threshold).values
    
    # Calculate metrics
    tp = np.sum(true_negative & pred_negative)
    fp = np.sum(~true_negative & pred_negative)
    fn = np.sum(true_negative & ~pred_negative)
    tn = np.sum(~true_negative & ~pred_negative)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Alert rate
    alert_rate = pred_negative.mean()
    
    # Top-k capture
    top_k_capture = {}
    if "y_true" in df.columns:
        # Convert to Series for proper indexing
        y_true_series = df["y_true"]
        
        for k in config.top_k_list:
            if k > len(df):
                k = len(df)
            
            top_k_idx = df["negative_prob"].nlargest(k).index
            top_k_capture[f"top_{k}_capture"] = np.sum(true_negative[top_k_idx]) / max(1, np.sum(true_negative))
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alert_rate": alert_rate,
        "n_true_negative": int(np.sum(true_negative)),
        "n_pred_negative": int(np.sum(pred_negative)),
        **top_k_capture,
    }


def _evaluate_spike_alerts(
    df: pd.DataFrame,
    config: RiskTriggerEvalConfig,
) -> Dict[str, Any]:
    """Evaluate spike risk alerts.

    Args:
        df: DataFrame with risk scores and y_true.
        config: RiskTriggerEvalConfig.

    Returns:
        Dict of metrics.
    """
    if "y_true" not in df.columns or "spike_prob" not in df.columns:
        return {"error": "Missing y_true or spike_prob"}
    
    # Define spike events (y_true > 1000, e.g., extreme price)
    spike_threshold_price = 1000.0
    true_spike = (df["y_true"] > spike_threshold_price).values
    
    # Predicted spike risk
    pred_spike = (df["spike_prob"] >= config.spike_threshold).values
    
    # Calculate metrics
    tp = np.sum(true_spike & pred_spike)
    fp = np.sum(~true_spike & pred_spike)
    fn = np.sum(true_spike & ~pred_spike)
    tn = np.sum(~true_spike & ~pred_spike)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Alert rate
    alert_rate = pred_spike.mean()
    
    # Top-k capture
    top_k_capture = {}
    if "y_true" in df.columns:
        for k in config.top_k_list:
            if k > len(df):
                k = len(df)
            
            top_k_idx = df["spike_prob"].nlargest(k).index
            top_k_capture[f"top_{k}_capture"] = np.sum(true_spike[top_k_idx]) / max(1, np.sum(true_spike))
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alert_rate": alert_rate,
        "n_true_spike": int(np.sum(true_spike)),
        "n_pred_spike": int(np.sum(pred_spike)),
        **top_k_capture,
    }


def _evaluate_combined_alerts(
    df: pd.DataFrame,
    config: RiskTriggerEvalConfig,
) -> Dict[str, Any]:
    """Evaluate combined high-risk alerts.

    Args:
        df: DataFrame with risk scores and y_true.
        config: RiskTriggerEvalConfig.

    Returns:
        Dict of metrics.
    """
    if "y_true" not in df.columns:
        return {"error": "Missing y_true"}
    
    # Define extreme events (top 10% of |y_true - median|)
    residual = np.abs(df["y_true"] - df["y_true"].median())
    extreme_threshold = np.percentile(residual, 90)
    true_extreme = (residual > extreme_threshold).values
    
    # Predicted high-risk (negative or spike)
    pred_high_risk = (
        (df.get("negative_prob", 0) >= config.negative_threshold) |
        (df.get("spike_prob", 0) >= config.spike_threshold)
    ).values
    
    # Calculate metrics
    tp = np.sum(true_extreme & pred_high_risk)
    fp = np.sum(~true_extreme & pred_high_risk)
    fn = np.sum(true_extreme & ~pred_high_risk)
    tn = np.sum(~true_extreme & ~pred_high_risk)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alert_rate": pred_high_risk.mean(),
        "n_true_extreme": int(np.sum(true_extreme)),
        "n_pred_high_risk": int(np.sum(pred_high_risk)),
    }


def _evaluate_delta_supply_alerts(
    df: pd.DataFrame,
    config: RiskTriggerEvalConfig,
) -> Dict[str, Any]:
    """Evaluate delta supply risk alerts.

    Args:
        df: DataFrame with risk scores.
        config: RiskTriggerEvalConfig.

    Returns:
        Dict of metrics.
    """
    if "deviation_down_prob" not in df.columns and "deviation_up_prob" not in df.columns:
        return {"error": "Missing deviation probabilities"}
    
    # Predicted delta supply risk
    pred_down = (df.get("deviation_down_prob", 0) >= config.delta_supply_threshold).values
    pred_up = (df.get("deviation_up_prob", 0) >= config.delta_supply_threshold).values
    pred_delta = pred_down | pred_up
    
    return {
        "alert_rate_down": pred_down.mean(),
        "alert_rate_up": pred_up.mean(),
        "alert_rate_combined": pred_delta.mean(),
        "n_pred_down": int(np.sum(pred_down)),
        "n_pred_up": int(np.sum(pred_up)),
    }


def _create_monthly_breakdown(
    df: pd.DataFrame,
    config: RiskTriggerEvalConfig,
) -> pd.DataFrame:
    """Create monthly breakdown of alert quality.

    Args:
        df: DataFrame with risk scores.
        config: RiskTriggerEvalConfig.

    Returns:
        DataFrame with monthly metrics.
    """
    # Group by target_month
    monthly_metrics = []
    
    for month in df["target_month"].unique():
        month_df = df[df["target_month"] == month].copy()
        
        # Evaluate for this month
        neg_eval = _evaluate_negative_alerts(month_df, config)
        spike_eval = _evaluate_spike_alerts(month_df, config)
        
        monthly_metrics.append({
            "target_month": month,
            "negative_precision": neg_eval.get("precision", np.nan),
            "negative_recall": neg_eval.get("recall", np.nan),
            "negative_f1": neg_eval.get("f1", np.nan),
            "spike_precision": spike_eval.get("precision", np.nan),
            "spike_recall": spike_eval.get("recall", np.nan),
            "spike_f1": spike_eval.get("f1", np.nan),
            "n_samples": len(month_df),
        })
    
    return pd.DataFrame(monthly_metrics)


def _run_threshold_sweep(
    df: pd.DataFrame,
    config: RiskTriggerEvalConfig,
) -> pd.DataFrame:
    """Run threshold sweep for alert evaluation.

    Args:
        df: DataFrame with risk scores.
        config: RiskTriggerEvalConfig.

    Returns:
        DataFrame with sweep results.
    """
    thresholds = np.linspace(0.1, 0.9, 9)
    
    sweep_results = []
    
    for thresh in thresholds:
        # Update config
        config.negative_threshold = thresh
        config.spike_threshold = thresh
        
        # Evaluate
        neg_eval = _evaluate_negative_alerts(df, config)
        spike_eval = _evaluate_spike_alerts(df, config)
        
        sweep_results.append({
            "threshold": thresh,
            "negative_precision": neg_eval.get("precision", np.nan),
            "negative_recall": neg_eval.get("recall", np.nan),
            "negative_f1": neg_eval.get("f1", np.nan),
            "spike_precision": spike_eval.get("precision", np.nan),
            "spike_recall": spike_eval.get("recall", np.nan),
            "spike_f1": spike_eval.get("f1", np.nan),
        })
    
    return pd.DataFrame(sweep_results)


class RiskTriggerEvaluator:
    """Evaluator for risk trigger alert quality.

    Usage:
        evaluator = RiskTriggerEvaluator()
        result = evaluator.evaluate(config)
        
        # Export results
        evaluator.export_results(result, config.out_dir)
    """

    def __init__(self):
        self.last_result: Optional[RiskTriggerEvalResult] = None

    def evaluate(self, config: RiskTriggerEvalConfig) -> RiskTriggerEvalResult:
        """Evaluate risk triggers.

        Args:
            config: RiskTriggerEvalConfig.

        Returns:
            RiskTriggerEvalResult.
        """
        result = evaluate_risk_triggers(config)
        self.last_result = result
        return result

    def export_results(
        self,
        result: Optional[RiskTriggerEvalResult] = None,
        out_dir: Optional[str | Path] = None,
    ):
        """Export evaluation results.

        Args:
            result: RiskTriggerEvalResult. If None, uses last_result.
            out_dir: Output directory. If None, uses config.out_dir.
        """
        if result is None:
            result = self.last_result
        
        if result is None:
            raise RuntimeError("No result to export. Call evaluate() first.")
        
        out_dir = Path(out_dir or result.config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Export summary
        with open(out_dir / "trigger_eval_summary.json", "w") as f:
            json.dump(result.summary, f, indent=2, default=str)
        
        # Export monthly
        result.monthly.to_csv(out_dir / "trigger_eval_monthly.csv", index=False)
        
        # Export threshold sweep
        result.threshold_sweep.to_csv(out_dir / "trigger_eval_threshold_sweep.csv", index=False)
        
        # Generate report
        report_path = _generate_trigger_eval_report(result, out_dir)
        result.report_path = report_path
        
        print(f"Results exported to {out_dir}")


def _generate_trigger_eval_report(
    result: RiskTriggerEvalResult,
    out_dir: Path,
) -> Path:
    """Generate trigger evaluation report.

    Args:
        result: RiskTriggerEvalResult.
        out_dir: Output directory.

    Returns:
        Path to report.
    """
    report_path = out_dir / "trigger_eval_report.md"
    
    with open(report_path, "w") as f:
        f.write("# Risk Trigger Evaluation Report\n\n")
        
        f.write("## Configuration\n\n")
        f.write(f"- **Risk pack**: {result.config.risk_pack_path}\n")
        f.write(f"- **Negative threshold**: {result.config.negative_threshold}\n")
        f.write(f"- **Spike threshold**: {result.config.spike_threshold}\n")
        f.write(f"- **Delta supply threshold**: {result.config.delta_supply_threshold}\n")
        f.write("\n")
        
        f.write("## Summary\n\n")
        
        # Negative alerts
        neg = result.summary.get("negative", {})
        if "error" not in neg:
            f.write("### Negative Alerts\n\n")
            f.write(f"- **Precision**: {neg.get('precision', 0.0):.4f}\n")
            f.write(f"- **Recall**: {neg.get('recall', 0.0):.4f}\n")
            f.write(f"- **F1**: {neg.get('f1', 0.0):.4f}\n")
            f.write(f"- **Alert rate**: {neg.get('alert_rate', 0.0):.2%}\n")
            f.write(f"- **N true negative**: {neg.get('n_true_negative', 0)}\n")
            f.write(f"- **N pred negative**: {neg.get('n_pred_negative', 0)}\n")
            
            for key, value in neg.items():
                if key.startswith("top_"):
                    f.write(f"- **{key.replace('_', ' ').title()}**: {value:.4f}\n")
            
            f.write("\n")
        
        # Spike alerts
        spike = result.summary.get("spike", {})
        if "error" not in spike:
            f.write("### Spike Alerts\n\n")
            f.write(f"- **Precision**: {spike.get('precision', 0.0):.4f}\n")
            f.write(f"- **Recall**: {spike.get('recall', 0.0):.4f}\n")
            f.write(f"- **F1**: {spike.get('f1', 0.0):.4f}\n")
            f.write(f"- **Alert rate**: {spike.get('alert_rate', 0.0):.2%}\n")
            f.write(f"- **N true spike**: {spike.get('n_true_spike', 0)}\n")
            f.write(f"- **N pred spike**: {spike.get('n_pred_spike', 0)}\n")
            
            for key, value in spike.items():
                if key.startswith("top_"):
                    f.write(f"- **{key.replace('_', ' ').title()}**: {value:.4f}\n")
            
            f.write("\n")
        
        f.write("## Conclusion\n\n")
        f.write("Risk trigger evaluation complete.\n")
        f.write("\n")
        f.write("## Files Generated\n\n")
        f.write("- `trigger_eval_summary.json`: Summary metrics.\n")
        f.write("- `trigger_eval_monthly.csv`: Monthly breakdown.\n")
        f.write("- `trigger_eval_threshold_sweep.csv`: Threshold sweep results.\n")
        f.write("- `trigger_eval_report.md`: This report.\n")
    
    return report_path
