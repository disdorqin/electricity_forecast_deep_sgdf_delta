"""Unified risk calibration library for electricity price prediction models.

Provides threshold sweeping, top-k capture analysis, calibration metrics,
and objective-based threshold selection for risk modules (spike detection,
negative price detection, etc.).

All functions operate on numpy arrays of true labels (binary) and predicted
probabilities (continuous [0, 1]).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


class ThresholdResult(float):
    """A float subclass that also carries dict-like metadata.

    Backward-compatible: ``isinstance(result, float)`` is True.
    Backtest scripts can index: ``result["best_threshold"]``.
    """

    def __new__(cls, value: float, meta: dict | None = None):
        obj = super().__new__(cls, value)
        obj._meta = meta or {}
        return obj

    def __getitem__(self, key: str):
        return self._meta[key]

    def __contains__(self, key: str) -> bool:
        return key in self._meta

    def get(self, key: str, default=None):
        return self._meta.get(key, default)


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | list[float] | None = None,
) -> pd.DataFrame:
    """Sweep thresholds and compute precision/recall/f1/support/alert_rate per threshold.

    Args:
        y_true: Binary ground truth labels (0 or 1), shape (n,).
        y_prob: Predicted probabilities, shape (n,), values in [0, 1].
        thresholds: Array of thresholds to sweep. If None, uses 101 evenly
            spaced values from 0.0 to 1.0.

    Returns:
        DataFrame with columns: threshold, precision, recall, f1, support,
        alert_rate, tp, fp, tn, fn.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    n = len(y_true)
    n_positive = float(np.sum(y_true == 1))

    rows = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(float)
        tp = float(np.sum((y_pred == 1) & (y_true == 1)))
        fp = float(np.sum((y_pred == 1) & (y_true == 0)))
        tn = float(np.sum((y_pred == 0) & (y_true == 0)))
        fn = float(np.sum((y_pred == 0) & (y_true == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        alert_rate = (tp + fp) / n if n > 0 else 0.0

        rows.append({
            "threshold": float(thr),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": int(n_positive),
            "alert_rate": round(alert_rate, 6),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        })

    return pd.DataFrame(rows)


def top_k_capture(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_pcts: list[float] | None = None,
) -> pd.DataFrame:
    """Compute top-k capture rates, lift, precision_at_k, recall_at_k.

    For each k_pct, selects the top k_pct% of samples by predicted probability
    and measures how many true positives are captured.

    Args:
        y_true: Binary ground truth labels (0 or 1), shape (n,).
        y_prob: Predicted probabilities, shape (n,), values in [0, 1].
        k_pcts: List of top-k percentages (e.g. [1, 3, 5, 10, 20]).
            Default is [1, 3, 5, 10, 20].

    Returns:
        DataFrame with columns: k_pct, k_count, tp_captured, total_positives,
        capture_rate, precision_at_k, recall_at_k, lift, baseline_rate.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if k_pcts is None:
        k_pcts = [1, 3, 5, 10, 20]

    n = len(y_true)
    total_positives = int(np.sum(y_true == 1))
    baseline_rate = total_positives / n if n > 0 else 0.0

    # Sort by predicted probability descending
    sorted_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[sorted_indices]

    rows = []
    for k_pct in k_pcts:
        k_count = max(1, int(np.ceil(n * k_pct / 100.0)))
        k_count = min(k_count, n)  # cannot exceed total samples

        top_k_true = y_true_sorted[:k_count]
        tp_captured = int(np.sum(top_k_true == 1))

        capture_rate = tp_captured / total_positives if total_positives > 0 else 0.0
        precision_at_k = tp_captured / k_count if k_count > 0 else 0.0
        recall_at_k = capture_rate  # same as capture_rate
        lift = precision_at_k / baseline_rate if baseline_rate > 0 else 0.0

        rows.append({
            "k_pct": k_pct,
            "k_count": k_count,
            "tp_captured": tp_captured,
            "total_positives": total_positives,
            "capture_rate": round(capture_rate, 6),
            "precision_at_k": round(precision_at_k, 6),
            "recall_at_k": round(recall_at_k, 6),
            "lift": round(lift, 6),
            "baseline_rate": round(baseline_rate, 6),
        })

    return pd.DataFrame(rows)


def lift_at_k(y_true: np.ndarray, y_prob: np.ndarray, k_pct: float) -> float:
    """Compute lift at a specific top-k percentage.

    Lift = precision_at_k / baseline_rate.
    A lift > 1 means the model concentrates positives in the top-k better
    than random selection.

    Args:
        y_true: Binary ground truth labels (0 or 1), shape (n,).
        y_prob: Predicted probabilities, shape (n,), values in [0, 1].
        k_pct: Top-k percentage (e.g. 10 means top 10%).

    Returns:
        Lift value (float >= 0). Returns 0.0 if baseline_rate is 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    n = len(y_true)
    if n == 0:
        return 0.0

    total_positives = int(np.sum(y_true == 1))
    baseline_rate = total_positives / n

    if baseline_rate == 0:
        return 0.0

    k_count = max(1, int(np.ceil(n * k_pct / 100.0)))
    k_count = min(k_count, n)

    sorted_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[sorted_indices]
    top_k_true = y_true_sorted[:k_count]

    tp_captured = int(np.sum(top_k_true == 1))
    precision_at_k = tp_captured / k_count if k_count > 0 else 0.0

    return float(precision_at_k / baseline_rate)


def precision_recall_curve_summary(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute a summary of the precision-recall curve.

    Args:
        y_true: Binary ground truth labels (0 or 1), shape (n,).
        y_prob: Predicted probabilities, shape (n,), values in [0, 1].

    Returns:
        Dict with keys:
          - auc_pr: area under the precision-recall curve (trapezoidal).
          - max_f1: maximum F1 score across thresholds.
          - max_f1_threshold: threshold at which max F1 occurs.
          - precision_at_50pct_recall: precision when recall >= 0.5.
          - n_thresholds: number of thresholds evaluated.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Sweep fine-grained thresholds
    thresholds = np.linspace(0.0, 1.0, 201)
    sweep_df = threshold_sweep(y_true, y_prob, thresholds)

    # AUC-PR via trapezoidal rule (recall on x-axis, precision on y-axis)
    # Sort by recall ascending for proper AUC computation
    sweep_sorted = sweep_df.sort_values("recall").reset_index(drop=True)
    recalls = sweep_sorted["recall"].values
    precisions = sweep_sorted["precision"].values
    auc_pr = float(np.trapezoid(precisions, recalls))
    auc_pr = abs(auc_pr)  # ensure non-negative

    # Max F1
    max_f1_idx = sweep_df["f1"].idxmax()
    max_f1 = float(sweep_df.loc[max_f1_idx, "f1"])
    max_f1_threshold = float(sweep_df.loc[max_f1_idx, "threshold"])

    # Precision at 50% recall
    above_50_recall = sweep_df[sweep_df["recall"] >= 0.5]
    if not above_50_recall.empty:
        precision_at_50_recall = float(above_50_recall["precision"].max())
    else:
        precision_at_50_recall = 0.0

    return {
        "auc_pr": round(auc_pr, 6),
        "max_f1": round(max_f1, 6),
        "max_f1_threshold": round(max_f1_threshold, 6),
        "precision_at_50pct_recall": round(precision_at_50_recall, 6),
        "n_thresholds": len(thresholds),
    }


def calibration_bucket_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """Bucket calibration table: pred_mean, actual_rate, count, calibration_error per bucket.

    Divides [0, 1] into n_buckets equal-width bins. For each bin, computes
    the mean predicted probability, the actual positive rate, the count of
    samples, and the absolute calibration error.

    Args:
        y_true: Binary ground truth labels (0 or 1), shape (n,).
        y_prob: Predicted probabilities, shape (n,), values in [0, 1].
        n_buckets: Number of equal-width buckets. Default 10.

    Returns:
        DataFrame with columns: bucket_idx, bin_low, bin_high, count,
        pred_mean, actual_rate, calibration_error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_buckets + 1)
    rows = []

    for i in range(n_buckets):
        low = bin_edges[i]
        high = bin_edges[i + 1]

        if i < n_buckets - 1:
            mask = (y_prob >= low) & (y_prob < high)
        else:
            # Last bucket includes the right edge
            mask = (y_prob >= low) & (y_prob <= high)

        count = int(np.sum(mask))
        if count > 0:
            pred_mean = float(np.mean(y_prob[mask]))
            actual_rate = float(np.mean(y_true[mask]))
            cal_error = abs(pred_mean - actual_rate)
        else:
            pred_mean = 0.0
            actual_rate = 0.0
            cal_error = 0.0

        rows.append({
            "bucket_idx": i,
            "bin_low": round(low, 4),
            "bin_high": round(high, 4),
            "count": count,
            "pred_mean": round(pred_mean, 6),
            "actual_rate": round(actual_rate, 6),
            "calibration_error": round(cal_error, 6),
        })

    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_buckets: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Weighted average of |pred_mean - actual_rate| across buckets, where
    weights are the fraction of samples in each bucket.

    Args:
        y_true: Binary ground truth labels (0 or 1), shape (n,).
        y_prob: Predicted probabilities, shape (n,), values in [0, 1].
        n_buckets: Number of equal-width buckets. Default 10.

    Returns:
        ECE value in [0, 1]. 0 means perfectly calibrated.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    n = len(y_true)
    if n == 0:
        return 0.0

    bucket_df = calibration_bucket_table(y_true, y_prob, n_buckets)

    total_count = bucket_df["count"].sum()
    if total_count == 0:
        return 0.0

    weights = bucket_df["count"].values / total_count
    errors = bucket_df["calibration_error"].values

    ece = float(np.sum(weights * errors))
    return round(ece, 6)


def select_threshold_by_objective(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective: str,
    **kwargs,
) -> ThresholdResult:
    """Select the best threshold by a named objective.

    Supported objectives:
      - "max_f1" or "f1": maximize F1 score.
      - "min_precision_30": lowest threshold where precision >= 0.30.
      - "min_recall_50": lowest threshold where recall >= 0.50.
      - "top10_lift": threshold that maximizes lift at top-10%.
      - "alert_budget_5pct": highest threshold where alert_rate <= 0.05.
      - "alert_budget_10pct": highest threshold where alert_rate <= 0.10.

    Args:
        y_true: Binary ground truth labels (0 or 1), shape (n,).
        y_prob: Predicted probabilities, shape (n,), values in [0, 1].
        objective: Name of the objective (see above).
        **kwargs: Additional parameters for specific objectives.

    Returns:
        ThresholdResult — a float subclass that also supports dict-like
        access: ``result["best_threshold"]``, ``result["f1"]``, etc.
        Backward-compatible: ``isinstance(result, float)`` is True.

    Raises:
        ValueError: If objective is unknown or no valid threshold found.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Normalise aliases.
    _OBJECTIVE_ALIASES = {"f1": "max_f1", "precision_recall": "min_precision_30",
                          "recall_precision": "min_recall_50"}
    objective = _OBJECTIVE_ALIASES.get(objective, objective)

    thresholds = np.linspace(0.0, 1.0, 201)
    sweep_df = threshold_sweep(y_true, y_prob, thresholds)

    def _metrics_at(thr: float) -> ThresholdResult:
        row = sweep_df.iloc[(sweep_df["threshold"] - thr).abs().argmin()]
        meta = {
            "best_threshold": round(float(row["threshold"]), 6),
            "f1": round(float(row["f1"]), 6),
            "precision": round(float(row["precision"]), 6),
            "recall": round(float(row["recall"]), 6),
        }
        return ThresholdResult(meta["best_threshold"], meta)

    if objective == "max_f1":
        idx = sweep_df["f1"].idxmax()
        return _metrics_at(float(sweep_df.loc[idx, "threshold"]))

    elif objective == "min_precision_30":
        target_precision = kwargs.get("target_precision", 0.30)
        valid = sweep_df[sweep_df["precision"] >= target_precision]
        if valid.empty:
            raise ValueError(
                f"No threshold achieves precision >= {target_precision}. "
                f"Max precision in sweep: {sweep_df['precision'].max():.4f}"
            )
        idx = valid["threshold"].idxmin()
        return _metrics_at(float(valid.loc[idx, "threshold"]))

    elif objective == "min_recall_50":
        target_recall = kwargs.get("target_recall", 0.50)
        valid = sweep_df[sweep_df["recall"] >= target_recall]
        if valid.empty:
            raise ValueError(
                f"No threshold achieves recall >= {target_recall}. "
                f"Max recall in sweep: {sweep_df['recall'].max():.4f}"
            )
        idx = valid["threshold"].idxmin()
        return _metrics_at(float(valid.loc[idx, "threshold"]))

    elif objective == "top10_lift":
        best_thr = 0.5
        best_lift = -1.0
        for thr in thresholds:
            y_pred = (y_prob >= thr).astype(float)
            n_selected = int(np.sum(y_pred == 1))
            if n_selected == 0:
                continue
            k_pct = max(1.0, n_selected / len(y_true) * 100.0)
            current_lift = lift_at_k(y_true, y_prob, k_pct)
            if current_lift > best_lift:
                best_lift = current_lift
                best_thr = float(thr)
        return _metrics_at(best_thr)

    elif objective == "alert_budget_5pct":
        target_alert_rate = kwargs.get("target_alert_rate", 0.05)
        valid = sweep_df[sweep_df["alert_rate"] <= target_alert_rate]
        if valid.empty:
            raise ValueError(
                f"No threshold achieves alert_rate <= {target_alert_rate}. "
                f"Min alert_rate in sweep: {sweep_df['alert_rate'].min():.4f}"
            )
        idx = valid["threshold"].idxmax()
        return _metrics_at(float(valid.loc[idx, "threshold"]))

    elif objective == "alert_budget_10pct":
        target_alert_rate = kwargs.get("target_alert_rate", 0.10)
        valid = sweep_df[sweep_df["alert_rate"] <= target_alert_rate]
        if valid.empty:
            raise ValueError(
                f"No threshold achieves alert_rate <= {target_alert_rate}. "
                f"Min alert_rate in sweep: {sweep_df['alert_rate'].min():.4f}"
            )
        idx = valid["threshold"].idxmax()
        return _metrics_at(float(valid.loc[idx, "threshold"]))

    else:
        raise ValueError(
            f"Unknown objective: '{objective}'. "
            f"Supported: max_f1, f1, min_precision_30, min_recall_50, "
            f"top10_lift, alert_budget_5pct, alert_budget_10pct"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience wrappers used by backtest scripts
# ═══════════════════════════════════════════════════════════════════════════════

def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute classification metrics at a given threshold, plus ROC-AUC.

    Returns:
        Dict with keys: precision, recall, f1, roc_auc, threshold.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    y_pred = (y_prob >= threshold).astype(float)
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # ROC-AUC (requires >= 2 classes present)
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = 0.0

    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "roc_auc": round(roc_auc, 6),
        "threshold": round(float(threshold), 6),
    }


def compute_topk_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_pcts: list[float] | None = None,
) -> pd.DataFrame:
    """Compute top-k capture metrics, returning a DataFrame.

    The returned DataFrame uses column name ``topk_pct`` (not ``k_pct``)
    and ``lift_vs_random`` (not ``lift``) for compatibility with backtest
    aggregation code.

    Returns:
        DataFrame with columns: topk_pct, k_count, tp_captured,
        total_positives, recall_at_k, lift_vs_random, precision_at_k.
    """
    if k_pcts is None:
        k_pcts = [1, 3, 5, 10, 20]

    raw = top_k_capture(y_true, y_prob, k_pcts=k_pcts)

    # Rename columns for backtest compatibility.
    rename_map = {
        "k_pct": "topk_pct",
        "lift": "lift_vs_random",
    }
    result = raw.rename(columns=rename_map)
    return result


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """Compute calibration bucket table with ECE appended as a column.

    Returns:
        DataFrame with columns from ``calibration_bucket_table`` plus
        an ``ece`` column (same value in every row for convenience).
    """
    bucket_df = calibration_bucket_table(y_true, y_prob, n_buckets=n_buckets)
    ece = expected_calibration_error(y_true, y_prob, n_buckets=n_buckets)
    bucket_df["ece"] = ece
    return bucket_df
